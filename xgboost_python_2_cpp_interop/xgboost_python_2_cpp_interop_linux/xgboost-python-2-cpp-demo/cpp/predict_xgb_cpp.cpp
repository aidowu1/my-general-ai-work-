// predict_xgb_cpp.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <xgboost/c_api.h>

void check(int rc) {
    if (rc != 0) {
        std::cerr << "XGBoost C API error code: " << rc << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    const std::string csv_file = (argc > 1) ? argv[1] : "xgb_diabetes_test.csv";
    const std::string model_file = (argc > 2) ? argv[2] : "xgb_diabetes_model.json";

    std::ifstream ifs(csv_file);
    if (!ifs.is_open()) {
        std::cerr << "Cannot open file: " << csv_file << "\n";
        return 1;
    }

    std::string header;
    std::getline(ifs, header);
    std::vector<std::string> cols;
    {
        std::istringstream hs(header);
        std::string token;
        while (std::getline(hs, token, ',')) cols.push_back(token);
    }
    int ncols = (int)cols.size();
    if (ncols < 2) {
        std::cerr << "Expected at least one feature + target column\n";
        return 1;
    }
    int feature_count = ncols - 1;

    std::vector<float> feature_data;
    std::vector<float> target;
    std::string line;
    size_t nrows = 0;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string cell;
        int col_idx = 0;
        std::vector<float> rowvals(feature_count);
        float target_val = 0.0f;
        while (std::getline(ss, cell, ',')) {
            float val = std::stof(cell);
            if (col_idx < feature_count) rowvals[col_idx] = val;
            else target_val = val;
            col_idx++;
        }
        if (col_idx != ncols) {
            std::cerr << "Unexpected number of columns in row " << nrows << "\n";
            return 1;
        }
        for (int j = 0; j < feature_count; ++j) feature_data.push_back(rowvals[j]);
        target.push_back(target_val);
        ++nrows;
    }
    ifs.close();

    std::cout << "Loaded CSV: rows=" << nrows << " features=" << feature_count << "\n";

    DMatrixHandle dmat = nullptr;
    int rc = XGDMatrixCreateFromMat(feature_data.data(), (bst_ulong)nrows, (bst_ulong)feature_count, NAN, &dmat);
    check(rc);

    BoosterHandle booster = nullptr;
    DMatrixHandle dmats_arr[1] = { dmat };
    rc = XGBoosterCreate(dmats_arr, 1, &booster);
    check(rc);

    rc = XGBoosterLoadModel(booster, model_file.c_str());
    check(rc);

    bst_ulong out_len = 0;
    const float* out_result = nullptr;
    rc = XGBoosterPredict(booster, dmat, 0, 0, &out_len, &out_result);
    check(rc);

    double se = 0.0;
    for (size_t i = 0; i < nrows && i < out_len; ++i) {
        double pred = (double)out_result[i];
        double gt = (double)target[i];
        double diff = pred - gt;
        se += diff * diff;
        if (i < 10) {
            std::cout << "row " << i << " pred=" << pred << " target=" << gt << "\n";
        }
    }
    double rmse = std::sqrt(se / (double)nrows);
    std::cout << "C++ inference RMSE: " << rmse << "\n";

    XGBoosterFree(booster);
    XGDMatrixFree(dmat);

    return 0;
}
