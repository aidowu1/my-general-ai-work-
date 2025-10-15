// predict_xgb_cpp.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>

// include XGBoost C API header (adjust include path when building)
#include <xgboost/c_api.h>

void check(int rc) {
    if (rc != 0) {
        // XGBoost C API returns non-zero on error
        std::cerr << "XGBoost C API error code: " << rc << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    const std::string csv_file = (argc > 1) ? argv[1] : "xgb_diabetes_test.csv";
    const std::string model_file = (argc > 2) ? argv[2] : "xgb_diabetes_model.json";

    // 1) Read CSV -> vector of rows
    std::ifstream ifs(csv_file);
    if (!ifs.is_open()) {
        std::cerr << "Cannot open file: " << csv_file << "\n";
        return 1;
    }

    std::string header;
    std::getline(ifs, header); // header line
    std::vector<std::string> cols;
    {
        std::istringstream hs(header);
        std::string token;
        while (std::getline(hs, token, ',')) {
            cols.push_back(token);
        }
    }
    const int ncols = (int)cols.size();
    if (ncols < 2) {
        std::cerr << "Expected at least one feature + target column\n";
        return 1;
    }
    const int feature_count = ncols - 1; // last col expected to be 'target'

    std::vector<float> feature_data; // row-major data for XGDMatrixCreateFromMat
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
            if (col_idx < feature_count) {
                rowvals[col_idx] = val;
            } else { // last column -> target
                target_val = val;
            }
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
    if (nrows == 0) {
        std::cerr << "No data rows found in CSV\n";
        return 1;
    }

    std::cout << "Loaded CSV: rows=" << nrows << " features=" << feature_count << "\n";

    // 2) Create DMatrix from dense matrix (row-major)
    DMatrixHandle dmat = nullptr;
    // missing value represented by NAN
    int rc = XGDMatrixCreateFromMat(feature_data.data(), (bst_ulong)nrows, (bst_ulong)feature_count, NAN, &dmat);
    check(rc);

    // 3) Create booster and load model
    BoosterHandle booster = nullptr;
    DMatrixHandle dmats_arr[1] = { dmat };
    rc = XGBoosterCreate(dmats_arr, 1, &booster);
    check(rc);

    rc = XGBoosterLoadModel(booster, model_file.c_str()); // JSON model file
    if (rc != 0) {
        std::cerr << "Failed to load model: " << model_file << std::endl;
        return 1;
    }

    // 4) Predict
    bst_ulong out_len = 0;
    const float* out_result = nullptr;
    // signature (booster, dmat, option_mask, ntree_limit, &out_len, &out_result)
    rc = XGBoosterPredict(booster, dmat, 0, 0, 1, &out_len, &out_result);
    check(rc);

    if (out_len != (bst_ulong)nrows) {
        std::cerr << "Prediction length mismatch: out_len=" << out_len << " nrows=" << nrows << "\n";
    }

    // 5) Compute RMSE and print first 10 results
    double se = 0.0;
    for (size_t i = 0; i < nrows; ++i) {
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

    // 6) Free resources
    XGBoosterFree(booster);
    XGDMatrixFree(dmat);

    return 0;
}
