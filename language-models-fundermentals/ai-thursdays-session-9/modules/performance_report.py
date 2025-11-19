from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd

import modules.configs as cfg

def generate_performance_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Generate a performance report i.e classification report.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Formatted performance report as a string
    """
    cr = classification_report(y_true, y_pred)
        
    report = cfg.LINE_DIVIDER + cfg.CHARRIAGE_RETURN
    report += "Classification Report:\n"
    report += cr
    report += cfg.CHARRIAGE_RETURN + cfg.LINE_DIVIDER + cfg.CHARRIAGE_RETURN    
    return report

def calculate_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate the Area Under the Receiver Operating Characteristic Curve (AUC-ROC).
    :param y_true: True labels
    :param y_scores: Predicted scores or probabilities
    :return: AUC-ROC value
    """    
    auc = roc_auc_score(y_true, y_scores)
    return auc

def generate_trading_strategy_performance_report(
        sp_data_df: pd.DataFrame,
        X_test_index: pd.Index,
        y_test: np.ndarray,
        y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Generate a performance report for trading strategy predictions. 
    :param sp_data_df: DataFrame containing S&P data with actual returns
    :param X_test_index: Index of the test data
    :param y_test: True labels for the test data
    :param y_pred: Predicted labels for the test data
    :return: DataFrame containing actual returns, predicted returns, and strategy returns
    """
    index_test_indexes = X_test_index.tolist()
    sp_data_test = sp_data_df[sp_data_df.index.isin(index_test_indexes)]

    results_df = pd.DataFrame({    
    'predict_price_direction': y_pred,
    'actual_price_direction': y_test,
    'return': sp_data_test['return'].values
    },
    index=X_test_index)

    results_df.predict_price_direction = results_df.predict_price_direction.map({1: 1, 0: -1})
    results_df.actual_price_direction = results_df.actual_price_direction.map({1: 1, 0: -1})
    results_df['strategy_return'] = results_df['predict_price_direction'] * results_df['return']
    results_df['actual_return'] = results_df['actual_price_direction'] * results_df['return']
    results_df['strategy_return_cum'] = results_df['strategy_return'].cumsum().apply(np.exp)
    results_df['actual_return_cum'] = results_df['actual_return'].cumsum().apply(np.exp)
    results_df.dropna(inplace=True)
    return results_df

