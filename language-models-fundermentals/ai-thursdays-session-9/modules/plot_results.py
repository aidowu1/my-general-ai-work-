import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import confusion_matrix
from datetime import datetime  
import pandas as pd
import numpy as np  

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot the confusion matrix using seaborn heatmap.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param class_names: List of class names for labeling the axes
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def plot_strategy_performance(dates, prices, signals):
    """
    Plot the stock prices along with buy/sell signals.
    :param dates: List of dates
    :param prices: List of stock prices
    :param signals: List of signals (1 for buy, -1 for sell, 0 for hold)
    """
    plt.figure(figsize=(14, 7))
    plt.plot(dates, prices, label='Stock Price', color='blue')
    
    buy_signals = [prices[i] if signals[i] == 1 else None for i in range(len(signals))]
    sell_signals = [prices[i] if signals[i] == -1 else None for i in range(len(signals))]
    
    plt.scatter(dates, buy_signals, label='Buy Signal', marker='^', color='green', alpha=1)
    plt.scatter(dates, sell_signals, label='Sell Signal', marker='v', color='red', alpha=1)
    
    plt.title('Stock Price with Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_strategy_actual_vs_predicted(dates, actual, predicted):
    """
    Plot actual vs predicted stock price directions.
    :param dates: List of dates
    :param actual: List of actual directions
    :param predicted: List of predicted directions
    """
    plt.figure(figsize=(14, 7))

    #new_dates =  pd.Series(np.array([datetime.strptime(date, '%Y-%m-%d').date() for date in dates.tolist()]))
    plt.plot(dates, actual, label='Actual Direction', color='blue', alpha=0.7)
    plt.plot(dates, predicted, label='Predicted Direction', color='orange', alpha=0.7)
    year_ticks = plt.MaxNLocator(10)
    plt.gca().xaxis.set_major_locator(year_ticks)
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    
    plt.title('Actual vs Predicted Stock Price Directions')
    plt.xlabel('Date')
    plt.ylabel('Direction')
    plt.legend()
    plt.show()

def create_correlation_matrix(
        df: pd.DataFrame,
        target: str='demand',
        figsize=(9,0.5),
        ret_id=False):
    """
    Create and plot a correlation matrix heatmap for the features in the DataFrame.
    :param df: Input DataFrame
    :param target: Target variable to correlate against
    :param figsize: Figure size for the plot
    :param ret_id: If True, return the correlation DataFrame instead of plotting
    :return: Correlation DataFrame if ret_id is True, else None
    """
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    corr_mat = df.corr().round(2);shape = corr_mat.shape[0]
    corr_mat = corr_mat.transpose()
    corr = corr_mat.loc[:, df.columns == target].transpose().copy()
    
    if(ret_id is False):
        f, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr,vmin=-0.3,vmax=0.3,center=0, 
                     cmap=cmap,square=False,lw=2,annot=True,cbar=False)
        plt.title(f'Feature Correlation to {target}')
    
    if(ret_id):
        return corr
    