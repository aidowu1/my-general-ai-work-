import pandas as pd
import numpy as np  


def compute_momenturm_indicator(
        df: pd.DataFrame,
        n: int) -> pd.Series:
    """
    Calculate the Momentum (MOM) indicator.
    :param df: DataFrame with 'close' price column  
    :param n: Number of periods for momentum calculation
    :return: Series representing the Momentum indicator 
    """     
    return pd.Series(df.diff(n), name='momentum_' + str(n))  

def compute_stochastic_oscilator(
        close: pd.Series, 
        low: pd.Series, 
        high: pd.Series, 
        n: int,
        id: int) -> pd.Series:
    """
    Calculate the Stochastic Oscillator (STOK) indicator.
    :param close: Series of closing prices
    :param low: Series of low prices
    :param high: Series of high prices
    :param n: Number of periods for stochastic calculation
    :param id: If 0, return %K line; if 1, return %D line (3-period SMA of %K)
    :return: Series representing the Stochastic Oscillator indicator
    """
    stok = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    if(id is 0):
        return stok
    else:
        return stok.rolling(3).mean()
    
def compute_rsi(
        df: pd.Series,
        period: int) -> pd.Series:
    """ 
    Calculate the Relative Strength Index (RSI) indicator.
    :param df: Series of closing prices
    :param period: Number of periods for RSI calculation
    :return: Series representing the RSI indicator
    """
    delta = df.diff().dropna()
    u = delta * 0; d = u.copy()
    u[delta > 0] = delta[delta > 0]; d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = u.ewm(com=period-1, adjust=False).mean() / d.ewm(com=period-1, adjust=False).mean()
    return 100 - 100 / (1 + rs)