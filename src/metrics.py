import numpy as np
import pandas as pd

def hhi(x):
    """
    Herfindhal-Hirschman Index

    x: numpy.ndarray counts of unit occurrences in time period
    """
    return np.square(x/x.sum()).sum()

def gini(x):
    """
    Gini Coefficient

    x: numpy.ndarray counts of unit occurrences in time period
    """
    if isinstance(x, pd.Series):
        x = x.values
    diffs = x.reshape((x.size), 1) - x
    return np.sum(np.abs(diffs)) / (2 * x.size**2 * x.mean())
    
def entropy(x):
    """
    Entropy

    x: numpy.ndarray counts of unit occurrences in time period
    """
    x = x[x>0]
    return -1 * np.sum(x/x.sum() * np.log2(x/x.sum()))

def nakamoto(x):
    """
    Nakamoto Coefficient

    x: numpy.ndarray counts of unit occurrences in time period
    """
    return np.nonzero(np.cumsum(np.flip(np.sort(x)) / x.sum()) >= 0.51)[0][0] + 1

def kld(x):
    """
    Kullback-Leibler Divergence

    x: numpy.ndarray counts of unit occurrences in time period
    """
    x = x[x>0]
    return np.log2(x.size) - np.sum(-1 * (x/x.sum()) * np.log2(x/x.sum()))

def nodes(x):
    """
    Nodes: the number of nodes

    x: numpy.ndarray counts of unit occurrences in time period
    """
    return x.size

def compute_metrics(df, func, group_column='date', value_column='count'):
    """
    Compute metrics for a given dataframe.

    df: pandas.DataFrame
    func: function to compute metrics
    """
    return (
        df.groupby(group_column)
        .agg({value_column: lambda x: func(x.to_numpy())})
        .rename({value_column: "value"}, axis=1)
        .reset_index()
    )
