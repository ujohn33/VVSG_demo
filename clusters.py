import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import dill as pickle

import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn import preprocessing
# K-mean clustering libraries
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import datetime
from datetime import date
from scipy.stats.stats import pearsonr
from src.group_ts_split import PurgedGroupTimeSeriesSplit

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cpadapter
from cpadapter.utils import train_cal_test_split
from cpadapter.visualization import conditional_band_interval_plot

target = ['Target']
input_dir = './data/'

# class Profile():
#     """
#     A class to create synthetic load profiles
#     """
#     def __init__(
#         self,
#         params: dict,
#     ):

#     self.params = params

@st.cache
def load_data():
    features = pd.read_csv(input_dir+'features.csv', index_col=0)
    metadata = pd.read_csv('data/EANLIJST_METADATA.csv', index_col=0, sep   = ';')
    # ADD the functietype column to the features
    features['function'] = metadata['Patrimonium Functietype']
    # read more metrics from csv
    metrics = pd.read_csv('data/ts_metrics.csv', usecols = ['ID', 'mean', 'std'], index_col='ID')
    # add the metrics to the features
    features = features.join(metrics)
    features.isnull().sum()
    features.dropna(inplace=True)
    features['ID'] = features.index
    return features

def load_params(data):
    params = {}
    params['Type of building'] = st.selectbox("Select the type of building:", data['function'].unique())
    return params

def make_test_df(params):
    df = pd.DataFrame()
    df['datetime'] = pd.date_range(start=params['s'], end=params['e'], freq='1H')
    df['month'] = pd.DatetimeIndex(df['datetime']).month
    df['weekday'] = pd.DatetimeIndex(df['datetime']).weekday
    df['hour'] = pd.DatetimeIndex(df['datetime']).hour

    labelencoder = preprocessing.LabelEncoder()

    for col in categoricals:
        df[col] = labelencoder.fit_transform(df[col])
        df[col] = df[col].astype('int')
    return df

def band_interval_plot(x, y: np.ndarray, lower: np.ndarray, upper: np.ndarray, conf_percentage: float, sort: bool) -> None:
    r"""Function used to plot the data in `y` and it's confidence interval
    This function plots `y`, with a line plot, and the interval defined by the
    `lower` and `upper` bounds, with a band plot.
    Parameters
    ----------
    y: numpy.ndarray
    Array of observation or predictions we want to plot
    lower: numpy.ndarray
    Array of lower bound predictions for the confidence interval
    upper: numpy.ndarray
    Array of upper bound predictions for the confidence interval
    conf_percetage: float
    The desired confidence level of the predicted confidente interval
    sort: bool
    Boolean variable that indicates if the data, and the respective lower
    and upper bound values should be sorted ascendingly.
    Returns
    -------
    None
    Notes
    -----
    This function must only be used for regression cases
    """
    if sort:
        idx = np.argsort(y)
        y = y[idx]
        lower = lower[idx]
        upper = upper[idx]
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(x, y.reshape(-1), label='data')
    conf = str(conf_percentage*100) + '%'
    ax.fill_between(x, lower, upper, label=conf, alpha=0.3)
    ax.legend()
    return fig
if __name__ == "__main__":
    st.title("Clustering")
    features = load_data()
    #params = load_params(21)


    # Define the number of clusters
    clust_num = st.slider('Number of clusters', min_value=2, max_value=10, value=3, key=21)

    subset = features.drop(['function', 'ID', 'mean', 'std'], axis=1).copy()
    # Scale all numerical features with standard scaler
    scaler = StandardScaler()
    subset = pd.DataFrame(scaler.fit_transform(subset), index=subset.index, columns=subset.columns)
    # K-mean clustering on the features cp
    kmeans = KMeans(n_clusters=clust_num, random_state=0).fit(subset)
    clusters = kmeans.labels_
    # Create two axis with PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(subset)
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }
    scat = subset.join(pd.DataFrame(components, index=features.index, columns=list(labels.values())))
    scat['ID'] = scat.index
    scat['cluster'] = clusters
    scat['mean'] = features['mean']
    scat['std'] = features['std']
    fig = px.scatter(scat, x=list(labels.values())[0], y=list(labels.values())[1], color='cluster', hover_name=features.function, hover_data=features[['ID', 'mean', 'std']])
    fig.update_traces(mode="markers")
    st.plotly_chart(fig)

    subset = features.drop(['function', 'ID', 'mean'], axis=1).copy()
    # Scale all numerical features with standard scaler
    scaler = StandardScaler()
    subset = pd.DataFrame(scaler.fit_transform(subset), index=subset.index, columns=subset.columns)
    subset = subset.join(features[['mean']])
    # Create two axis with PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(subset)
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }
    scat = subset.join(pd.DataFrame(components, index=features.index, columns=list(labels.values())))
    scat['ID'] = scat.index
    scat['cluster'] = clusters
    scat['function'] = features['function']
    fig = px.scatter(scat, x=list(labels.values())[0], y=list(labels.values())[1], color='cluster', hover_name=features.function, hover_data=features[['ID', 'mean', 'std']])
    fig.update_traces(mode="markers")
    st.plotly_chart(fig)

    counts = scat.groupby('cluster').function.value_counts().sort_values(ascending=False).unstack().T
    st.table(data = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index].fillna(0).astype(int))

    
    data = pd.read_csv('data/full.csv', index_col='datetime', parse_dates=True)
    #fig2 = go.Figure()
    
    cmap = cm.get_cmap('plasma')
    fig2, ax = plt.subplots()
    for k, clust in enumerate(range(clust_num)):
        id_list = scat[scat.cluster == clust].index
        data_clust = data[data.ID.isin(id_list)].loc[:, ['Target']]
        plot_data = data_clust.resample('D').mean()
        # rename column from Target to cluster
        plot_data.rename(columns={'Target': f'Cluster: {clust}'}, inplace=True)
        plot_data.plot(ax=ax, color=cmap( k / clust_num))
        # y axis label
        ax.set_ylabel('Share of Annual electricity consumption')
        # set legend
        ax.legend()
        #plot_data['cluster'] = clust
        # plotly plot 
        #fig2.add_trace(go.Scatter(x=plot_data.index, y=plot_data.Target, name=f'Cluster {k}',mode='lines', line=dict(color=cor)))
    st.pyplot(fig2)
    st.plotly_chart(fig2)

