# IMPORTS
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

np.random.seed(42)

# FUNCTIONS

# make a function to get inertia scores for KMeans with varying values for n_clusters
def get_KMeans_inertia_scores(df, columns, max_n_clusters=10):
    """
    This function will
    - accept a dataframe, list of columns used to cluster, and a max value for number of clusters
    - make/fit/use a KMeans model
    - visualize number of clusters vs inertia
    - return dataframe with number of clusters and inertia scores
    """
    # initialize the dataframe we're going to send into KMeans
    X = df[columns]
    
    # initialize a list to capture number_of_clusters and inertia_
    inertia_list = []

    # fill up the inertia_list with data
    for k in range(2,max_n_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia_list.append([k, kmeans.inertia_])
    
    # make the list into a dataframe
    inertia_df = pd.DataFrame(inertia_list, columns=['n_clusters','inertia'])

    # plot the data
    plt.plot(inertia_df.n_clusters, inertia_df.inertia)
    plt.grid()
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia score')
    plt.title(f'Inertia score with {columns}')
    plt.show()
    
    return 

# defining a function to plot clusters, hued on a cluster prediction
def get_cluster_plot(df, x_col, y_col, cluster_col, centroids):
    """
    This function will
    - accept a dataframe with at least 3 columns: x_col, y_col, cluster_col
    - accept a centroids dataframe that has the centroid point for each cluster (0,1,...)
      in cluster_col
    - make a scatter plot with hue=cluster_col
    """
    
    sns.scatterplot(df[x_col], df[y_col], hue=df[cluster_col])
    plt.scatter(centroids[x_col], centroids[y_col], marker='x', c='black', s=500)
    plt.title('Data with cluster prediction hued')
    plt.show()
    return

def get_multiple_KMeans_cluster_plots(df, columns, x_col, y_col, max_n_clusters=4):
    """
    This function will
    - accept a dataframe, list of columns used to cluster, and a max value for number of clusters
    - make/fit/use several KMeans model with n_clusters varying from 2 to max_n_clusters
    - visualize with get_cluster_plots
    """    
    # initialize the dataframe we're going to send into KMeans
    X = df[columns]
    
    # fill up the inertia_list with data
    for k in range(2,max_n_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X[columns])
        X['clusters'] = kmeans.predict(X[columns])
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=[x_col, y_col])
        get_cluster_plot(X, x_col, y_col, 'clusters', centroids)
        
    return