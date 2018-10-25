from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score

#X - data points to cluster
#clusterer - sklearn clustering module
#min_clusters (optional) - starting number of clusters
#max_clusters (optional) - ending number of clusters (inclusive)
def cluster_test(X,clusterer,min_clusters=0,max_clusters=10):
    for i in range(min_clusters,max_clusters+1):
        clusterer.set_params(n_clusters = i)
        labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X,labels)
        print('For n_clusters =', i, "The average silhouette score is: ", silhouette_avg)   

    