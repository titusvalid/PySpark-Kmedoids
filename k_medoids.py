'''
Light Weight K-Medoids Spark Implementation by Chus
'''

from pyspark import SparkContext, RDD
from pyspark.sql import SparkSession
import numpy as np
from copy import deepcopy
from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors, VectorUDT


def example_distance_func(data1, data2):
    return float(np.linalg.norm(np.array(data1) - np.array(data2)))

def find_new_center(cluster_points):
    cluster_list = list(cluster_points)
    return min(cluster_list, key=lambda x: np.sum([example_distance_func(x, y) for y in cluster_list]))

def kmedoids_run_spark(rdd: RDD, n_clusters, dist_func, max_iter=1000, tol=0.001, sc=None):
    print("kmedoids_run_spark")
    indexed_rdd = rdd.filter(lambda x: len(x) > 0).zipWithIndex().map(lambda x: (x[1], x[0]))

    print("Sample of indexed_rdd after zipWithIndex:", indexed_rdd.take(10))
    
    n_samples = indexed_rdd.count()
    print("Number of samples:", n_samples)
    
    centers_idx = np.random.choice(n_samples, n_clusters, replace=False).tolist()
    print("Center indices:", centers_idx)

    centers_data = indexed_rdd.filter(lambda x: x[0] in centers_idx).collect()
    centers_data = [x[1] for x in centers_data]
    print("Initial Centers:", centers_data)
    broadcast_centers = sc.broadcast(centers_data)

    empty_rdd = indexed_rdd.filter(lambda x: len(x[1]) == 0)
    print("Empty RDD count:", empty_rdd.count())

    print("kmedoids_run_spark 2")
    for _ in range(max_iter):
        cost_rdd = indexed_rdd.map(lambda x: (x[0], min([(i, dist_func(x[1], center)) for i, center in enumerate(broadcast_centers.value)], key=lambda t: t[1])))
        print("kmedoids_run_spark 3")
        new_centers_data = []
        for i in range(n_clusters):
            cluster_rdd = cost_rdd.filter(lambda x: x[1][0] == i).map(lambda x: x[0])
            print("cluster_rdd:")
            print("Count of cluster_rdd:", cluster_rdd.count())
            print("Elements in cost_rdd:", cost_rdd.take(10))
            print("Elements in cluster_rdd:", cluster_rdd.take(10))
            print("Sample of indexed_rdd:", indexed_rdd.take(10))

            pair_cluster_rdd = cluster_rdd.map(lambda x: (x, None))
            cluster_data = indexed_rdd.join(pair_cluster_rdd).map(lambda x: x[1][0])


            cluster_data.foreach(print)
            print("kmedoids_run_spark 4")
            first_five = cluster_data.collect()[:10]
            print("First 50 cluster_data:", first_five)

            sum_cluster_data = cluster_data.reduce(lambda x, y: np.array(x) + np.array(y))
            new_center = cluster_data.reduce(lambda a, b: a if dist_func(a, sum_cluster_data) < dist_func(b, sum_cluster_data) else b)
            new_centers_data.append(new_center)

        if all(dist_func(a, b) <= tol for a, b in zip(centers_data, new_centers_data)):
            break

        centers_data = new_centers_data
        broadcast_centers = sc.broadcast(centers_data)

    labeled_rdd = indexed_rdd.map(lambda x: (x[0], min([(i, dist_func(x[1], center)) for i, center in enumerate(broadcast_centers.value)], key=lambda t: t[1])[0]))
    
    return centers_data, labeled_rdd

class SparkKMedoids:
    def __init__(self, n_clusters, dist_func=example_distance_func, max_iter=1000, tol=0.001, sc=None):
        print("SparkKMedoids")
        self.n_clusters = n_clusters
        self.dist_func = dist_func
        self.max_iter = max_iter
        self.tol = tol
        self.sc = sc

    def fit(self, rdd: RDD):
        self.centers_, self.labeled_rdd_ = kmedoids_run_spark(rdd, self.n_clusters, self.dist_func, self.max_iter, self.tol, self.sc)
        return self

    def predict(self, df, feature_col="features"):
        raise NotImplementedError()
        

    '''
    Main API of KMedoids Clustering

    Parameters
    --------
        n_clusters: number of clusters
        dist_func : distance function
        max_iter: maximum number of iterations
        tol: tolerance


    Methods
    -------
        fit(X): fit the model
            - X: 2-D numpy array, size = (n_sample, n_features)

        predict(X): predict cluster id given a test dataset.
    '''
