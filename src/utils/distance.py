import sys
import numpy as np
from scipy.spatial.distance import pdist, euclidean
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans

"""
Utility functions for calculating pairwise distances
"""

# Define distance metric, sorting type, decimal precision constant variables
DISTANCE_METRIC = 'euclidean'
SORT_TYPE = 'mergesort'
DECIMAL_PRECISION = 5


def get_pairwise_distances_matrix(is_sparse, matrix):
    """
    Calculates and sorts the pairwise distance matrix by distance.
    :param bool is_sparse: flag stating if matrix is sparse

    :param matrix: dense or sparse matrix representation
    :type matrix: numpy.array or scipy.sparse.coo_matrix

    :return paired_distances: matrix of pairwise distances
    :rtype paired_distances: numpy.array
    """
    if not is_sparse:
        paired_distances = pdist(matrix, metric=DISTANCE_METRIC)
        row_indices, column_indices = np.triu_indices(matrix.shape[0], 1)
        paired_distances = np.column_stack((row_indices, column_indices, paired_distances))
    else:
        paired_distances = pairwise_distances(matrix, metric=DISTANCE_METRIC)
        row_indices, column_indices = np.triu_indices(paired_distances.shape[0], 1)
        paired_distances = np.column_stack((row_indices, column_indices, paired_distances[row_indices, column_indices]))
    paired_distances = paired_distances[paired_distances[:, 2].argsort(kind=SORT_TYPE)]
    return paired_distances


def get_row_minimal_to_target_row(is_sparse, matrix, row_i):
    """
    Finds row with minimal distance to the target row
    :param bool is_sparse: flag stating if matrix is sparse

    :param matrix: dense or sparse matrix representation
    :type matrix: numpy.array or scipy.sparse.coo_matrix

    :param int row_i: target row index

    :return int minimum_row_j: row index of row with minimal distance to the target row

    :return float minimum_distance: the distance between the two rows
    """
    minimum_row_j, minimum_distance = -1, sys.maxint
    if not is_sparse:
        for row_j in xrange(matrix.shape[0]):
            if row_i == row_j:
                continue
            distance = euclidean(matrix[row_j], matrix[row_i])
            if distance < minimum_distance:
                minimum_row_j, minimum_distance = row_j, distance
    else:
        paired_distances = get_pairwise_distances_matrix(is_sparse, matrix)
        for i in xrange(paired_distances.shape[0]):
            if paired_distances[i][0] == row_i:
                minimum_row_j, minimum_distance = int(paired_distances[i][1]), paired_distances[i][2]
                break
            elif paired_distances[i][1] == row_i:
                minimum_row_j, minimum_distance = int(paired_distances[i][0]), paired_distances[i][2]
                break
    return minimum_row_j, minimum_distance


def get_row_pair_output(paired_distance, include_distance):
    """
    Concatenates row pair indices and optionally the distance
    :param paired_distance: array containing the row pair indices and distance
    :type paired_distance: numpy.array

    :param bool include_distance: flag stating if command returns distance

    :return string output_string: string representation of row pair indices and optionally the distance
    """
    output_string = str(int(paired_distance[0])) + ' ' + str(int(paired_distance[1])) + ' '
    if include_distance:
        output_string += str(round(paired_distance[2], DECIMAL_PRECISION))
    return output_string


def get_centroids(matrix, n_centroids):
    """
    Gets the centroids after clustering the matrix with k-means
    :param matrix: dense or sparse matrix representation
    :type matrix: numpy.array or scipy.sparse.coo_matrix

    :param int n_centroids: target number of centroids

    :return kmeans.cluster_centers_: matrix of the centroid vectors
    :rtype kmeans.cluster_centers_: numpy.array
    """
    kmeans = KMeans(n_clusters=n_centroids).fit(matrix)
    return kmeans.cluster_centers_
