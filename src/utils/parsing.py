import os
import json
import csv
import pickle
import numpy as np
from scipy.sparse import coo_matrix

"""
Util functions for parsing data inputs
"""


def parse_options(**kwargs):
    opts = [(k, v) for k, v in kwargs.items() if v is not None]
    if len(opts) != 1:
        raise ValueError("Exactly one input data type must be provided, got %d" % (len(opts)))
    return opts[0]


def get_matrix_from_json(value):
    """
    Parses a dense matrix from a json object.
    :param string value: json object or json file path

    :return dense_matrix: dense matrix representation
    :rtype dense_matrix: numpy.array
    """

    if os.path.isfile(value):
        with open(value, 'rb') as file:
            value = file.read()

    dense_matrix = np.array(json.loads(value))
    return dense_matrix


def get_matrix_from_csv(file_path):
    """
    Parses a dense matrix from a csv file.
    :param string file_path: local file path

    :return dense_matrix: dense matrix representation
    :rtype dense_matrix: numpy.array
    """

    dense_matrix = None
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as file:
            csv_list = [[int(feature) for feature in row] for row in csv.reader(file)]
            dense_matrix = np.array(csv_list)
    return dense_matrix


def get_matrix_from_pickle(file_path):
    """
    Parses a dense matrix from a pickle file.
    :param string file_path: local file path

    :return dense_matrix: dense matrix representation
    :rtype dense_matrix: numpy.array
    """

    dense_matrix = None
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as file:
            dense_matrix = np.array(pickle.loads(file.read()))
    return dense_matrix


def get_matrix_from_coo(file_path):
    """
    Parses a sparse matrix from a coo file.
    :param string file_path: local file path

    :return sparse_matrix: sparse matrix representation
    :rtype sparse_matrix: scipy.sparse.coo_matrix
    """

    sparse_matrix = None
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as file:
            row_indices, column_indices, data = list(), list(), list()
            for row in file:
                features = row.split()
                row_indices.append(int(features[0]))
                column_indices.append(int(features[1]))
                data.append(int(features[2]))
            sparse_matrix = coo_matrix((data, (row_indices, column_indices)))
    return sparse_matrix


def parse_command_line_options(**kwargs):
    """
    Parses matrix and command parameters from command-line options.
    :param dict kwargs: command-line options as keyword, value pairs

    :return bool is_sparse: flag stating if matrix is sparse

    :return matrix: dense or sparse matrix representation
    :rtype matrix: numpy.array or scipy.sparse.coo_matrix

    :return bool include_distance: flag stating if command returns distance
    """
    include_distance, input_type, value = False, None, None
    opts = [(k, v) for k, v in kwargs.items() if v is not None]
    for opt in opts:
        if opt[0] == 'distance':
            include_distance = True
        else:
            input_type, value = opt[0], str(opt[1])

    is_sparse, matrix = False, None
    if input_type == 'json_data':
        matrix = get_matrix_from_json(value)
    elif input_type == 'csv_file':
        matrix = get_matrix_from_csv(value)
    elif input_type == 'pickle_file':
        matrix = get_matrix_from_pickle(value)
    elif input_type == 'sparse_coo':
        is_sparse = True
        matrix = get_matrix_from_coo(value)
    return is_sparse, matrix, include_distance
