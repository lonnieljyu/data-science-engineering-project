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
    Parses a json object, from the command line or a json file, and returns a dense numpy matrix.
    :param value:
    :return dense_matrix:
    """

    if os.path.isfile(value):
        with open(value, 'rb') as file:
            value = file.read()

    dense_matrix = np.array(json.loads(value))
    return dense_matrix


def get_matrix_from_csv(file_path):
    """
    Parses a csv file and returns a dense numpy matrix.
    :param file_path:
    :return dense_matrix:
    """

    dense_matrix = None
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as file:
            csv_list = [[int(feature) for feature in row] for row in csv.reader(file)]
            dense_matrix = np.array(csv_list)
    return dense_matrix


def get_matrix_from_pickle(file_path):
    """
    Parses a pickle file and returns a dense numpy matrix.
    :param file_path:
    :return dense_matrix:
    """

    dense_matrix = None
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as file:
            dense_matrix = np.array(pickle.loads(file.read()))
    return dense_matrix


def get_matrix_from_coo(file_path):
    """
    Parses a coo file and returns a sparse numpy matrix.
    :param file_path:
    :return sparse_matrix:
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


def parse_command_line_input(**kwargs):
    """
    Parses command line input and calls the relevant parsing function.
    Returns the is_sparse flag, the matrix, and the include_distance flag.
    :param kwargs:
    :return is_sparse, matrix, include_distance:
    """

    include_distance, input_type, value = False, None, None
    opts = [(k, v) for k, v in kwargs.items() if v is not None]
    for opt in opts:
        if opt[0] == 'distance':
            include_distance = True
        else:
            input_type, value = opt[0], opt[1]

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
