import sys
import numpy as np
from scipy.spatial.distance import pdist, euclidean
from sklearn.metrics.pairwise import pairwise_distances
import click
from utils.parsing import parse_options, parse_command_line_options


@click.group()
def main():
    """
    pymatrix: A command line tool for working with matrices.
    """
    pass


@click.command()
@click.argument('n', type=click.INT)
@click.option('-j', '--json-data', type=click.STRING,
              help='input matrix as a valid json object')
@click.option('-f', '--csv-file', type=click.Path(exists=True),
              help='read matrix from a csv file')
@click.option('-p', '--pickle-file', type=click.Path(exists=True),
              help='read matrix from a pickle file')
@click.option('-s', '--sparse-coo', type=click.Path(exists=True),
              help='read matrix in COO format from a file')
def echo(n, **kwargs):
    """
    Display the passed options N times
    """

    input_type, value = parse_options(**kwargs)

    for _ in range(n):
        click.echo(
            "\nThe given input was of type: %s\n"
            "And the value was: %s\n"
            % (input_type, value)
        )


@click.command()
@click.argument('row_i', type=click.INT)
@click.option('-j', '--json-data', type=click.STRING,
              help='input matrix as a valid json object')
@click.option('-f', '--csv-file', type=click.Path(exists=True),
              help='read matrix from a csv file')
@click.option('-p', '--pickle-file', type=click.Path(exists=True),
              help='read matrix from a pickle file')
@click.option('-s', '--sparse-coo', type=click.Path(exists=True),
              help='read matrix in COO format from a file')
@click.option('--distance', default=False, type=click.BOOL, is_flag=True,
              help='print the distance between the pair of rows')
def closest_to(row_i, **kwargs):
    """
    Find the row that is the minimal distance from row_i and
    optionally display the distance as well

    Output Format:\n
      i j [d_ij]
    """

    # Set metric, sorting, decimal precision parameters
    DISTANCE_METRIC = 'euclidean'
    SORT_TYPE = 'mergesort'
    DECIMAL_PRECISION = 5

    # Get matrix and include_distance parameters
    is_sparse, matrix, include_distance = parse_command_line_options(**kwargs)
    if matrix is None:
        print('Invalid matrix input.')
        return

    # Calculate the minimum row distance and the corresponding row index
    minimum_row_j, minimum_distance = -1, sys.maxint
    if not is_sparse:
        for row_j in xrange(matrix.shape[0]):
            if row_i == row_j: continue
            distance = euclidean(matrix[row_j], matrix[row_i])
            if distance < minimum_distance:
                minimum_row_j, minimum_distance = row_j, distance
    else:
        # Build pairwise distance matrix from the sparse matrix and sort by minimum distance
        paired_distances = pairwise_distances(matrix, metric=DISTANCE_METRIC)
        row_indices, column_indices = np.triu_indices(paired_distances.shape[0], 1)
        paired_distances = np.column_stack((row_indices, column_indices, paired_distances[row_indices, column_indices]))
        paired_distances = paired_distances[paired_distances[:,2].argsort(kind=SORT_TYPE)]

        # Select the relevant minimum row distance and the corresponding row index
        for i in xrange(paired_distances.shape[0]):
            if paired_distances[i][0] == row_i:
                minimum_row_j, minimum_distance = int(paired_distances[i][1]), paired_distances[i][2]
                break
            elif paired_distances[i][1] == row_i:
                minimum_row_j, minimum_distance = int(paired_distances[i][0]), paired_distances[i][2]
                break

    # Build and print output string
    output_string = ''
    if minimum_row_j < row_i:
        output_string += str(minimum_row_j) + ' ' + str(row_i) + ' '
    else:
        output_string += str(row_i) + ' ' + str(minimum_row_j) + ' '
    if include_distance:
        output_string += str(round(minimum_distance, DECIMAL_PRECISION))
    print output_string


@click.command()
@click.argument('n', type=click.INT)
@click.option('-j', '--json-data', type=click.STRING,
              help='input matrix as a valid json object')
@click.option('-f', '--csv-file', type=click.Path(exists=True),
              help='read matrix from a csv file')
@click.option('-p', '--pickle-file', type=click.Path(exists=True),
              help='read matrix from a pickle file')
@click.option('-s', '--sparse-coo', type=click.Path(exists=True),
              help='read matrix in COO format from a file')
@click.option('--distance', default=False, type=click.BOOL, is_flag=True,
              help='print the distance between the pair of rows')
def closest(n, **kwargs):
    """
    Find the N distinct pairs of rows that are the smallest distance
    apart and optionally display the distance as well

    Output Format:\n
      i j [d_ij]
    """
    # Set metric, sorting, decimal precision parameters
    DISTANCE_METRIC = 'euclidean'
    SORT_TYPE = 'mergesort'
    DECIMAL_PRECISION = 5

    # Get matrix and include_distance parameters
    is_sparse, matrix, include_distance = parse_command_line_options(**kwargs)
    if matrix is None:
        print('Invalid matrix input.')
        return

    # Build pairwise distance matrix and sort by minimum distance
    if not is_sparse:
        paired_distances = pdist(matrix, metric=DISTANCE_METRIC)
        row_indices, column_indices = np.triu_indices(matrix.shape[0], 1)
        paired_distances = np.column_stack((row_indices, column_indices, paired_distances))
        paired_distances = paired_distances[paired_distances[:, 2].argsort(kind=SORT_TYPE)]
    else:
        paired_distances = pairwise_distances(matrix)
        row_indices, column_indices = np.triu_indices(paired_distances.shape[0], 1)
        paired_distances = np.column_stack((row_indices, column_indices, paired_distances[row_indices, column_indices]))
        paired_distances = paired_distances[paired_distances[:, 2].argsort(kind=SORT_TYPE)]

    # Build and print output strings
    for i in xrange(n):
        output_string = str(int(paired_distances[i][0])) + ' ' + str(int(paired_distances[i][1])) + ' '
        if include_distance:
            output_string += str(round(paired_distances[i][2], DECIMAL_PRECISION))
        print output_string


@click.command()
@click.argument('n', type=click.INT)
@click.option('-j', '--json-data', type=click.STRING,
              help='input matrix as a valid json object')
@click.option('-f', '--csv-file', type=click.Path(exists=True),
              help='read matrix from a csv file')
@click.option('-p', '--pickle-file', type=click.Path(exists=True),
              help='read matrix from a pickle file')
@click.option('-s', '--sparse-coo', type=click.Path(exists=True),
              help='read matrix in COO format from a file')
@click.option('--distance', default=False, type=click.BOOL, is_flag=True,
              help='print the distance between the pair of rows')
def furthest(n, **kwargs):
    """
    Find the N distinct pairs of rows that are the furthest distance
    apart and optionally display the distance as well

    Output Format:\n
      i j [d_ij]
    """
    is_sparse, matrix, include_distance = parse_command_line_options(**kwargs)
    if matrix is None:
        print('Invalid matrix input.')
        return


@click.command()
@click.argument('n_centroids', type=click.INT)
@click.option('-j', '--json-data', type=click.STRING,
              help='input matrix as a valid json object')
@click.option('-f', '--csv-file', type=click.Path(exists=True),
              help='read matrix from a csv file')
@click.option('-p', '--pickle-file', type=click.Path(exists=True),
              help='read matrix from a pickle file')
@click.option('-s', '--sparse-coo', type=click.Path(exists=True),
              help='read matrix in COO format from a file')
def centroids(n_centroids, **kwargs):
    """
    Cluster the given data set and return the N centroids,
    one for each cluster
    """
    is_sparse, matrix = parse_command_line_options(**kwargs)
    if matrix is None:
        print('Invalid matrix input.')
        return

main.add_command(echo)
main.add_command(closest_to)
main.add_command(closest)
main.add_command(furthest)
main.add_command(centroids)
