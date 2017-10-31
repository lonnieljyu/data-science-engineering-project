import click
from utils.parsing import parse_options, parse_command_line_options
from utils.distance import get_pairwise_distances_matrix, get_row_pair_output, get_row_minimal_to_target_row, \
    get_centroids


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

    is_sparse, matrix, include_distance = parse_command_line_options(**kwargs)
    if matrix is None:
        print('Invalid matrix input.')
        return
    last_row_index = max(matrix.row) if is_sparse else matrix.shape[0] - 1
    if row_i > last_row_index:
        print('Invalid row_i input: ' + str(row_i))
        return

    minimum_row_j, minimum_distance = get_row_minimal_to_target_row(is_sparse, matrix, row_i)
    if minimum_row_j < row_i:
        paired_distance = [minimum_row_j, row_i, minimum_distance]
    else:
        paired_distance = [row_i, minimum_row_j, minimum_distance]
    output_string = get_row_pair_output(paired_distance, include_distance)
    print(output_string)


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

    is_sparse, matrix, include_distance = parse_command_line_options(**kwargs)
    if matrix is None:
        print('Invalid matrix input.')
        return

    paired_distances = get_pairwise_distances_matrix(is_sparse, matrix)
    for i in xrange(min(n, paired_distances.shape[0])):
        output_string = get_row_pair_output(paired_distances[i], include_distance)
        print(output_string)


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

    paired_distances = get_pairwise_distances_matrix(is_sparse, matrix)
    last_row_index = paired_distances.shape[0] - 1
    for i in xrange(last_row_index, max(-1, last_row_index - n), -1):
        output_string = get_row_pair_output(paired_distances[i], include_distance)
        print(output_string)


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
    is_sparse, matrix, include_distance = parse_command_line_options(**kwargs)
    if matrix is None:
        print('Invalid matrix input.')
        return
    number_of_rows = max(matrix.row) + 1 if is_sparse else matrix.shape[0]
    if n_centroids <= 0 or n_centroids > number_of_rows:
        print('Invalid n_centroids input: ' + str(n_centroids))
        return

    cluster_centers = get_centroids(matrix, n_centroids)
    for i in xrange(cluster_centers.shape[0]):
        print cluster_centers[i]


main.add_command(echo)
main.add_command(closest_to)
main.add_command(closest)
main.add_command(furthest)
main.add_command(centroids)
