import numpy
from matplotlib import pyplot; pyplot.ion()


def track_matrix(size, subSize, values):
    """Creates a block matrix composed as three values. Number of sub-blocks
    is calculated by the integer multiple of size and subSize.

    Parameters:
        size (int): size of block matrix
        subSize (int): size of individual blocks
        values (ndarray): numpy array containing 2 values (xx, yy). 0 is assigned to xy.

    Returns:
        ndarray: track matrix."""

    track_matrix = numpy.zeros((size, size))
    ints = int(size/subSize)

    for i in range(ints):
        l = (i+1)*subSize

        for j in range(ints):
            h = (j+1)*subSize

            if i%2==0 and j%2==0:
                track_matrix[i*subSize:l, j*subSize:h] = values[0]

            if i%2==1 and j%2==1:
                track_matrix[i*subSize:l, j*subSize:h] = values[1]

    return track_matrix


if __name__ == '__main__':
    size = 144
    subSize = 36
    values = numpy.array((1,2))

    m = track_matrix(size, subSize, values)
    pyplot.figure('track matrix')
    pyplot.imshow(m, interpolation='nearest')
