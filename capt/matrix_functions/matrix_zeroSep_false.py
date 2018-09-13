import numpy
from matplotlib import pyplot; pyplot.ion()


def matrix_zeroSep_false(n_subap):
    """Creates matrix of 1s and 0s, with 0s at covariance matrix 
    sub-aperture separations of zero.
    
    Parameters:
        n_subap (ndarray): number of sub-apertures within each SHWFS.
        
    Returns:
        ndarray: matrix of 1s and 0s."""
    
    n_wfs = len(n_subap)
    zero_diagonals = numpy.ones((2*n_subap[0]*n_wfs, 2*n_subap[0]*n_wfs))
    
    for i in range(n_wfs*2):
        diag1 = numpy.arange(n_subap[0]*n_wfs*2 - i*n_subap[0])
        diag2 = numpy.arange(i*n_subap[0], n_subap[0]*n_wfs*2)
        zero_diagonals[diag1, diag2] = 0

    mirror = numpy.swapaxes(zero_diagonals,1,0)
    zero_diagonals = zero_diagonals*mirror
    return zero_diagonals


if __name__=='__main__':
    m = matrix_zeroSep_false(2, numpy.array([36, 36]))
    pyplot.imshow(m)