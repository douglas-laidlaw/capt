import numpy
from matplotlib import pyplot; pyplot.ion()

def transform_matrix(n_subap, n_wfs):
    """Generate transformation matrix used in L3S to mitigate ground-layer turbulence
    
    Parameters:
		n_subap (int): number of SHWFS sub-apertures.
		n_wfs (int): number of SHWFSs.
    
    Returns:
        ndarray: transformation matrix."""
    
    t_matrix = numpy.zeros((2* numpy.sum(n_subap), 2 * numpy.sum(n_subap)))#, dtype='float16')
    for i in range(n_wfs):
        wfs_n_subap = n_subap[i] 
        diag1 = numpy.arange(2*wfs_n_subap*n_wfs - i*2*wfs_n_subap)
        diag2 = numpy.arange(i*2*wfs_n_subap, 2*wfs_n_subap*n_wfs)
        if i==0:
            t_matrix[diag1,diag2]=(1-(1./n_wfs))
        else:
            t_matrix[diag1,diag2]=(-1./n_wfs)
    mirror = numpy.swapaxes(t_matrix,1,0)
    d = numpy.arange(2*n_wfs*wfs_n_subap)
    t_matrix = t_matrix + mirror
    t_matrix[d,d] = t_matrix[d,d]/2.
    return t_matrix

if __name__=='__main__':
    trans = transform_matrix(numpy.array([36, 36]),2)
