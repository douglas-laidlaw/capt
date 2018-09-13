import numpy

def tt_removal_matrix(n_subap, n_wfs):
	"""Generates matrix to remove tip-tilt from SHWFS centroids and/or covariance matrix.
	
	Parameters:
		n_subap (int): number of sub-apertures with each SHWFS (works for symmetrical SHWFSs).
		n_wfs (int): number of GSs.
		
	Returns:
		ndarray: matrix to remove tip-tilt."""

	tt_mat = numpy.ones((n_subap, n_subap)) * (-1./n_subap)
	diag = numpy.arange(n_subap)
	tt_mat[diag, diag] = 1 - (1./n_subap)

	remove_tt_mat = numpy.zeros((2 * n_subap * n_wfs, 2 * n_subap * n_wfs))
	for i in range(2*n_wfs):
		remove_tt_mat[i*n_subap:(i+1)*n_subap, i*n_subap:(i+1)*n_subap] = tt_mat

	return remove_tt_mat