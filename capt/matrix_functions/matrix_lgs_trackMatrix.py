import numpy
from matplotlib import pyplot; pyplot.ion()


def matrix_lgs_trackMatrix_locs(size, n_subaps):
	"""Creates a block matrix composed as three values. Number of sub-blocks
	is calculated by the integer multiple of size and subSize.

	Parameters:
		size (int): size of block matrix
		subSize (int): size of individual blocks
		values (ndarray): numpy array containing 2 values (xx, yy). 0 is assigned to xy.

	Returns:
		ndarray: track matrix."""

	lgs_trackMatrix = numpy.zeros((size, size))
	n_wfs = len(n_subaps)
	tracker = 1
	
	n1=0
	for i in numpy.arange(n_wfs):
		m1=0
		for j in range(i+1):
			n2 = n1 + 2 * n_subaps[i]
			m2 = m1 + 2 * n_subaps[j]

			lgs_trackMatrix[n1:n1+n_subaps[i], m1:m1+n_subaps[j]] = tracker
			tracker += 1

			lgs_trackMatrix[n1+n_subaps[i]:n2, m1+n_subaps[j]:m2] = tracker
			tracker += 1

			m1 += 2 * n_subaps[i]
		n1 += 2 * n_subaps[j]

	return mirror_covariance_matrix(lgs_trackMatrix, n_subaps)


def mirror_covariance_matrix(cov_mat, n_subaps):
	"""Mirrors a covariance matrix around the diagonal.

	Parameters:
		cov_mat (ndarray): The covariance matrix to mirror.
		n_subaps (ndarray): number of sub-apertures within each SHWFS.

	Returns:
		ndarray: complete covariance matrix."""

	total_slopes = cov_mat.shape[0]
	n_wfs = len(n_subaps)

	n1 = 0
	for n in range(n_wfs):
		m1 = 0
		for m in range(n + 1):
			if n != m:
				n2 = n1 + 2 * n_subaps[n]
				m2 = m1 + 2 * n_subaps[m]

				cov_mat[m1: m2, n1: n2] = (numpy.swapaxes(cov_mat[n1: n2, 
					m1: m2], 1, 0))

				m1 += 2 * n_subaps[m]
		n1 += 2 * n_subaps[n]
	return cov_mat


if __name__ == '__main__':
	size = 216
	n_subap = numpy.array([36]*3)

	m = matrix_lgs_trackMatrix_locs(size, n_subap)
	# m2 = mirror_covariance_matrix(m, n_subap)
	pyplot.figure('track matrix')
	pyplot.imshow(m, interpolation='nearest')
