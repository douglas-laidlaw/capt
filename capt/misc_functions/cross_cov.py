import numpy

def cross_cov(shwfs_centroids):
	"""Calculates the cross-covariance of SHWFS centroid measurements.

	Parameters:
		shwfs_centroids (ndarray): SHWFS centroid measurments.

	Returns:
		ndarray: covariance matrix of SHWFS centroid measurements."""

	covData = numpy.zeros((shwfs_centroids.shape[1], shwfs_centroids.shape[0]))
	for i in range(covData.shape[0]):
		covData[i] = shwfs_centroids[:,i]
	cov_matrix = numpy.cov(covData)
	return cov_matrix
