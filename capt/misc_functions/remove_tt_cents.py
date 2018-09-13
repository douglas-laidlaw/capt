import numpy
from astropy.io import fits
from capt.misc_functions.cross_cov import cross_cov
from matplotlib import pyplot; pyplot.ion()
from capt.misc_functions.make_pupil_mask import make_pupil_mask
from capt.misc_functions.mapping_matrix import get_mappingMatrix
from capt.map_functions.covMap_fromMatrix import covMap_fromMatrix




def remove_tt_cents(shwfs_centroids, n_subap, n_wfs):
	"""Removes tip-tilt from SHWFS centroids.
	
	Parameters:
		shwfs_centroids (ndarray): shwfs centroid positions in x and y over some time interval.
		n_subap (ndarray): sub-apertures within each shwfs.
		n_wfs (int): number of GSs.
	
	Returns:
		ndarray: shwfs centroid positions with tip-tilt removed.
		ndarray: matrix used to remove tip-tilt from shwfs centroid positions."""

	n_subap = n_subap[0]
	remove_tt_mat = remove_tt_matrix(n_subap, n_wfs)
	
	shwfs_centroids_tt_removed = numpy.matmul(numpy.matmul(remove_tt_mat, 
			shwfs_centroids.T).T, remove_tt_mat.T)

	# for i in range(n_wfs*2):		
		# shwfs_centroids[:, i*n_subap: (i+1)*n_subap] -= numpy.mean(shwfs_centroids[:, 
		# i*n_subap: (i+1)*n_subap], 1).reshape(shwfs_centroids.shape[0], 1)

	return shwfs_centroids_tt_removed, remove_tt_mat



def remove_tt_matrix(n_subap, n_wfs):
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




if __name__=='__main__':
	n_subap = numpy.array([36,36])
	nx_subap = numpy.array([7, 7])
	obs_diam = 1.
	tel_diam = 4.2
	n_wfs = 2

	cents = fits.getdata('testFitsFiles/canary_nl0_h0m_it10k_r00p1_L025m_gspos0cn20a0c20.fits')
	pupil_mask = make_pupil_mask('circle', n_subap, nx_subap[0], obs_diam, tel_diam)
	matrix_region_ones = numpy.ones(n_subap)
	mm, mmc, md = get_mappingMatrix(pupil_mask, matrix_region_ones)
	mat_f = cross_cov(cents)
	cov_map = covMap_fromMatrix(mat_f, n_wfs, nx_subap, n_subap, pupil_mask, 'x and y', mm, mmc, md)
	
	pyplot.figure('fresh')
	pyplot.imshow(cov_map)
	pyplot.figure('fresh xx')
	pyplot.plot(numpy.arange(-6,7), cov_map[:, 19]/cov_map[:, 19].max())
	
	
	t, tt_matrix = remove_tt_cents(cents, numpy.array([36, 36]),2)
	mat = cross_cov(t)
	cov_map = covMap_fromMatrix(mat, n_wfs, nx_subap, n_subap, pupil_mask, 'x and y', mm, mmc, md)
	
	pyplot.figure('removed')
	pyplot.imshow(cov_map)
	pyplot.figure('removed xx')
	pyplot.plot(numpy.arange(-6,7), cov_map[:, 19]/cov_map[:, 19].max())

	pyplot.figure()
	pyplot.imshow(tt_matrix)