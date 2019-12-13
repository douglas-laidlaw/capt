import numpy
import itertools
from scipy.special import comb
from capt.misc_functions.mapping_matrix import covMap_superFast
from capt.misc_functions.make_pupil_mask import make_pupil_mask
from capt.misc_functions.mapping_matrix import get_mappingMatrix
from matplotlib import pyplot; pyplot.ion()

def covMap_fromMatrix(cov_matrix, n_wfs, nx_subap, n_subap, pupil_mask, roi_axis, mm, mmc, md):
	"""Calculate covariance map from a given covariance matrix. Stacks all SHWFS combinations 
	into a single covariance map array.
	
	Parameters:
		cov_matrix (ndarray): covariance matrix.
		n_wfs (int): number of SHWFSs.
		nx_subap (int): number of sub-apertures across telescope's pupil.
		n_subap (ndarray): number of SHWFS sub-apertures.
		pupil_mask (ndarray): mask of SHWFS sub-apertures within the telescope's pupil.
		roi_axis (str): in which axis/axes to make covariance map. 'x', 'y', 'x+y' or 'x and y'.
		mm (ndarray): Mapping Matrix.
		mmc (ndarray): Mapping Matrix coordinates.
		md (ndarray): covariance map sub-aperture baseline pairing density.
	
	Returns:
		ndarray: covariance map"""

	if roi_axis=='x and y' or roi_axis=='x+y':
		nAxes = numpy.array([0,1])
		mapPlace = numpy.array([0,1])
	if roi_axis=='x':
		nAxes = numpy.array([0])
		mapPlace = numpy.array([0,0])
	if roi_axis=='y':
		nAxes = numpy.array([1])
		mapPlace = numpy.array([0,0])

	cov_mapDim = (2*int(nx_subap[0]))-1
	combs = comb(n_wfs, 2, exact=True)
	cov_map = numpy.zeros((combs, cov_mapDim,  len(nAxes)*cov_mapDim), dtype='float64')
	selector = numpy.array((range(n_wfs)))
	selector = numpy.array((list(itertools.combinations(selector, 2))))
	choice = numpy.array([0,1])

	for i in range(combs):
		wfs1_nx_subap = nx_subap[selector[i,0]]
		wfs2_nx_subap = nx_subap[selector[i,0]]
		
		wfs1_n_subap = n_subap[selector[i,0]]
		wfs2_n_subap = n_subap[selector[i,1]]

		ys = (choice+selector[i,0]) * wfs1_n_subap * 2
		xs = (choice+selector[i,1]) * wfs2_n_subap * 2
		covMat = cov_matrix[int(ys[0]):int(ys[1]),int(xs[0]):int(xs[1])]
		for j in nAxes:
			covMatROI = covMat[j*wfs1_n_subap:(j+1)*wfs1_n_subap, j*wfs2_n_subap:(j+1)*wfs2_n_subap]
			cov_map[i, :, ((wfs1_nx_subap*2)-1)*mapPlace[j]:((wfs2_nx_subap*2)-1)*(mapPlace[j]+1)] = covMap_superFast(cov_mapDim, covMatROI, mm, mmc, md)

	cov_map = cov_map.reshape(combs*cov_mapDim, cov_mapDim*len(nAxes))
	if roi_axis=='x+y':
    		cov_map = (cov_map[:, :cov_mapDim] + cov_map[:, cov_mapDim:])/2.
    		
	return cov_map


if __name__ == '__main__':
	"""Test CANARY"""
	# n_wfs = 2
	# nx_subap = numpy.array([7, 7])
	# n_subap = numpy.array([36, 36])
	# tel_diam = 4.2
	# obs_diam = 1.0
	# roi_axis = 'x+y'
	# pupil_mask = make_pupil_mask('circle', numpy.array([n_subap[0]]), nx_subap[0], obs_diam, tel_diam)
	# mm, mmc, md = get_mappingMatrix(pupil_mask, numpy.ones((n_subap[0], n_subap[0])))

	# cov_matrix = numpy.ones((n_wfs * n_subap[0] * 2, n_wfs * n_subap[0] * 2))
	# cov_map  = covMap_fromMatrix(cov_matrix, n_wfs, nx_subap, n_subap, pupil_mask, roi_axis, mm, mmc, md)


	"""Test AOF"""
	n_wfs = 1
	nx_subap = numpy.array([40, 40])
	n_subap = numpy.array([1240, 1240])
	tel_diam = 8.2
	obs_diam = 1.1
	roi_axis = 'x+y'
	pupil_mask = make_pupil_mask('circle', numpy.array([n_subap[0]]), nx_subap[0], obs_diam, tel_diam)
	mm, mmc, md = get_mappingMatrix(pupil_mask, numpy.ones((n_subap[0], n_subap[0])))

	cov_matrix = numpy.ones((n_wfs * n_subap[0] * 2, n_wfs * n_subap[0] * 2))
	cov_map  = covMap_fromMatrix(cov_matrix, n_wfs, nx_subap, n_subap, pupil_mask, roi_axis, mm, mmc, md)