import numpy
from matplotlib import pyplot; pyplot.ion()
from capt.misc_functions.make_pupil_mask import make_pupil_mask
from capt.roi_functions.roi_referenceArrays import roi_referenceArrays

def lgs_trackMatrix(xy_separations, map_axis):
	combs = xy_separations.shape[0]
	width = xy_separations.shape[1]
	length = xy_separations.shape[2]
	
	if map_axis=='x and y':
		no_axes = 2
	else:
		no_axes = 1

	lgs_trackMatrix = numpy.zeros((combs*width, xy_separations.shape[2]*no_axes))
	tracker = 1
	for i in range(combs):
		for ax in range(no_axes):
			xy_separations[i, :, :, ax][numpy.isnan(xy_separations[i, :, :, ax])==False] = tracker
			lgs_trackMatrix[i*width: (i+1)*width, ax*length: (ax+1)*length] = xy_separations[i, :, :, ax]
			tracker +=1
	
	lgs_trackMatrix[numpy.isnan(lgs_trackMatrix)==True] = 0.
	
	return lgs_trackMatrix


def roi_lgs_track(combs, no_axes, lgs_trackMatrix, track_values):
	tracker = 1
	for i in range(combs):
		for j in range(no_axes):
			
			lgs_trackMatrix[lgs_trackMatrix==tracker] = track_values[i, j]
			tracker += 1

	return lgs_trackMatrix
	


if __name__ == '__main__':



	n_wfs = 3
	gs_pos = numpy.array(([0,-20], [0,20], [20,0]))
	roi_belowGround = 6
	roi_envelope = 6

	n_subap = numpy.array([36]*n_wfs)
	nx_subap = numpy.array([7]*n_wfs)
	obs_diam = 1.
	tel_diam = 4.2 
	pupil_mask = make_pupil_mask('circle', n_subap, nx_subap[0], obs_diam, tel_diam)
	# matrix_region_ones = numpy.ones((n_subap[0], n_subap[0]))
	# mm, mmc, md = get_mappingMatrix(pupil_mask, matrix_region_ones)    
	onesMat, wfsMat_1, wfsMat_2, allMapPos, selector, xy_separations = roi_referenceArrays(pupil_mask, gs_pos, tel_diam, roi_belowGround, roi_envelope)

	m = lgs_trackMatrix(xy_separations, 'x and y')

	# track_values = numpy.array(([1,2], [3,4], [5,6]))*1./7.

	# t = roi_lgs_track(xy_separations.shape[0], xy_separations.shape[3], m, track_values)