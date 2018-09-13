import math
import numpy
import itertools
from aotools.functions import circle
from capt.roi_functions.gamma_vector import gamma_vector 

def roi_from_map(covMaps, gs_pos, pupil_mask, selector, belowGround, envelope):
	"""Cuts Region of Interest (ROI) from covariance map. ROI is defined by the
	GS stellar seperation. Width and length of ROI are increased by envelope and 
	belowGround, respectively.

	Parameters:
		covMaps (ndarray): covariance map(s).
		gs_pos (ndarray): GS asterism in telescope FoV.
		pupil_mask (ndarray):  mask of sub-aperture positions within telescope pupil.
		selector (ndarray): array of all covariance map combinations
		belowGround (int): number of sub-aperture separations the ROI expresses 'below-ground'.
		envelope (int): number of sub-aperture separations the ROI expresses either side of stellar separation.

	Returns:
		ndarray: covariance map ROI."""

	nSubaps = int(pupil_mask.sum())
	nxSubaps = int(pupil_mask.shape[0])
	covMapDim = (2*nxSubaps)-1
	blankCovMap = numpy.zeros((covMapDim, covMapDim))
	nAxes = int(covMaps.shape[1]/covMapDim)
	map_roi = numpy.zeros(( (1+(2*envelope))*selector.shape[0], (nxSubaps+belowGround)*nAxes ))

	for j in range(nAxes):
		for i in range(selector.shape[0]):
			spec_GsPos = numpy.array((gs_pos[selector[i,0]], gs_pos[selector[i,1]]))
			vector, vectorMap, theta = gamma_vector(blankCovMap, 'False', spec_GsPos, belowGround, envelope, False)
			for k in range(vector.shape[0]):
				for l in range(vector.shape[1]):
					map_roi[ i*(1+(2*envelope)) + k, j*(nxSubaps+belowGround) + l ] = covMaps[ vector[k,l,0] + (i*covMapDim), vector[k,l,1] + (j*covMapDim) ]

	return map_roi


if __name__ == '__main__':
	maps = numpy.ones((39,26))
	gs_pos = numpy.array(([0,0],[0,-1],[1,0]))
	pupil_mask = circle(7./2, 7)
	pupil_mask[3,3]=0
	envelope = 0
	selector = numpy.array((range(gs_pos.shape[0])))
	selector = numpy.array((list(itertools.combinations(selector, 2))))

	m = roi_from_map(maps, gs_pos, pupil_mask, selector, 0, 0)