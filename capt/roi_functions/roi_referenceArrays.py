import numpy
import itertools
from astropy.io import fits
from scipy.special import comb
from aotools.functions import circle
from matplotlib import pyplot; pyplot.ion()
from capt.misc_functions.cross_cov import cross_cov
from capt.roi_functions.gamma_vector import gamma_vector
from capt.misc_functions.make_pupil_mask import make_pupil_mask
from capt.misc_functions.mapping_matrix import get_mappingMatrix, covMap_superFast, arrayRef
from matplotlib import pyplot; pyplot.ion()


def roi_referenceArrays(pupil_mask, gs_pos, tel_diam, belowGround, envelope):
    """Collection of arrays used to simplify covariance map ROI processing.
    
    Parameters:
        pupil_mask (ndarray): mask of sub-aperture positions within telescope pupil.
        gs_pos (ndarray): GS positions.
        tel_diam (float): telescope diameter.
        belowGround (int): number of sub-aperture separations the ROI expresses 'below-ground'.
        envelope (int): number of sub-aperture separations the ROI expresses either side of stellar separation.

    Returns:
        mm (ndarray): Mapping Matrix.
        sa_mm (ndarray): Mapping Matrix sub-aperture numbering of SHWFS 1.
        sb_mm (ndarray): Mapping Matrix sub-aperture numbering of SHWFS 2.
        allMapPos (ndarray): ROI coordinates in each covariance map combination.
        selector (ndarray): array of all covariance map combinations.
        xy_separations (ndarray): x and y sub-aperture separations corresponding to allMapPos."""

    #Collect mapping matrix (mm), x/y sub-aperture mapping matrix
    nxSubaps = pupil_mask.shape[0]
    subapDiam = tel_diam/nxSubaps
    nSubaps = numpy.int(numpy.sum(pupil_mask))
    covMapDim = (nxSubaps*2)-1
    blankCovMap = numpy.zeros((covMapDim, covMapDim))
    posMatrix = numpy.ones((nSubaps, nSubaps))
    
    #mm - mapping matrix, mmc - mapping matrix coordinates, md - map denominator
    mm, mmc, md = get_mappingMatrix(pupil_mask, posMatrix)
    mm = mm.astype(int)
    onesMap = covMap_superFast(covMapDim, posMatrix, mm, mmc, md)
    nanMap = onesMap.copy()
    nanMap[onesMap==0] = numpy.nan

    #sa_mm - mm sub-aperture numbering of shwfs1, sb_mm - mm sub-aperture numbering of shwfs2
    sa_mm, mmc, md = get_mappingMatrix(pupil_mask, posMatrix*numpy.arange(nSubaps)[::-1])
    sb_mm, mmc, md = get_mappingMatrix(pupil_mask, numpy.rot90(posMatrix*numpy.arange(nSubaps)[::-1],3))
    sa_mm = sa_mm.astype(int)
    sb_mm = sb_mm.astype(int)

    #Determine no. of GS combinations
    combs = int(comb(gs_pos.shape[0], 2, exact=True))
    selector = numpy.array((range(gs_pos.shape[0])))
    selector = numpy.array((list(itertools.combinations(selector, 2))))
    fakePosMap, b, t = gamma_vector(blankCovMap, 1, 'False', belowGround, envelope, False)
    # print('No. of GS combinations:', combs, '\n')

    #Sub-aperture postions for generateSliceCovariance and map vectors for mapSliceFromSlopes
    sliceWidth = 1+(2*envelope)
    sliceLength = pupil_mask.shape[0] + belowGround
    subapNum = numpy.ones((2, combs, sliceWidth, sliceLength))
    subapLocations = numpy.zeros((2, combs, sliceWidth, sliceLength, 2))
    subapLocations1 = numpy.zeros((combs, sliceWidth, sliceLength, 2))
    subapLocations2 = numpy.zeros((combs, sliceWidth, sliceLength, 2))

    allMapPos = numpy.zeros((combs, sliceWidth, sliceLength, 2)).astype(int)
    vectorMap = numpy.zeros((combs*blankCovMap.shape[0], blankCovMap.shape[1]))
    refArray = arrayRef(pupil_mask)

    #x and y separations in covariance map space - used to get xy_separations
    yMapSep = nanMap.copy() * numpy.arange(-(pupil_mask.shape[0]-1), (pupil_mask.shape[0])) * tel_diam/pupil_mask.shape[0] 
    xMapSep = yMapSep.copy().T
    xy_separations = numpy.zeros((combs, sliceWidth, sliceLength, 2))

    for k in range(combs):
        posMap = 0.
        posMap, vector, t = gamma_vector(blankCovMap, 'False', gs_pos[selector[k]], belowGround, envelope, False)    #gets covariance map vector

        xy_separations[k] = numpy.stack((xMapSep[posMap.T[0], posMap.T[1]], yMapSep[posMap.T[0], posMap.T[1]])).T

        #Make all vector coordinates off-map 2*covMapDiam
        for i in range(posMap.shape[0]):
            for j in range(posMap.shape[1]):
                if onesMap[posMap[i,j,0], posMap[i,j,1]] == 0:
                    posMap[i,j] = 2*covMapDim, 2*covMapDim


        #Sub-aperture positions for specific GS combination. All non-positions are nan
        subapPos1 = numpy.ones(posMap.shape)*numpy.nan
        subapPos2 = numpy.ones(posMap.shape)*numpy.nan

        for i in range(posMap.shape[0]):
            subapPos = []
            for j in range(posMap.shape[1]):
                if posMap[i,j,0] != 2*covMapDim:
                    pupil_mask1 = numpy.zeros(pupil_mask.shape)
                    pupil_mask2 = numpy.zeros(pupil_mask.shape)

                    #Sub-aperture numbers that correspond to the required covariance measurements. Uses mapping matrix as look-up table.
                    subapNum1 =  sa_mm[:, posMap[i,j,0]*covMapDim + (posMap[i,j,1])][numpy.where(mm[:, posMap[i,j,0]*covMapDim + (posMap[i,j,1])]==1)[0]][0]
                    subapNum2 =  sb_mm[:, posMap[i,j,0]*covMapDim + (posMap[i,j,1])][numpy.where(mm[:, posMap[i,j,0]*covMapDim + (posMap[i,j,1])]==1)[0]][0]
                    # print subapNum1, '\n'
                    subapNum[0, k, i, j] = subapNum1
                    subapNum[1, k, i, j] = subapNum2

                    #Converts first set of sub-aperture to their displacement from the centre of the telescope's pupil.
                    pupil_mask1[numpy.where(refArray==subapNum1)] = 1
                    subapPos1[i,j] = numpy.asarray(numpy.where(pupil_mask1 == 1)).T * subapDiam
                    # subapPos1[i,j] -= tel_diam/2.
                    # subapPos1[i,j] += subapDiam/2.

                    #Converts second set of sub-aperture to their displacement from the centre of the telescope's pupil.
                    pupil_mask2[numpy.where(refArray==subapNum2)] = 1
                    subapPos2[i,j] = numpy.asarray(numpy.where(pupil_mask2 == 1)).T * subapDiam
                    # subapPos2[i,j] -= tel_diam/2.
                    # subapPos2[i,j] += subapDiam/2.

        subapLocations1[k] = subapPos1
        subapLocations2[k] = subapPos2
        allMapPos[k] = posMap
        vectorMap[k*blankCovMap.shape[0]: (k+1)*blankCovMap.shape[0]] = vector

    #subapPositons is a list of shape (2, combs, 1+2*envelope, vectorLength (i.e. 7/8 for CANARY), 2) 
    subapLocations[0] = subapLocations1
    subapLocations[1] = subapLocations2
    # subapLocations.append(subapLocations1)
    # subapLocations.append(subapLocations2)

    subapPos.append(subapPos1[0])
    subapPos.append(subapPos2[0])

    # print('\n'+'################ REFERENCE ARRAYS OBTAINED ###############', '\n')
    return mm, sa_mm, sb_mm, allMapPos, selector, xy_separations


if __name__=='__main__':
    mask = 'circle'
    n_subap = 36
    nx_subap = 7
    obs_diam = 1.2
    tel_diam = 4.2
    pupil_mask = make_pupil_mask(mask, n_subap, nx_subap, obs_diam, tel_diam)
    gs_pos = numpy.array([(0, -20), (0, 20)])
    belowGround = 0
    envelope = 0

    mm, sa_mm, sb_mm, allMapPos, selector, xy_separations = roi_referenceArrays(pupil_mask, gs_pos, tel_diam, belowGround, envelope)