import numpy
import time
import myFont
import itertools
import eltConfig
import aofConfig
import canaryConfig
from scipy.misc import comb
from crossCov import crossCov
from astropy.io import fits
from aotools.functions import circle
from matplotlib import pyplot; pyplot.ion()
from superFastCovMap import  arrayRef, superFastCovMap, getMappingMatrix
from covMapVector import covMapVector
from covMapsFromMatrix import covMapsFromMatrix



def referenceArrays(pupil, gsPos, telDiam, belowGround, envelope):
    #Collect mapping matrix (onesMat), x/y sub-aperture mapping matrix
    nxSubaps = pupil.shape[0]
    subapDiam = telDiam/nxSubaps
    nSubaps = numpy.int(numpy.sum(pupil))
    covMapDim = (nxSubaps*2)-1
    blankCovMap = numpy.zeros((covMapDim, covMapDim))
    posMatrix = numpy.ones((nSubaps, nSubaps))
    onesMat, coords, denom = getMappingMatrix(pupil, posMatrix)
    onesMap = superFastCovMap(covMapDim, posMatrix, onesMat, coords, denom)
    nanMap = onesMap.copy()
    nanMap[onesMap==0] = numpy.nan

    wfsMat_1, coords, denom = getMappingMatrix(pupil, posMatrix*numpy.arange(nSubaps)[::-1])
    wfsMat_2, coords, denom = getMappingMatrix(pupil, numpy.rot90(posMatrix*numpy.arange(nSubaps)[::-1],3))
    wfsMat_1 = wfsMat_1.astype(int)
    wfsMat_2 = wfsMat_2.astype(int)

    #Determine no. of GS combinations
    combs = int(comb(gsPos.shape[0], 2, exact=True))
    selector = numpy.array((range(gsPos.shape[0])))
    selector = numpy.array((list(itertools.combinations(selector, 2))))
    fakePosMap, b, t = covMapVector(blankCovMap, 1, 'False', belowGround, envelope, False)
    print('No. of GS combinations:', combs, '\n')

    #Sub-aperture postions for generateSliceCovariance and map vectors for mapSliceFromSlopes
    sliceWidth = 1+(2*envelope)
    sliceLength = pupil.shape[0] + belowGround
    subapNum = numpy.ones((2, combs, sliceWidth, sliceLength))
    subapLocations = numpy.zeros((2, combs, sliceWidth, sliceLength, 2))
    subapLocations1 = numpy.zeros((combs, sliceWidth, sliceLength, 2))
    subapLocations2 = numpy.zeros((combs, sliceWidth, sliceLength, 2))

    allMapPos = numpy.zeros((combs, sliceWidth, sliceLength, 2)).astype(int)
    vectorMap = numpy.zeros((combs*blankCovMap.shape[0], blankCovMap.shape[1]))
    refArray = arrayRef(pupil)

    #x and y separations in covariance map space - used to get xy_separations
    yMapSep = nanMap.copy() * numpy.arange(-(pupil.shape[0]-1), (pupil.shape[0])) * telDiam/pupil.shape[0] 
    xMapSep = yMapSep.copy().T
    xy_separations = numpy.zeros((combs, sliceWidth, sliceLength, 2))

    for k in range(combs):
        posMap = 0.
        posMap, vector, t = covMapVector(blankCovMap, 'False', gsPos[selector[k]], belowGround, envelope, False)    #gets covariance map vector

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
                    pupilMask1 = numpy.zeros(pupil.shape)
                    pupilMask2 = numpy.zeros(pupil.shape)

                    #Sub-aperture numbers that correspond to the required covariance measurements. Uses mapping matrix as look-up table.
                    subapNum1 =  wfsMat_1[:, posMap[i,j,0]*covMapDim + (posMap[i,j,1])][numpy.where(onesMat[:, posMap[i,j,0]*covMapDim + (posMap[i,j,1])]==1)[0]][0]
                    subapNum2 =  wfsMat_2[:, posMap[i,j,0]*covMapDim + (posMap[i,j,1])][numpy.where(onesMat[:, posMap[i,j,0]*covMapDim + (posMap[i,j,1])]==1)[0]][0]
                    # print subapNum1, '\n'
                    subapNum[0, k, i, j] = subapNum1
                    subapNum[1, k, i, j] = subapNum2

                    #Converts first set of sub-aperture to their displacement from the centre of the telescope's pupil.
                    pupilMask1[numpy.where(refArray==subapNum1)] = 1
                    subapPos1[i,j] = numpy.asarray(numpy.where(pupilMask1 == 1)).T * subapDiam
                    # subapPos1[i,j] -= telDiam/2.
                    # subapPos1[i,j] += subapDiam/2.

                    #Converts second set of sub-aperture to their displacement from the centre of the telescope's pupil.
                    pupilMask2[numpy.where(refArray==subapNum2)] = 1
                    subapPos2[i,j] = numpy.asarray(numpy.where(pupilMask2 == 1)).T * subapDiam
                    # subapPos2[i,j] -= telDiam/2.
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

    print('\n'+'################ REFERENCE ARRAYS OBTAINED ###############', '\n')
    return onesMat, wfsMat_1, wfsMat_2, allMapPos, selector, xy_separations
    







if __name__=='__main__':
    
    """ DEMONSTRATION """
    config = canaryConfig.telescopeParams
    slopes = fits.getdata('../fitsFiles/canary_noNoise_wssInf_it10k_h0kma9p3km_r00p1_L025_wdAll45_gsPos0cn20a0c20_fftOver1_ct0p1.fits') * (1./3600)*(numpy.pi/180)*(4.5/16.)

    telDiam = config["Telescope"]["diameter"]
    pupil = config["WFS"]["pupilMask"]
    nxSubaps = config["WFS"]["nxSubaps"]
    nSubaps = config["WFS"]["nSubaps"]
    covMapDim = (pupil.shape[0]*2)-1
    ones = numpy.ones((int(pupil.sum()), int(pupil.sum())))
    mm, mmc, n = getMappingMatrix(pupil, ones)
    output = 'x and y'
    mappingType = 'mean'


    cts = time.time()
    covMat = crossCov(slopes)
    covMaps = covMapsFromMatrix(covMat, gsPos, nxSubaps, nSubaps, pupil, output, mm, mmc, n)
    ctf = time.time()

    # pyplot.figure()
    # pyplot.imshow(covMaps)



    envelope = 1
    belowGround = 2
    gsPos = numpy.array([[0,-20],[0,20], [10,0]])
    onesMat, wfsMat_1, wfsMat_2, subapLoc, allMapPos, selector, xy_separations = referenceArrays(pupil, gsPos, telDiam, belowGround, envelope)
    # sts = time.time()
    # mapSlices = mapSliceFromSlopes(slopes, allMapPos, covMapDim, nSubaps, onesMat, wfsMat_1, wfsMat_2, selector, output, mappingType)
    # stf = time.time()
    # print('\n', 'Covariance Slice Speed Factor:', str((ctf-cts)/(stf-sts))+'x', '\n')

    # pyplot.figure()
    # pyplot.imshow(vectorMap)
    # pyplot.figure()
    # pyplot.imshow(mapSlices)