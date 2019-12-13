import numpy
from matplotlib import pyplot; pyplot.ion()
from capt.misc_functions.make_pupil_mask import make_pupil_mask



def matrixSubapSep(pupil_mask):
    """Used to visualise sub-aperture separation within covariance matrix ROI.

    Parameters:
        pupil_mask (ndarray): mask of SHWFS sub-apertures within the telescope's pupil.

    Returns:
        ndarray: sub-aperture separation in x within covariance matrix ROI.
        ndarray: sub-aperture separation in y within covariance matrix ROI.
        ndarray: SHWFS sub-aperture locations. To be used for sub-aperture seperation calculations."""

    ref = arrayRef(pupil_mask)
    nSubaps = numpy.int(pupil_mask.sum())
    subapSepMatrixX = numpy.zeros((nSubaps, nSubaps))
    subapSepMatrixY = numpy.zeros((nSubaps, nSubaps))
    for i in range(nSubaps):								#cycle through covariance matrix
        for j in range(nSubaps):
            cent1 = numpy.where(j==ref)										#take [i,j] coordinates from
            cent2 = numpy.where(i==ref)										#arrayRef(pupil_mask)
            xsep = (cent2[1][0]-cent1[1][0])
            ysep = (cent1[0][0]-cent2[0][0])					#x and y seperation

            subapSepMatrixX[j, i] = xsep
            subapSepMatrixY[j, i] = -ysep

    return subapSepMatrixX, subapSepMatrixY, ref




def arrayRef(pupil_mask):
    """Creates array of SHWFS sub-aperture locations. To be used for sub-aperture seperation calculations.

    Parameters:
        pupil_mask (ndarray): mask of SHWFS sub-apertures within the telescope's pupil.
    
    Returns:
        ref (ndarray): telescope pupil SHWFS sub-aperture locations."""

    ref = numpy.zeros((pupil_mask.shape[0],pupil_mask.shape[1]),numpy.float)
    count=0
    for i in range(pupil_mask.shape[0]):
        for j in range(pupil_mask.shape[1]):												#systematically cycle through pupil_mask
            if pupil_mask[i,j]==0:														  #if pupil_mask[i,j]=0 change value to -1
                ref[i,j] = -1
            if pupil_mask[i,j]==1:														  #if pupil_mask[i,j]=1 insert number corresponding
                if count<numpy.sum(pupil_mask):											 #the number of times a "1" has previously been
                    ref[i,j] = range(numpy.sum(pupil_mask, dtype=numpy.int64))[count]	   #found.
                if count>=numpy.sum(pupil_mask):
                    ref[i,j] = range(numpy.sum(pupil_mask, dtype=numpy.int64))[count]	   #once last "1" reached, end.
                count+=1
    return ref





def squarePupilMatrix(matrixGrid, pupil_mask):
    """Expands covariance matrix ROI (e.g. x1x2) to the dimensions it would have if all 
    SHWFS sub-apertures were in use i.e. if pupil_mask was a square array of 1s. 
    
    Parameters:
        matrixGrid (ndarray): array that matches covariance matrix ROI shape (e.g. x1x2).
        pupil_mask (ndarray): mask of SHWFS sub-apertures within the telescope's pupil.

    Returns:
        ndarray: matrixGrid expanded to include null SHWFS sub-apertures."""

    pupilFlat = pupil_mask.flatten()
    mapMatrix1 = numpy.zeros((int(pupil_mask.sum()), pupil_mask.shape[0]**2))
    mapMatrix = numpy.zeros((pupil_mask.shape[0]**2, pupil_mask.shape[0]**2))
    for i in range(matrixGrid.shape[0]):
        mapMatrix1[i,numpy.where(pupilFlat==1)] = matrixGrid[i]
    for i in range(mapMatrix1.shape[1]):
        mapMatrix[numpy.where(pupilFlat==1),i] = mapMatrix1[:,i]
    return mapMatrix





def zeroDisY(ySquareGrid, nx_subap):
    """Shifts squarePupilMatrix to align coordinates with equal values of displacememnt in y.
    
    Parameters:
        ySquareGrid (ndarray): mapMatrix from squarePupilMatrix.
        nx_subap (int): number of sub-apertures across telescope pupil diameter.

    Returns:
        ndarray: ySquareGrid with y displacements aligned."""

    yDisZero = numpy.zeros((ySquareGrid.shape[0], (ySquareGrid.shape[0]*2) - nx_subap))
    yDisZero[:, ySquareGrid.shape[0] - nx_subap:] = ySquareGrid
    for i in range(nx_subap):
        yDisZero[i*nx_subap:(i+1)*nx_subap] = numpy.roll(yDisZero[i*nx_subap:(i+1)*nx_subap], -i*nx_subap)
    return yDisZero






def zeroDisXY(yDisZero, nx_subap):
    """Shifts zeroDisY to align coordinates with equal values of displacememnt in x, at every y displacement.
    
    Parameters:
        yDisZero (ndarray): yDisZero from zeroDisY.
        nx_subap (int): number of sub-apertures across telescope pupil diameter.

    Returns:
        xyDisZero (ndarray): yDisZero with x displacements aligned."""

    covMapDim = (nx_subap*2-1)
    diff = numpy.int((covMapDim-nx_subap)/2.)
    xDisZero = numpy.zeros((yDisZero.shape[0], (nx_subap*2-1)**2))
    for i in range(nx_subap*2-1):
        xDisZero[:,(i*nx_subap) + (i+1)*nx_subap-1-i: i*nx_subap + (i+2)*nx_subap-1-i] = yDisZero[:,i*nx_subap:(i+1)*nx_subap]
    xyDisZero = numpy.zeros(xDisZero.shape)

    for i in range(nx_subap):
        for j in range(nx_subap):
            xyDisZero[i*nx_subap + j] = numpy.roll(xDisZero[i*nx_subap+j], -j)

    return xyDisZero




def get_mappingMatrix(pupil_mask, grid):
    """Retrieve Mapping Matrix.
    
    Parameters:
        pupil_mask (ndarray): mask of SHWFS sub-apertures within the telescope's pupil.

    Returns:
        mappingMatrix (ndarray): covariance matrix ROI (e.g. x1x2) with x and y sub-aperture separations aligned.
        mappingMatrixCoords (ndarray): coordinates of mappingMatrix sub-aperture separation locations.
        covMapDenom (ndarray): covariance map sub-aperture separation density."""

    nx_subap = pupil_mask.shape[0]
    nSubaps = numpy.int(numpy.sum(pupil_mask))
    squarePupil = squarePupilMatrix(grid, pupil_mask)
    zeroY = zeroDisY(squarePupil, nx_subap)
    mappingMatrix = (zeroDisXY(zeroY, nx_subap))

    onesMatrix = numpy.ones((numpy.int(numpy.sum(pupil_mask)), numpy.int(numpy.sum(pupil_mask))))
    mappingOnes = (zeroDisXY(zeroDisY(squarePupilMatrix(onesMatrix, pupil_mask), nx_subap), nx_subap))
    mappingMatrixCoords = numpy.where(mappingOnes==1)

    covMapDenom = covMap_superFast((nx_subap*2)-1, onesMatrix, mappingOnes, mappingMatrixCoords, 1.)
    covMapDenom[numpy.where(covMapDenom==0)] = 1

    return mappingMatrix.astype('float64'), mappingMatrixCoords, covMapDenom.astype('float64')




def covMap_superFast(covMapDim, covMatROI, mappingMatrix, mappingMatrixCoords, covMapDenom):
    """Calculates the covariance map from a covariance matrix ROI (e.g. x1x2).
    
    Parameters:
        covMapDim (int): dimension of covariance map, (2*nx_subap)-1
        covMatROI (ndarray): covariance matrix ROI (e.g. x1x2).
        mappingMatrix (ndarray): covariance matrix ROI (e.g. x1x2) with x and y sub-aperture separations aligned.
        mappingMatrixCoords (int): coordinates of mappingMatrix sub-aperture separation locations.
        covMapDenom (ndarray): covariance map sub-aperture separation density.

    Returns:
        ndarray: covariance map"""

    mm = mappingMatrix.copy()
    mm[mappingMatrixCoords[0], mappingMatrixCoords[1]] = covMatROI.flatten()

    return ((numpy.sum(mm, 0).reshape(covMapDim, covMapDim))/covMapDenom)


if __name__ == '__main__':
    # pupil_mask = make_pupil_mask('circle', numpy.array([36]), 7, 1., 4.2)
    # grid = numpy.ones((36, 36))
    # mm_canary, mmc_canary, md_canary = get_mappingMatrix(pupil_mask, grid)
    # grid = numpy.ones((30, 30))
    # pupil_mask[:,3]=0
    # mm_canary, mmc_canary, md_canary = get_mappingMatrix(pupil_mask, grid)
    # c_map = covMap_superFast(13, grid, mm_canary, mmc_canary, md_canary)
    # numpy.save('mm_canary.npy', mm_canary)
    # numpy.save('mmc_canary.npy', mmc_canary)
    # numpy.save('md_canary.npy', md_canary)

    pupil_mask = make_pupil_mask('circle', numpy.array([1240]), 40, 1.1, 8.20)
    grid = numpy.ones((1240, 1240))
    mm_aof, mmc_aof, md_aof = get_mappingMatrix(pupil_mask, grid)
    c_map = covMap_superFast(79, grid, mm_aof, mmc_aof, md_aof)
    numpy.save('mm_aof.npy', mm_aof)
    numpy.save('mmc_aof.npy', mmc_aof)
    numpy.save('md_aof.npy', md_aof)

    # pupil_mask = make_pupil_mask('circle', numpy.array([4260]), 74, 4., 39.)
    # grid = numpy.ones((4260, 4260))
    # mm_har, mmc_har, md_har = get_mappingMatrix(pupil_mask, grid)
    # numpy.save('mm_harmoni.npy', mm_har)
    # numpy.save('mmc_harmoni.npy', mmc_har)
    # numpy.save('md_harmoni.npy', md_har)