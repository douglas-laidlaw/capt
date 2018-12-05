import time
import numpy
import itertools
from scipy.misc import comb
from astropy.io import fits
from aotools.functions import circle
from matplotlib import pyplot; pyplot.ion()
from capt.misc_functions.cross_cov import cross_cov
from capt.roi_functions.gamma_vector import gamma_vector
from capt.misc_functions.make_pupil_mask import make_pupil_mask
from capt.roi_functions.roi_referenceArrays import roi_referenceArrays
from capt.misc_functions.mapping_matrix import get_mappingMatrix, covMap_superFast, arrayRef




def calculate_roi_covariance(shwfs_centroids, allMapPos, covMapDim, n_subap, mm, sa_mm, sb_mm, selector, roi_axis, mapping_type):
    """Takes SHWFS centroids and directly calculates the covariance map ROI (does not require going via covariance matrix).
    
    Parameters:
        shwfs_centroids (ndarray): SHWFS centroid measurements.
        allMapPos (ndarray): covariance map ROI coordinates within covariance map (for each GS combination).
        covMapDim (int): covariance map length in x or y (should be equal).
        n_subap (ndarray): number of sub-apertures within each SHWFS.
        mm (ndarray): Mapping Matrix.
        sa_mm (ndarray): Mapping Matrix sub-aperture numbering of SHWFS 1.
        sb_mm (ndarray): Mapping Matrix sub-aperture numbering of SHWFS 2.
        selector (ndarray): array of all covariance map combinations.
        roi_axis (str): in which axis to express ROI ('x', 'y', 'x+y' or 'x and y')
        mapping_type (str): how to calculate overall sub-aperture separation covariance ('mean' or 'median')

    Returns:
        roi_covariance (ndarray): covariance map ROI.
        time_taken (float): time taken to complete calculation."""

    timeStart = time.time()

    #subtracts mean at each sub-aperture axis (first step in calculating cross-covariance).
    shwfs_centroids = (shwfs_centroids - shwfs_centroids.mean(0)).T

    if roi_axis=='x' or roi_axis=='y' or roi_axis=='x+y':
        roi_covariance = numpy.zeros((allMapPos.shape[0]*allMapPos.shape[1], allMapPos.shape[2]))
    if roi_axis=='x and y':
        roi_covariance = numpy.zeros((allMapPos.shape[0]*allMapPos.shape[1], allMapPos.shape[2]*2))

    allMapPos[allMapPos==2*covMapDim] = 0
    allMapPos = allMapPos[:, :, :, 1] + allMapPos[:, :, :, 0] * covMapDim

    wfs1_n_subap = n_subap[0]
    wfs2_n_subap = n_subap[0]

    av_mm = numpy.zeros(mm.shape)
    cov_mm = numpy.zeros(mm.shape)


    for i in range(allMapPos.shape[0]):
        
        loc = (mm[:, allMapPos[i].flatten()]==1)
        where_loc = numpy.where(loc==1)
        subapsWfs1 = sa_mm[:, allMapPos[i].flatten()][loc]
        subapsWfs2 = sb_mm[:, allMapPos[i].flatten()][loc]

        #create roi to perform averaging
        av_mm[loc] = 1
        av = numpy.sum(av_mm, 0).reshape(allMapPos.shape[1], allMapPos.shape[2])
        av[av==0] = 1.

        #calculated xx covariance
        if roi_axis!='y':
            #Convert from sub-aperture no. to x-slope no. within the respective WFSs
            subapsXX1 = subapsWfs1 + (selector[i][0]*2*wfs1_n_subap)
            subapsXX2 = subapsWfs2 + (selector[i][1]*2*wfs2_n_subap)

            st = time.time()
            stepSize = 5000
            for j in range(loc.sum()):
                cov_mm[where_loc[0][j*stepSize: (j+1)*stepSize], where_loc[1][j*stepSize: (j+1)*stepSize]] = ((shwfs_centroids[subapsXX1[j*stepSize: (j+1)*stepSize]] * shwfs_centroids[subapsXX2[j*stepSize: (j+1)*stepSize]]).sum(1)/(shwfs_centroids.shape[1]-1))
                print(j*stepSize)
            print('for time: {}'.format(time.time() - st))
            stop
            #covariance calculation and fill mapping matrix roi with covariance values before reshaping and then averaging 
            cov_mm[loc] = ((shwfs_centroids[subapsXX1] * shwfs_centroids[subapsXX2]).sum(1)/(shwfs_centroids.shape[1]-1))
            if mapping_type!='mean':
                raise Exception('Mapping Type not known.')
            else:
                roi_cov_xx = numpy.sum(cov_mm, 0).reshape(allMapPos.shape[1], allMapPos.shape[2])/av
                cov_mm *= 0
                stop

        #calculated yy covariance
        if roi_axis!='x':
            #Convert from sub-aperture no. to y-slope no. within the respective WFSs
            subapsYY1 = subapsWfs1 + (selector[i][0]*2*wfs1_n_subap) + wfs1_n_subap
            subapsYY2 = subapsWfs2 + (selector[i][1]*2*wfs2_n_subap) + wfs2_n_subap

            cov_mm[loc] = ((shwfs_centroids[subapsYY1] * shwfs_centroids[subapsYY2]).sum(1)/(shwfs_centroids.shape[1]-1))

            #covariance calculation and fill mapping matrix roi with covariance values before reshaping and then averaging 
            if mapping_type!='mean':
                raise Exception('Mapping Type not known.')
            else:
                roi_cov_yy = numpy.sum(cov_mm, 0).reshape(allMapPos.shape[1], allMapPos.shape[2])/av
                cov_mm *= 0

        av_mm *= 0

        if roi_axis=='x':
            roi_covariance[i*allMapPos.shape[1]:(i+1)*allMapPos.shape[1]] = roi_cov_xx
        if roi_axis=='y':
            roi_covariance[i*allMapPos.shape[1]:(i+1)*allMapPos.shape[1]] = roi_cov_yy
        if roi_axis=='x+y':
            roi_covariance[i*allMapPos.shape[1]:(i+1)*allMapPos.shape[1]] = (roi_cov_xx+roi_cov_yy)/2.
        if roi_axis=='x and y':
            roi_covariance[i*allMapPos.shape[1]:(i+1)*allMapPos.shape[1]] = numpy.hstack((roi_cov_xx, roi_cov_yy))

        stop


    timeStop = time.time()
    time_taken = timeStop - timeStart

    return roi_covariance, time_taken






if __name__=='__main__':
    n_wfs = 3
    gs_pos = numpy.array(([0,-40], [0, 0], [30,0]))
    tel_diam = 4.2
    roi_belowGround = 6
    roi_envelope = 6
    nx_subap = numpy.array([7]*n_wfs)
    n_subap = numpy.array([36]*n_wfs)

    pupil_mask = make_pupil_mask('circle', n_subap, nx_subap[0], 
            1., tel_diam)

    onesMat, wfsMat_1, wfsMat_2, allMapPos, selector, xy_separations = roi_referenceArrays(
                pupil_mask, gs_pos, tel_diam, roi_belowGround, roi_envelope)
    
    # shwfs_centroids = fits.getdata('../../../../windProfiling/wind_paper/canary/data/test_fits/canary_noNoise_it10k_nl3_h0a10a20km_r00p1_L025_ws10a15a20_wd260a80a350_infScrn_wss448_gsPos0cn40a0c0a30c0.fits')
    shwfs_centroids = numpy.ones((10000, 36*2*3))
    covMapDim = 13
    roi_axis = 'x and y'
    mapping_type = 'mean'

    nr, nt = calculate_roi_covariance(shwfs_centroids, allMapPos, covMapDim, n_subap, onesMat, wfsMat_1, wfsMat_2, selector, roi_axis, mapping_type)