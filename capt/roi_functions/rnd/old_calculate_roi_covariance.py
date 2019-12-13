import time
import numpy
import itertools
from memprof import *
from scipy.misc import comb
from astropy.io import fits
from aotools.functions import circle
from matplotlib import pyplot; pyplot.ion()
from capt.misc_functions.cross_cov import cross_cov
from capt.roi_functions.gamma_vector import gamma_vector
from capt.misc_functions.make_pupil_mask import make_pupil_mask
from capt.roi_functions.roi_referenceArrays import roi_referenceArrays
from capt.misc_functions.mapping_matrix import get_mappingMatrix, covMap_superFast, arrayRef



# @memprof()
def calculate_roi_covariance(shwfs_centroids, allMapPos, covMapDim, n_subap, mm, sa, sb, selector, roi_axis, mapping_type):
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


    for i in range(allMapPos.shape[0]):

        loc = mm[:, allMapPos[i].flatten()]
        sa_mm = sa[:, allMapPos[i].flatten()]
        sb_mm = sb[:, allMapPos[i].flatten()]

        #create roi to perform averaging
        av = numpy.sum(loc, 0).reshape(allMapPos.shape[1], allMapPos.shape[2])
        av[av==0] = 1.

        x_s_covs = numpy.zeros((loc.shape[0], loc.shape[1]))
        y_s_covs = numpy.zeros((loc.shape[0], loc.shape[1]))
        sa_mm += loc
        sb_mm += loc


        for j in range(1,wfs1_n_subap+1):
            
            #integer shift for each GS combination 
            subap1_xComb_shift = selector[i][0]*2*wfs1_n_subap
            subap2_xComb_shift = selector[i][1]*2*wfs1_n_subap
            
            #sub-aperture number in x
            x_subap = (j-1) + subap1_xComb_shift
            #locations of sub-aperture number in x within sa_mm
            sa_loc_xSubap = numpy.where(sa_mm==j)

            if roi_axis!='y':
            
                #shwfs centroids for sub-aperture number in x
                # subap1 = shwfs_centroids[x_subap]
                #sub-aperture numbers in sb_mm to be paired with x_subap
                # subaps2 = sb_mm[sa_loc_xSubap] + subap2_xComb_shift - 1
                #shwfs centroids for sub-apertures in x to be paired with subap1
                # subaps2 = shwfs_centroids[subaps2]

                # cova = (subap1 * subaps2).sum(1)/(shwfs_centroids.shape[1]-1)

                #calculate cross-covariance
                cova = (shwfs_centroids[x_subap] * (shwfs_centroids[sb_mm[sa_loc_xSubap] + subap2_xComb_shift - 1])).sum(1)/(shwfs_centroids.shape[1]-1)
                x_s_covs[sa_loc_xSubap] = cova

                # shutit


                if j==wfs1_n_subap:
                    roi_cov_xx = (numpy.sum(x_s_covs, 0).reshape(allMapPos.shape[1], allMapPos.shape[2]))/av
                    # del subap1
                    # del subaps2
                    # del cova
                    # del x_s_covs
                    # del loc


            if roi_axis!='x':

                subap1_yComb_shift = subap1_xComb_shift + wfs1_n_subap
                subap2_yComb_shift = subap2_xComb_shift + wfs2_n_subap
                y_subap = x_subap + wfs1_n_subap

                # #shwfs centroids for sub-aperture number in y
                # subap1 = shwfs_centroids[y_subap]
                # #sub-aperture numbers in sb_mm to be paired with y_subap
                # subaps2 = sb_mm[sa_loc_xSubap] + subap2_yComb_shift - 1
                # #shwfs centroids for sub-apertures in y to be paired with subap1
                # subaps2 = shwfs_centroids[subaps2]

                # #calculate cross-covariance
                # cova = (subap1 * subaps2).sum(1)/(shwfs_centroids.shape[1]-1)
                # y_s_covs[sa_loc_xSubap] = cova

                cova = (shwfs_centroids[y_subap] * (shwfs_centroids[sb_mm[sa_loc_xSubap] + subap2_yComb_shift - 1])).sum(1)/(shwfs_centroids.shape[1]-1)
                y_s_covs[sa_loc_xSubap] = cova

                if j==wfs1_n_subap:
                    roi_cov_yy = (numpy.sum(y_s_covs, 0).reshape(allMapPos.shape[1], allMapPos.shape[2]))/av

                    # del subap1
                    # del subaps2
                    # del cova
                    # del y_s_covs
                    # del loc


        if roi_axis=='x':
            roi_covariance[i*allMapPos.shape[1]:(i+1)*allMapPos.shape[1]] = roi_cov_xx
        if roi_axis=='y':
            roi_covariance[i*allMapPos.shape[1]:(i+1)*allMapPos.shape[1]] = roi_cov_yy
        if roi_axis=='x+y':
            roi_covariance[i*allMapPos.shape[1]:(i+1)*allMapPos.shape[1]] = (roi_cov_xx+roi_cov_yy)/2.
        if roi_axis=='x and y':
            roi_covariance[i*allMapPos.shape[1]:(i+1)*allMapPos.shape[1]] = numpy.hstack((roi_cov_xx, roi_cov_yy))

    timeStop = time.time()
    time_taken = timeStop - timeStart

    return roi_covariance, time_taken






if __name__=='__main__':
    n_wfs = 3
    gs_pos = numpy.array(([0,-40], [0, 0], [30,0]))
    # gs_pos = numpy.array(([0,-40], [0, 0]))
    tel_diam = 4.2
    roi_belowGround = 2
    roi_envelope = 4
    nx_subap = numpy.array([7]*n_wfs)
    n_subap = numpy.array([36]*n_wfs)

    pupil_mask = make_pupil_mask('circle', n_subap, nx_subap[0], 
            1., tel_diam)

    onesMat, wfsMat_1, wfsMat_2, allMapPos, selector, xy_separations = roi_referenceArrays(
                pupil_mask, gs_pos, tel_diam, roi_belowGround, roi_envelope)
    
    shwfs_centroids = fits.getdata('../../../../windProfiling/wind_paper/canary/data/test_fits/canary_noNoise_it10k_nl3_h0a10a20km_r00p1_L025_ws10a15a20_wd260a80a350_infScrn_wss448_gsPos0cn40a0c0a30c0.fits')#[:, :72*2]
    # shwfs_centroids = numpy.ones((10000, 36*2*3))
    covMapDim = 13
    roi_axis = 'x and y'
    mapping_type = 'mean'

    nr, nt = calculate_roi_covariance(shwfs_centroids, allMapPos, covMapDim, n_subap, onesMat, wfsMat_1, wfsMat_2, selector, roi_axis, mapping_type)
    print('Time taken: {}'.format(nt))