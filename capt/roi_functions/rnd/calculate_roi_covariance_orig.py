
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



def configure_cent_data(shwfs_centroids):
    """Re-shapes SHWFS centroid array and subtracts mean at each iteration (first step in calculating cross-covariance).
    
    Parameters:
        shwfs_centroids (ndarray): SHWFS centroid measurements.

    Returns:
        ndarray: re-shaped form SHWFS centroid measurements input with mean subtracted (first step in calculating covariance)."""

    slopeData = numpy.zeros((shwfs_centroids.shape[1], shwfs_centroids.shape[0]))
    for i in range(slopeData.shape[0]):
        slopeData[i] = shwfs_centroids[:,i] - numpy.mean(shwfs_centroids[:,i])
    return slopeData




def calculate_roi_covariance(shwfs_centroids, allMapPos, covMapDim, n_subap, mm, sa_mm, sb_mm, selector, roi_axis, mapping_type):
    """Takes SHWFS centroids and directly calculates the covariance map ROI (doesn't require going via covariance matrix).
    
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
        roi_covariance (ndarray): covariance map ROI."""

    timeStart = time.time()
    shwfs_centroids = configure_cent_data(shwfs_centroids)

    if roi_axis=='x' or roi_axis=='y' or roi_axis=='x+y':
        roi_covariance = numpy.zeros((allMapPos.shape[0]*allMapPos.shape[1], allMapPos.shape[2]))
    if roi_axis=='x and y':
        roi_covariance = numpy.zeros((allMapPos.shape[0]*allMapPos.shape[1], allMapPos.shape[2]*2))

    #Cycle over no. GS combinations
    for k in range(allMapPos.shape[0]):
        if roi_axis=='x' or roi_axis=='y' or roi_axis=='x+y':
            single_roi_covariance = numpy.zeros((allMapPos.shape[1], allMapPos.shape[2]))
        if roi_axis=='x and y':
            single_roi_covariance = numpy.zeros((allMapPos.shape[1], allMapPos.shape[2]*2)) 

        wfs1_n_subap = n_subap[selector[k, 0]]
        wfs2_n_subap = n_subap[selector[k, 1]]

        #Cycle over covariance map ROI width
        for i in range(allMapPos.shape[1]):
            #Cycle over sub-aperture separations
            for j in range(allMapPos.shape[2]):
                #Dismiss positions that aren't within the covariance map
                if allMapPos[k,i,j,0] != 2*covMapDim:
                    
                    #Find the sub-aperture numbers that corresponds to the required sub-aperture separation 
                    loc = numpy.where(mm[:, allMapPos[k,i,j,1] + (allMapPos[k,i,j,0])*covMapDim]==1)[0]
                    subapsWfs1 =  sa_mm[:, allMapPos[k,i,j,1] + (allMapPos[k,i,j,0])*covMapDim][loc]
                    subapsWfs2 =  sb_mm[:, allMapPos[k,i,j,1] + (allMapPos[k,i,j,0])*covMapDim][loc]
                    
                    # subapsWfs2 =  sb_mm[:, allMapPos[k,:,:,1] + (allMapPos[k,:,:,0])*covMapDim][loc]


                    #Convert from sub-aperture no. to x-slope no. within the respective WFSs
                    subapsXX1 = subapsWfs1 + (selector[k][0]*2*wfs1_n_subap)
                    subapsXX2 = subapsWfs2 + (selector[k][1]*2*wfs2_n_subap)
                    #Convert from sub-aperture no. to y-slope no. within the respective WFSs
                    subapsYY1 = subapsWfs1 + (selector[k][0]*2*wfs1_n_subap) + wfs1_n_subap
                    subapsYY2 = subapsWfs2 + (selector[k][1]*2*wfs2_n_subap) + wfs2_n_subap

                    if mapping_type=='mean':
                        
                        #Calculates the mean of xx + yy covariance
                        if roi_axis=='x+y':
                            meanXX = numpy.mean((shwfs_centroids[subapsXX1] * shwfs_centroids[subapsXX2]).sum(1)/(shwfs_centroids.shape[1]-1))
                            meanYY = numpy.mean((shwfs_centroids[subapsYY1] * shwfs_centroids[subapsYY2]).sum(1)/(shwfs_centroids.shape[1]-1))
                            single_roi_covariance[i,j] = (meanXX + meanYY)/2.
                        #Calculates the mean covariance in the x-axis
                        if roi_axis=='x':
                            single_roi_covariance[i,j] = numpy.mean((shwfs_centroids[subapsXX1] * shwfs_centroids[subapsXX2]).sum(1)/(shwfs_centroids.shape[1]-1))
                        #Calculates the mean covariance in the y-axis
                        if roi_axis=='y':
                            single_roi_covariance[i,j] = numpy.mean((shwfs_centroids[subapsYY1] * shwfs_centroids[subapsYY2]).sum(1)/(shwfs_centroids.shape[1]-1))
                        #Calculates the mean covariance in the x and y axes and keeps them separate
                        if roi_axis=='x and y':
                            single_roi_covariance[i,j] = numpy.mean((shwfs_centroids[subapsXX1]*shwfs_centroids[subapsXX2]).sum(1)/(shwfs_centroids.shape[1]-1))
                            single_roi_covariance[i,j+allMapPos.shape[2]] = numpy.mean((shwfs_centroids[subapsYY1] * shwfs_centroids[subapsYY2]).sum(1)/(shwfs_centroids.shape[1]-1))

                    if mapping_type=='median':
                        #Calculates the mean of xx + yy covariance
                        if roi_axis=='x+y':
                            medianXX = numpy.median((shwfs_centroids[subapsXX1] * shwfs_centroids[subapsXX2]).sum(1)/(shwfs_centroids.shape[1]-1))
                            medianYY = numpy.median((shwfs_centroids[subapsYY1] * shwfs_centroids[subapsYY2]).sum(1)/(shwfs_centroids.shape[1]-1))
                            single_roi_covariance[i,j] = (medianXX + medianYY)/2.
                        #Calculates the mean covariance in the x-axis
                        if roi_axis=='x':
                            single_roi_covariance[i,j] = numpy.median((shwfs_centroids[subapsXX1] * shwfs_centroids[subapsXX2]).sum(1)/(shwfs_centroids.shape[1]-1))
                        #Calculates the mean covariance in the y-axis
                        if roi_axis=='y':
                            single_roi_covariance[i,j] = numpy.median((shwfs_centroids[subapsYY1] * shwfs_centroids[subapsYY2]).sum(1)/(shwfs_centroids.shape[1]-1))
                        #Calculates the mean covariance in the x and y axes and keeps them separate
                        if roi_axis=='x and y':
                            single_roi_covariance[i,j] = numpy.median((shwfs_centroids[subapsXX1]*shwfs_centroids[subapsXX2]).sum(1)/(shwfs_centroids.shape[1]-1))
                            single_roi_covariance[i,j+allMapPos.shape[2]] = numpy.median((shwfs_centroids[subapsYY1] * shwfs_centroids[subapsYY2]).sum(1)/(shwfs_centroids.shape[1]-1))

        roi_covariance[k*allMapPos.shape[1]: (k+1)*allMapPos.shape[1]] = single_roi_covariance

    timeStop = time.time()
    time_taken = timeStop - timeStart
    # print('\n')
    # print('################ ROI COVARIANCE CALCULATED ################','\n')
    # print('################### TIME TAKEN : '+"%6.4f" % time_taken+' ###################')
    # print('################### TIME TAKEN : '+str(numpy.round(time_taken,4))+' ###################','\n')
    return roi_covariance, time_taken
    


if __name__=='__main__':
    n_wfs = 3
    gs_pos = numpy.array(([0,-40], [0, 0], [30,0]))
    tel_diam = 4.2
    roi_belowGround = 2
    roi_envelope = 4
    nx_subap = numpy.array([7]*n_wfs)
    n_subap = numpy.array([36]*n_wfs)

    pupil_mask = make_pupil_mask('circle', n_subap, nx_subap[0], 
            1., tel_diam)

    onesMat, wfsMat_1, wfsMat_2, allMapPos, selector, xy_separations = roi_referenceArrays(
                pupil_mask, gs_pos, tel_diam, roi_belowGround, roi_envelope)
    
    shwfs_centroids = fits.getdata('../../../../windProfiling/wind_paper/canary/data/test_fits/canary_noNoise_it10k_nl3_h0a10a20km_r00p1_L025_ws10a15a20_wd260a80a350_infScrn_wss448_gsPos0cn40a0c0a30c0.fits')
    covMapDim = 13
    roi_axis = 'x and y'
    mapping_type = 'mean'

    r, t = calculate_roi_covariance(shwfs_centroids, allMapPos, covMapDim, n_subap, onesMat, wfsMat_1, wfsMat_2, selector, roi_axis, mapping_type)
    print('Time taken: {}'.format(t))