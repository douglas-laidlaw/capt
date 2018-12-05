import capt
import numpy
from astropy.io import fits
from esoTurbProfile import esoTurbProfile
from matplotlib import pyplot; pyplot.ion()
import capt.misc_functions.matplotlib_format
from capt.misc_functions.make_pupil_mask import make_pupil_mask


def generate_covariance(configuration, air_mass, tas, pix_arc, 
        shwfs_centroids, input_matrix=False):
    
    conf = capt.turbulence_profiler(configuration)
    pupil_mask = fits.getdata('/Volumes/PHD FILES/aof_data/pols/2017-06-15/lgsLoopData_102/reduced_pupilMask.fits')
    conf.pupil_mask = pupil_mask
    
    roi = 0
    roi = capt.turbulence_profiler.make_covariance_roi(
        conf, conf.pupil_mask, air_mass, tas, conf.layer_alt, conf.guess_r0, conf.guess_L0, conf.tt_track, 
        False, conf.shwfs_shift, conf.shwfs_rot, l3s1_transform=False, tt_track_present=False, 
        lgs_track_present=False, offset_present=True)

    mat=0
    mat = capt.turbulence_profiler.make_covariance_matrix(
        conf, conf.pupil_mask, air_mass, tas, conf.layer_alt, conf.guess_r0, conf.guess_L0, conf.tt_track, 
        False, conf.shwfs_shift, conf.shwfs_rot, l3s1_transform=False, target_array='Covariance Map ROI', 
        tt_track_present=False, lgs_track_present=False, offset_present=True)

    # pyplot.figure()
    # pyplot.imshow(mat)
    # pyplot.figure()
    # pyplot.imshow(roi)
    return mat, roi

if __name__ == '__main__':

    air_mass = 1.
    pix_arc = numpy.nan
    shwfs_centroids = numpy.nan
    canary_tas = numpy.array([(-64, 0), (64., 0.)])
    configuration = capt.configuration('../conf/aof_example_conf.yaml')

    matrix, roi = generate_covariance(
        configuration, air_mass, canary_tas, pix_arc, shwfs_centroids)

    pupil_mask = fits.getdata('/Volumes/PHD FILES/aof_data/pols/2017-06-15/lgsLoopData_102/reduced_pupilMask.fits')
    auto_map = fits.getdata('/Volumes/PHD FILES/aof_data/pols/2017-06-15/lgsLoopData_102/pols_wfs1_auto-covarianceMap.fits')

    # roi = numpy.hstack((numpy.rot90(roi[:,:79],1), numpy.rot90(roi[:,79:],1)))

    # pyplot.figure('diff')
    # diff = matrix-roi
    # diff[diff==0] = numpy.nan
    # pyplot.imshow(diff)

    pyplot.figure('matrix')
    matrix[matrix==0] = numpy.nan
    # matrix[numpy.abs(matrix)>0] = 1
    pyplot.imshow(matrix)

    pyplot.figure('roi')
    roi[roi==0] = numpy.nan
    # roi[numpy.abs(roi)>0] = 1
    pyplot.imshow(roi)

    # pyplot.figure()
    # pyplot.imshow(matrix-roi)