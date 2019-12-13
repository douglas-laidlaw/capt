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
    
    roi = capt.turbulence_profiler.make_covariance_roi(
        conf, conf.pupil_mask, air_mass, tas, conf.gs_alt, conf.layer_alt, conf.guess_r0, conf.guess_L0, conf.tt_track, 
        False, conf.shwfs_shift, conf.shwfs_rot, l3s1_transform=False, tt_track_present=False, 
        lgs_track_present=False, offset_present=True)
    mat=0

    # mat = capt.turbulence_profiler.make_covariance_matrix(
    #     conf, conf.pupil_mask, air_mass, tas, conf.gs_alt, conf.layer_alt, conf.guess_r0, conf.guess_L0, conf.tt_track, 
    #     False, conf.shwfs_shift, conf.shwfs_rot, l3s1_transform=False, target_array='Covariance Map ROI', 
    #     tt_track_present=False, lgs_track_present=False, offset_present=False)

    return mat, roi

if __name__ == '__main__':

    air_mass = 1.
    pix_arc = numpy.nan
    shwfs_centroids = numpy.nan
    canary_tas = numpy.array([(-64, 0), (64., 0.)])
    configuration = capt.configuration('../conf/aof_example_conf.yaml')

    matrix, roi = generate_covariance(configuration, air_mass, canary_tas, pix_arc, shwfs_centroids)

    # pyplot.figure('matrix')
    # pyplot.imshow(matrix)

    pyplot.figure()
    pyplot.imshow(roi)