import capt
import time
import numpy
from astropy.io import fits
from matplotlib import pyplot; pyplot.ion()
import capt.misc_functions.matplotlib_format


def generate_covariance(configuration, air_mass, tas, pix_arc, 
        shwfs_centroids, input_matrix=False):
    
    conf = capt.turbulence_profiler(configuration)
    
    st = time.time()
    mat = capt.turbulence_profiler.make_covariance_matrix(
        conf, conf.pupil_mask, air_mass, tas, conf.gs_alt, conf.layer_alt, conf.guess_r0, conf.guess_L0, conf.tt_track, 
        False, conf.shwfs_shift, conf.shwfs_rot, l3s1_transform=False, target_array='Covariance Matrix', 
        tt_track_present=False, offset_present=False)
    print(time.time() - st)

    roi = numpy.nan
    roi = capt.turbulence_profiler.make_covariance_roi(
        conf, conf.pupil_mask, air_mass, tas, conf.gs_alt, conf.layer_alt, conf.guess_r0, conf.guess_L0, conf.tt_track, 
        False, conf.shwfs_shift, conf.shwfs_rot, l3s1_transform=False, 
        tt_track_present=True, offset_present=False)

    pyplot.figure()
    pyplot.imshow(mat)
    pyplot.figure()
    pyplot.imshow(roi)
    return mat, roi

if __name__ == '__main__':

    air_mass = 1.
    pix_arc = numpy.nan
    shwfs_centroids = numpy.nan
    canary_tas = numpy.array([(0, 0), (0, 40.)])
    configuration = capt.configuration('../conf/canary_example_conf.yaml')

    m, r = generate_covariance(configuration, air_mass, canary_tas, pix_arc, shwfs_centroids)