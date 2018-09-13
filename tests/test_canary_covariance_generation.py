import capt
import numpy
from astropy.io import fits
from matplotlib import pyplot; pyplot.ion()


def generate_covariance(configuration, air_mass, tas, pix_arc, 
        shwfs_centroids, input_matrix=False):
    
    conf = capt.turbulence_profiler(configuration)
    
    mat = capt.turbulence_profiler.make_covariance_matrix(
        conf, air_mass, tas, conf.layer_alt, conf.guess_r0, conf.guess_L0, conf.tt_track, 
        False, conf.shwfs_shift, conf.shwfs_rot, l3s1_transform=False, target_array='Covariance Matrix', 
        tt_track_present=False, offset_present=False, fitting_L0=True)

    # roi = capt.turbulence_profiler.make_covariance_roi(
    #     conf, air_mass, tas, conf.layer_alt, conf.guess_r0, conf.guess_L0, conf.tt_track, 
    #     False, conf.shwfs_shift, conf.shwfs_rot, l3s1_transform=False, tt_track_present=True, 
    #     offset_present=True, fitting_L0=True)

    pyplot.figure()
    pyplot.imshow(mat)
    # pyplot.figure()
    # pyplot.imshow(roi)
    return mat#, roi

if __name__ == '__main__':

    air_mass = 1.
    pix_arc = numpy.nan
    shwfs_centroids = numpy.nan
    canary_tas = numpy.array([(0, 0), (0, 40.)])
    configuration = capt.configuration('../conf/canary_example_conf.yaml')

    m = generate_covariance(
        configuration, air_mass, canary_tas, pix_arc, shwfs_centroids)