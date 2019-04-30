import capt
import numpy
from astropy.io import fits
from matplotlib import pyplot; pyplot.ion()
import capt.misc_functions.matplotlib_format


def get_turbulenceProfile_information(configuration, air_mass, tas, pix_arc, 
    shwfs_centroids, input_matrix=False):
    
    conf = capt.turbulence_profiler(configuration)
    input_matrix = capt.turbulence_profiler.make_covariance_matrix(conf, conf.pupil_mask, air_mass, tas, 
        conf.gs_alt, numpy.array([0]), numpy.array([0.8]), conf.guess_L0, conf.tt_track, None, conf.shwfs_shift, conf.shwfs_rot)

    conf = capt.turbulence_profiler(configuration)
    results = conf.perform_turbulence_profiling(air_mass, tas, pix_arc, shwfs_centroids, cov_matrix=input_matrix)
    
    return results

if __name__ == '__main__':

    air_mass = 1.
    pix_arc = numpy.nan
    shwfs_centroids = numpy.nan
    tas = numpy.array([(0, 0), (0, 40)])
    configuration = capt.configuration('../conf/canary_example_conf.yaml')

    r = get_turbulenceProfile_information(configuration, air_mass, tas, pix_arc, shwfs_centroids)

    pyplot.figure()
    pyplot.imshow(r.cov_meas)

    

