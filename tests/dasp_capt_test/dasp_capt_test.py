import capt
import numpy
from astropy.io import fits
from matplotlib import pyplot; pyplot.ion()


def get_turbulenceProfile_information(configuration, air_mass, tas, pix_arc, 
    shwfs_centroids, input_matrix=False):
    
    conf = capt.turbulence_profiler(configuration)
    results = conf.perform_turbulence_profiling(air_mass, tas, pix_arc, shwfs_centroids)

    generate_matrix = capt.turbulence_profiler.make_covariance_matrix(
        conf, air_mass, tas, conf.layer_alt, results.r0, results.L0, results.tt_track, 
        results.lgs_track, results.shwfs_shift, results.shwfs_rot, l3s1_transform=False, target_array='Covariance Matrix', 
        tt_track_present=results.tt_track_present, offset_present=results.offset_present)
    
    return results, generate_matrix

if __name__ == '__main__':

    air_mass = 1.
    pix_arc = numpy.nan
    tas = numpy.array([(40, 0), (0, 0)])
    configuration = capt.configuration('dasp_capt_conf.yaml')
    cents_wfs1 = fits.getdata('saveOutput_1b0.fits').byteswap()
    cents_wfs2 = fits.getdata('saveOutput_2b0.fits').byteswap()
    shwfs_centroids = numpy.stack((cents_wfs1, cents_wfs2))

    results, cov_matrix  = get_turbulenceProfile_information(configuration, air_mass, tas, pix_arc, shwfs_centroids)

    pyplot.figure('meas')
    pyplot.imshow(results.cov_meas)

    pyplot.figure('fit')
    pyplot.imshow(results.cov_fit)

    pyplot.figure('fit matrix')
    pyplot.imshow(cov_matrix)