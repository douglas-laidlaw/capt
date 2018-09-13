import numpy
from astropy.io import fits
from matplotlib import pyplot; pyplot
from capt.misc_functions.make_pupil_mask import make_pupil_mask


def dasp_cents_reshape(shwfs_centroids, pupil_mask, n_wfs):
    """Takes DASP SHWFS centroids and reshapes to (n-iterations, 2*n_subaps*n_wfs)

    Parameters:
        shwfs_centroids (ndarray): DASP SHWFS centroid measurements.
        pupil_mask (ndarray): pupil mask.
        n_wfs (int): number of SHWFSs.

    Returns:
        ndarray: re-shaped SHWFS centroid array."""
   
    output_shwfs_centroids = numpy.zeros((shwfs_centroids.shape[1], int(pupil_mask.sum()) * 2 * n_wfs))
    pupil_flat = pupil_mask.flatten()
    n_subap = int(pupil_mask.sum())
    
    for wfs_n in range(shwfs_centroids.shape[0]):
        axis0_cents = shwfs_centroids[wfs_n,...,0]
        axis1_cents = shwfs_centroids[wfs_n,...,1]
        
        for it in range(shwfs_centroids.shape[1]):
            output_shwfs_centroids[it, 2 * n_subap * wfs_n: (2 * n_subap * wfs_n) + n_subap] = axis0_cents[it, pupil_mask==1].flatten()
            output_shwfs_centroids[it, (2 * n_subap * wfs_n) + n_subap: (2 * n_subap * wfs_n) + (n_subap*2)] = axis1_cents[it, pupil_mask==1].flatten()

    return output_shwfs_centroids


if __name__ == "__main__":
    cents_wfs1 = fits.getdata('saveOutput_1b0.fits').byteswap()
    cents_wfs2 = fits.getdata('saveOutput_2b0.fits').byteswap()
    #or, this would be faster:
    #cents_wfs1 = util.FITS.Read('saveOutput_1b0.fits')[1]

    p = make_pupil_mask('circle', numpy.array([36, 36]), 7, 1., 4.2)

    shwfs_centroids = numpy.stack((cents_wfs1, cents_wfs2))
    o = dasp_cents_reshape(shwfs_centroids, p, 2)

    c = crossCov(o)
    pyplot.figure()
    pyplot.imshow(c)