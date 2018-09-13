import numpy
import aotools
from matplotlib import pyplot; pyplot.ion()

def make_pupil_mask(mask, n_subap, nx_subap, obs_diam, tel_diam):
    """Generates a SOAPY pupil mask
    - code has been adapted from SOAPY make_mask.py

    Parameters:
        mask (str): must be 'circle'
        nx_subap (int): number of sub-apertures across telescope's pupil.
        obs_diam (float): diameter of central obscuration.
        tel_diam (float): diameter of telescope pupil.

    Returns:
        ndarray: mask of SHWFS sub-apertures within the telescope's pupil."""

    if mask == "circle":
        pupil_mask = aotools.circle(nx_subap / 2.,
                                  nx_subap)
        if obs_diam != None:
            pupil_mask -= aotools.circle(
                nx_subap * ((obs_diam/tel_diam) / 2.),
                nx_subap)

    else:
        raise Exception('Only circlular pupil masks have been integrated (sorry).')

    if n_subap[0]!=int(numpy.sum(pupil_mask)):
        raise Exception('Error in the number of sub-apertures within pupil mask.')

    return pupil_mask.astype(int)

if __name__ == '__main__':
    # """CANARY"""
    # obs_diam = 1.
    # tel_diam = 4.2
    # nx_subap = 7
    # n_subap = numpy.array([36])
    # mask = 'circle'
    # p = make_pupil_mask(mask, n_subap, nx_subap, obs_diam, tel_diam)
    
    
    """AOF"""
    obs_diam = 1.1
    tel_diam = 8.2
    nx_subap = 40
    n_subap = numpy.array([1240])
    mask = 'circle'
    p = make_pupil_mask(mask, n_subap, nx_subap, obs_diam, tel_diam)

    # """HARMONI"""
    # obs_diam = 4
    # tel_diam = 39
    # nx_subap = 74
    # n_subap = numpy.array([4260])
    # mask = 'circle'
    # p = make_pupil_mask(mask, n_subap, nx_subap, obs_diam, tel_diam)