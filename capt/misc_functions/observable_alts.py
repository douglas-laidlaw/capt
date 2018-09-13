import numpy
import itertools
from scipy.special import comb as combinations

def observable_alts(gs_pos, telDiam, air_mass):
    """Calculates minimum and maximum h_max for a given GS asterism
    - determining altitudes which are visible within a covariance map.
    
    Parameters:
        gs_pos (ndarray): GS asterism in telescope FoV.
        tel_diam (float): telescope diameter.
        air_mass (float): observation's air mass.
    
    Returns:
        float: minimum GS asterism h_max.
        float: maximum GS asterism h_max."""

    combs = int(combinations(gs_pos.shape[0], 2, exact=True))
    selector = numpy.array((range(gs_pos.shape[0])))
    selector = numpy.array((list(itertools.combinations(selector, 2))))   
    
    minAlt = 1e20
    maxAlt = 0.
    for comb in range(combs):
        gs_pos0 = gs_pos[selector[comb, 0]]
        gs_pos1 = gs_pos[selector[comb, 1]]

        sep = numpy.sqrt(((gs_pos0-gs_pos1)**2).sum())
        sep *= numpy.pi/180./3600.

        maxObsAlt = (telDiam/sep) / air_mass

        if maxObsAlt>maxAlt:
            maxAlt = maxObsAlt
        if maxObsAlt<minAlt:
            minAlt = maxObsAlt

    return minAlt, maxAlt