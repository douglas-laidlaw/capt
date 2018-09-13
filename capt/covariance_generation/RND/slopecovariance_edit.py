"""
Slope Covariance Matrix Generation
----------------------------------

WARNING!!! NO XY COVARIANCE

Slope covariance matrix routines for AO systems observing through Von Karmon turbulence. Such matrices have a variety
of uses, though they are especially useful for creating 'tomographic reconstructors' that can reconstruct some 'psuedo'
WFS measurements in a required direction (where there might be an interesting science target but no guide stars),
given some actual measurements in other directions (where the some suitable guide stars are).

NOTE: THIS CODE HAS BEEN TESTED QUALITIVELY AND SEEMS OK...NEEDS MORE RIGOROUS TESTING!

.. codeauthor:: Andrew Reeves <a.p.reeves@durham.ac.uk>

"""

import myFont
import time
import multiprocessing

import numpy
import scipy.special
import itertools
from scipy.misc import comb

from aotools.functions import pupil
from matplotlib import pyplot; pyplot.ion()
from covMapsFromMatrix import covMapsFromMatrix
# from covarianceMapSlice import covarianceMapSlice


class CovarianceMatrix(object):
    """
    A creator of slope covariance matrices in Von Karmon turbulence, based on the paper by Martin et al, SPIE, 2012.

    Given a list of paramters describing an AO WFS system and the atmosphere above the telescope, this class can
    compute the covariance matrix between all the WFS measurements. This can support LGS sources that exist at a
    finite altitude. When computing the covariance matrix, Python's multiprocessing module is used to spread the work
    between different processes and processing cores.

    On initialisation, this class performs some initial calculations and parameter sorting. To create the
    covariance matrix run the ``make_covariace_matrix`` method. This may take some time depending on your paramters...

    Parameters:
        n_wfs (int): Number of wavefront sensors present.
        pupil_masks (ndarray): A map of the pupil for each WFS which is nx_subaps by ny_subaps. 1 if subap active, 0 if not.
        telescope_diameter (float): Diameter of the telescope
        subap_diameters (ndarray): The diameter of the sub-apertures for each WFS in metres
        gs_altitudes (ndarray): Reciprocal (1/metres) of the Guide star alitude for each WFS
        gs_positions (ndarray): X,Y position of each WFS in arcsecs. Array shape (Wfs, 2)
        wfs_wavelengths (ndarray): Wavelength each WFS observes
        n_layers (int): The number of atmospheric turbulence layers
        layer_altitudes (ndarray): The altitude of each turbulence layer in meters
        layer_r0s (ndarray): The Fried parameter of each turbulence layer
        layer_L0s (ndarray): The outer-scale of each layer in metres
        threads (int, optional): Number of processes to use for calculation. default is 1
    """

    def __init__(
        self, n_wfs, pupil_masks, telescope_diameter, subap_diameters, gs_altitudes, gs_positions, wfs_wavelengths,
        n_layers, layer_altitudes, layer_r0s, layer_L0s, stycMethod, threads=1):

        self.n_wfs = n_wfs
        self.subap_diameters = subap_diameters
        self.wfs_wavelengths = wfs_wavelengths
        self.telescope_diameter = telescope_diameter
        self.pupil_masks = pupil_masks
        self.gs_altitudes = gs_altitudes
        self.gs_positions = gs_positions[:,::-1]        # Swap X and Y GS positions to agree with legacy code

        self.n_layers = n_layers
        self.layer_altitudes = layer_altitudes
        self.layer_r0s = layer_r0s
        self.layer_L0s = layer_L0s
        # print("n_layers: {}".format(n_layers))

        self.n_subaps = []
        self.total_subaps = 0
        for wfs_n in range(n_wfs):
            self.n_subaps.append(pupil_masks[wfs_n].sum())
            self.total_subaps += self.n_subaps[wfs_n]
        self.n_subaps = numpy.array(self.n_subaps, dtype="int")
        self.total_subaps = int(self.total_subaps)
        self.stycMethod = stycMethod
        self.radSqaured_to_arcsecSqaured = ((180./numpy.pi) * 3600)**2


    def make_covariance_matrix(self):
        """
        Calculate and build the covariance matrix

        Returns:
            ndarray: Covariance Matrix
        """
        # make a list of the positions of the centre of every sub-aperture
        # for each WFS un units of metres from the centre of the pupil
        self.subap_positions = numpy.zeros((self.n_wfs, self.n_subaps[0], 2)) 

        for wfs_n in range(self.n_wfs):
            wfs_subap_pos = (numpy.array(numpy.where(self.pupil_masks[wfs_n] == 1)).T * self.subap_diameters[wfs_n])
            self.subap_positions[wfs_n] = wfs_subap_pos

        # Create a list with n_layers elements, each of which is a list of the WFS meta-subap positions at that altitude
        
        self.subap_layer_positions = numpy.zeros((self.n_layers, self.subap_positions.shape[0], self.subap_positions.shape[1], self.subap_positions.shape[2]))
        self.subap_layer_diameters = numpy.zeros((self.n_layers, self.n_wfs))

        for layer_n, layer_altitude in enumerate(self.layer_altitudes):
            subap_n = 0
            wfs_pos = []
            wfs_subap_diameters = []
            for wfs_n in range(self.n_wfs):

                # Scale for LGS
                if self.gs_altitudes[wfs_n] != 0:
                    # print("Its an LGS!")
                    scale_factor = (1 - layer_altitude/self.gs_altitudes[wfs_n])
                    positions = scale_factor * self.subap_positions[wfs_n].copy()
                else:
                    scale_factor = 1.
                    positions = self.subap_positions[wfs_n].copy()

                # translate due to GS position
                gs_pos_rad = numpy.array(self.gs_positions[wfs_n]) * (numpy.pi/180.) * (1./3600.)
                # print("GS Positions: {} rad".format(gs_pos_rad))
                translation = gs_pos_rad * layer_altitude

                # print("Translation: {} m".format(translation))
                # print("Max position before translation: {} m".format(abs(positions).max()))
                positions = translation + positions
                # print("Max position: {} m".format(abs(positions).max()))

                self.subap_layer_diameters[layer_n, wfs_n] = self.subap_diameters[wfs_n] * scale_factor
                self.subap_layer_positions[layer_n, wfs_n] = positions

        self._make_covariance_matrix()
        self.covariance_matrix = mirror_covariance_matrix(self.covariance_matrix, self.n_subaps)

        return self.covariance_matrix * self.radSqaured_to_arcsecSqaured

    def _make_covariance_matrix(self):

        # Now compile the covariance matrix
        self.covariance_matrix = numpy.zeros((2 * self.total_subaps, 2 * self.total_subaps))
        for layer_n in range(self.n_layers):
            # print("Compute Layer {}".format(layer_n))

            subap_ni = 0
            for wfs_i in range(self.n_wfs):
                subap_nj = 0
                # Only loop over upper diagonal of covariance matrix as its symmetrical
                for wfs_j in range(wfs_i+1):
                    cov_xx, cov_yy, cov_xy = wfs_covariance(
                            self.n_subaps[wfs_i], self.n_subaps[wfs_j],
                            self.subap_layer_positions[layer_n, wfs_i], self.subap_layer_positions[layer_n, wfs_j],
                            self.subap_layer_diameters[layer_n, wfs_i], self.subap_layer_diameters[layer_n, wfs_j],
                            self.layer_r0s[layer_n], self.layer_L0s[layer_n], self.stycMethod)

                    subap_ni = self.n_subaps[:wfs_i].sum()
                    subap_nj = self.n_subaps[:wfs_j].sum()

                    # Coordinates of the XX covariance
                    cov_mat_coord_x1 = subap_ni * 2
                    cov_mat_coord_x2 = subap_ni * 2 + self.n_subaps[wfs_i]

                    cov_mat_coord_y1 = subap_nj * 2
                    cov_mat_coord_y2 = subap_nj * 2 + self.n_subaps[wfs_j]
                    
                    # # # print("covmat coords: ({}: {}, {}: {})".format(cov_mat_coord_x1, cov_mat_coord_x2, cov_mat_coord_y1, cov_mat_coord_y2))
                    r0_scale = ((self.wfs_wavelengths[wfs_i] * self.wfs_wavelengths[wfs_j])
                            / (8 * (numpy.pi**2) * self.subap_layer_diameters[layer_n][wfs_i] * self.subap_layer_diameters[layer_n][wfs_j] * self.layer_r0s[layer_n]**(5./3))
                                )

                    self.covariance_matrix[
                            cov_mat_coord_x1: cov_mat_coord_x2, cov_mat_coord_y1: cov_mat_coord_y2
                            ] += cov_xx * r0_scale

                    self.covariance_matrix[
                            cov_mat_coord_x1 + self.n_subaps[wfs_i]: cov_mat_coord_x2 + self.n_subaps[wfs_i],
                            cov_mat_coord_y1: cov_mat_coord_y2] += cov_xy * r0_scale
                    self.covariance_matrix[
                            cov_mat_coord_x1: cov_mat_coord_x2,
                            cov_mat_coord_y1 + self.n_subaps[wfs_j]: cov_mat_coord_y2 + self.n_subaps[wfs_j]
                            ] += cov_xy * r0_scale

                    self.covariance_matrix[
                            cov_mat_coord_x1 + self.n_subaps[wfs_i]: cov_mat_coord_x2 + self.n_subaps[wfs_i],
                            cov_mat_coord_y1 + self.n_subaps[wfs_j]: cov_mat_coord_y2 + self.n_subaps[wfs_j]
                            ] += cov_yy * r0_scale





def wfs_covariance(n_subaps1, n_subaps2, wfs1_positions, wfs2_positions, wfs1_diam, wfs2_diam, r0, L0, stycMethod):
    """
    Calculates the covariance between 2 WFSs

    Parameters:
        n_subaps1 (int): number of sub-apertures in WFS 1
        n_subaps2 (int): number of sub-apertures in WFS 1
        wfs1_positions (ndarray): Central position of each sub-apeture from telescope centre for WFS 1
        wfs2_positions (ndarray): Central position of each sub-apeture from telescope centre for WFS 2
        wfs1_diam: Diameter of WFS 1 sub-apertures
        wfs2_diam: Diameter of WFS 2 sub-apertures
        r0: Fried parameter of turbulence
        L0: Outer scale of turbulence
        stycMethod: If True use phase structure function differencing method, otherwise use numerical integration method

    Returns:
        slope covariance of X with X , slope covariance of Y with Y, slope covariance of X with Y
    """        

    xy_separations = calculate_wfs_separations(n_subaps1, n_subaps2, wfs1_positions, wfs2_positions)

    if stycMethod == True:
        # print("Min separation: {}".format(abs(xy_separations).min()))
        cov_xx = compute_covariance_xx(xy_separations, wfs1_diam*0.5, wfs2_diam*0.5, r0, L0)
        cov_yy = compute_covariance_yy(xy_separations, wfs1_diam*0.5, wfs2_diam*0.5, r0, L0)
        cov_xy = compute_covariance_xy(xy_separations, wfs1_diam*0.5, wfs2_diam*0.5, r0, L0)
        
        return cov_xx, cov_yy, cov_xy

    else:
        assert abs(wfs1_diam - wfs2_diam) < 1.e-10    # numerical integration code doesn't support different subap sizes
        d = float(wfs1_diam)
        maxDelta = int(numpy.abs(xy_separations).max()/d)+1
        sf_dx = d/100.
        sf_n = int(4*maxDelta*(d/sf_dx))
        sf = structure_function_vk(numpy.arange(sf_n)*sf_dx, r0, L0)
        sf[0] = 0.    #get rid of nan

        nSamp = 8    #hard-wired parameter
        cov = compute_ztilt_covariances(xy_separations, sf, sf_dx, nSamp, d)

        return cov[0],cov[2],cov[1]



def calculate_wfs_separations(n_subaps1, n_subaps2, wfs1_positions, wfs2_positions):
    """
    Calculates the separation between all sub-apertures in two WFSs

    Parameters:
        n_subaps1 (int): Number of sub-apertures in WFS 1
        n_subaps2 (int): Number of sub-apertures in WFS 2
        wfs1_positions (ndarray): Array of the X, Y positions of the centre of each sub-aperture with respect to the centre of the telescope pupil
        wfs2_positions (ndarray): Array of the X, Y positions of the centre of each sub-aperture with respect to the centre of the telescope pupil

    Returns:
        ndarray: 2-D Array of sub-aperture separations
    """


    xy_separations = numpy.zeros((n_subaps1, n_subaps2, 2)) + 1e-20

    for i, (x2, y2) in enumerate(wfs1_positions):
        for j, (x1, y1) in enumerate(wfs2_positions):
            xy_separations[i, j] = (x2-x1), (y1-y2)

    return xy_separations


def compute_covariance_yy(separation, subap1_rad, subap2_rad, r0, L0):
    x1 = separation[:, :, 0] + (subap2_rad - subap1_rad) #* 0.5
    r1 = numpy.sqrt(x1**2 + separation[:, :, 1]**2)

    x2 = separation[:, :, 0] - (subap2_rad + subap1_rad) #* 0.5
    r2 = numpy.sqrt(x2**2 + separation[:, :, 1]**2)

    x3 = separation[:, :, 0] + (subap2_rad + subap1_rad) #* 0.5
    r3 = numpy.sqrt(x3**2 + separation[:, :, 1]**2)

    x4 = separation[:, :, 0] - (subap2_rad - subap1_rad)
    r4 = numpy.sqrt(x4**2 + separation[:, :, 1]**2)

    Cxx = (- new_structure_function_vk(r1, L0)
            + new_structure_function_vk(r2, L0)
            + new_structure_function_vk(r3, L0)
            - new_structure_function_vk(r4, L0)
           )

    return Cxx

def compute_covariance_xx(separation, subap1_rad, subap2_rad, r0, L0):
    y1 = separation[..., 1] + (subap2_rad - subap1_rad) #* 0.5
    r1 = numpy.sqrt(separation[..., 0]**2 + y1**2)

    y2 = separation[..., 1] - (subap2_rad + subap1_rad) #* 0.5
    r2 = numpy.sqrt(separation[..., 0]**2 + y2**2)

    y3 = separation[..., 1] + (subap2_rad + subap1_rad) #* 0.5
    r3 = numpy.sqrt(separation[..., 0]**2 + y3**2)

    y4 = separation[..., 1] - (subap2_rad - subap1_rad) #* 0.5
    r4 = numpy.sqrt(separation[..., 0]**2 + y4**2)

    Cyy = (- new_structure_function_vk(r1, L0)
           + new_structure_function_vk(r2, L0)
           + new_structure_function_vk(r3, L0)
           - new_structure_function_vk(r4, L0)
           )

    return Cyy

def compute_covariance_xy(seperation, subap1_rad, subap2_rad, r0, L0):
    
    x1 = seperation[..., 0] + subap1_rad #* 0.5
    y1 = seperation[..., 1] - subap2_rad #* 0.5
    r1 = numpy.sqrt(x1**2 + y1**2)

    x2 = seperation[..., 0] - subap1_rad # 0.5
    y2 = seperation[..., 1] + subap2_rad #* 0.5
    r2 = numpy.sqrt(x2**2 + y2**2)

    x3 = seperation[..., 0] + subap1_rad #* 0.5
    y3 = seperation[..., 1] + subap2_rad #* 0.5
    r3 = numpy.sqrt(x3**2 + y3**2)

    x4 = seperation[..., 0] - subap1_rad #* 0.5
    y4 = seperation[..., 1] - subap2_rad #* 0.5
    r4 = numpy.sqrt(x4**2 + y4**2)

    # print("\n",r1, r2, r3, r4)

    Cxy = (new_structure_function_vk(r1, L0)
            + new_structure_function_vk(r2, L0)
            - new_structure_function_vk(r3, L0)
            - new_structure_function_vk(r4, L0)
           )

    return Cxy


def compute_ztilt_covariances(seps, sf, sf_dx, nSamp, d):
    nSub1,nSub2 = seps.shape[0],seps.shape[1]

    scaling = 206265.*206265.*3.* (500.e-9/(numpy.pi*d))**2;   # scaling to arcsec^2
    fudgeFactor = (206265.**2) * (500.e-9**2) / ( 8. * (numpy.pi**2) * (d**2) )    #further scaling to get from arcsec^2 to units used elsewhere in this module

    rxy = (numpy.arange(nSamp) - float(nSamp)/2 + 0.5) / float(nSamp)
    tilt = 2.*(3.**0.5)*rxy
    nSamp2 = float(nSamp**2)
    nSamp4 = float(nSamp**4)
    ra_intgrl = numpy.zeros((nSamp,nSamp),numpy.float)
    rb_intgrl = numpy.zeros((nSamp,nSamp),numpy.float)
    Dphi = numpy.zeros((nSamp,nSamp,nSamp,nSamp),numpy.float)
    cov = numpy.zeros((3,nSub1,nSub2),numpy.float)

    for i in range(nSub1):
        for j in range(nSub2):
            dbl_intgrl = 0.
            for ia in range(nSamp):
                for ja in range(nSamp):
                    for ib in range(nSamp):
                        for jb in range(nSamp):
                            x = seps[i,j,0]/d - rxy[ia] + rxy[ib]
                            y = seps[i,j,1]/d - rxy[ja] + rxy[jb]
                            r = numpy.sqrt(x*x + y*y) * d / sf_dx
                            r1 = int(r)
                            Dphi[ia,ja,ib,jb] = (r - float(r1))*sf[r1+1] + (float(r1+1)-r)*sf[r1]    #linear interpolation
                            ra_intgrl[ib,jb] += Dphi[ia,ja,ib,jb]
                            rb_intgrl[ia,ja] += Dphi[ia,ja,ib,jb]
                            dbl_intgrl += Dphi[ia,ja,ib,jb]
            xxtiltcov = 0.
            xytiltcov = 0.
            yytiltcov = 0.
            for ia in range(nSamp):
                for ja in range(nSamp):
                    for ib in range(nSamp):
                        for jb in range(nSamp):
                            phiphi = 0.5*(ra_intgrl[ib,jb] + rb_intgrl[ia,ja])/nSamp2
                            phiphi -= 0.5*Dphi[ia,ja,ib,jb]
                            phiphi -= 0.5*dbl_intgrl/nSamp4
                            xxtiltcov += phiphi*tilt[ia]*tilt[ib]
                            xytiltcov += phiphi*tilt[ja]*tilt[ib]
                            yytiltcov += phiphi*tilt[ja]*tilt[jb]
            cov[0,i,j] = scaling*xxtiltcov/nSamp4
            cov[1,i,j] = scaling*xytiltcov/nSamp4
            cov[2,i,j] = scaling*yytiltcov/nSamp4

    return cov/fudgeFactor



def new_structure_function_vk(separation, L0):
    dprf0 = (2*numpy.pi/L0)*separation
    k1 = 0.1716613621245709486
    if dprf0.max() > 4.71239:
        print 'Not doing it right!'

    a = 5./6
    x2a = dprf0**(2.*a)
    x22 = dprf0 * dprf0/4.

    Ga = [
            0, 12.067619015983075, 5.17183672113560444,
            0.795667187867016068,
            0.0628158306210802181, 0.00301515986981185091,
            9.72632216068338833e-05, 2.25320204494595251e-06,
            3.93000356676612095e-08, 5.34694362825451923e-10,
            5.83302941264329804e-12,
            ]
    Gma = [ -3.74878707653729304, -2.04479295083852408,
            -0.360845814853857083, -0.0313778969438136685,
            -0.001622994669507603, -5.56455315259749673e-05,
            -1.35720808599938951e-06, -2.47515152461894642e-08,
            -3.50257291219662472e-10, -3.95770950530691961e-12,
            -3.65327031259100284e-14
        ]
    x2n = 0.5

    s = Gma[0] * x2a
    s*= x2n

    # Prepare recurrence iteration for next step
    x2n *= x22

    for n in xrange(10):
        s += (Gma[n+1]*x2a + Ga[n+1]) * x2n
        # Prepare recurrent iteration for next step
        x2n *= x22
    
    s *= k1 * L0**(5./3)
    return -s




def mirror_covariance_matrix(cov_mat, n_subaps):
    """
    Mirrors a covariance matrix around the axis of the diagonal.

    Parameters:
        cov_mat (ndarray): The covariance matrix to mirror
        n_subaps (ndarray): Number of sub-aperture in each WFS
    """
    total_slopes = cov_mat.shape[0]
    n_wfs = n_subaps.shape[0]

    n1 = 0
    for n in range(n_wfs):
        m1 = 0
        for m in range(n + 1):
            if n != m:
                n2 = n1 + 2 * n_subaps[n]
                m2 = m1 + 2 * n_subaps[m]

                cov_mat[m1: m2, n1: n2] = (
                    numpy.swapaxes(cov_mat[n1: n2, m1: m2], 1, 0)
                )

                m1 += 2 * n_subaps[m]
        n1 += 2 * n_subaps[n]
    return cov_mat



if __name__ == "__main__":
    GSPOS = numpy.array(([0,-20],[0,20]))
    TEL_DIAM = 4.2
    NWFS = GSPOS.shape[0]
    NXSUBAPS = numpy.array([7]*NWFS)
    NSUBAPS = numpy.array([36]*NWFS)
    SUBAPDIAM = numpy.array([TEL_DIAM/NXSUBAPS[0]]*NWFS)
    GSALT = numpy.array([0]*NWFS)
    GSTYPE = numpy.array([1]*NWFS)
    PUPILSHIFT = numpy.array([0]*NWFS)
    PUPILMAG = numpy.array([NXSUBAPS[0]]*NWFS)
    PUPILROT = numpy.array([0]*NWFS)
    OBS = 0.285
    NCPU = 1
    PART = 0
    PUPIL_MASK = pupil.circle(NXSUBAPS[0]/2., NXSUBAPS[0])
    PUPIL_MASK[(NXSUBAPS-1)/2, (NXSUBAPS-1)/2] = 0
    PUPIL_MASK = [PUPIL_MASK]*NWFS
    waveL = 500e-9
    gam = numpy.array([waveL]*NWFS)
    combs = int(comb(GSPOS.shape[1], 2, exact=True))
    selector = numpy.array((range(GSPOS.shape[0])))
    selector = numpy.array((list(itertools.combinations(selector, 2))))

    NLAYERS = 2
    r0 = numpy.array([0.1]*NLAYERS)
    L0 = numpy.array([25.]*NLAYERS)
    LAYERHEIGHTS = numpy.array([0., 9282.])

    params = CovarianceMatrix(NWFS, PUPIL_MASK, TEL_DIAM, SUBAPDIAM, GSALT, GSPOS, gam, NLAYERS, LAYERHEIGHTS, r0, L0, True)
    mat = params.make_covariance_matrix()

    # m = covMapsFromMatrix(mat, GSPOS, NXSUBAPS[0], NSUBAPS[0], PUPIL_MASK[0], 2)
    # l = covarianceMapSlice(m, GSPOS, PUPIL_MASK[0], selector, 1)
