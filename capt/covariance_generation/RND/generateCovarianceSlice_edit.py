import numpy
import time
import math
import itertools
from astropy.io import fits
from matplotlib import pyplot; pyplot.ion()
import slopeSliceCovariance
from aotools.functions import circle
from covMapVector import covMapVector
from superFastCovMap import superFastCovMap, getMappingMatrix
import scipy.special
from scipy.misc import comb
from slopeSliceCovariance import mapSliceFromSlopes
from covMapsFromMatrix import covMapsFromMatrix

 
class CovarianceSlice(object):
     
    def __init__(self, nwfs, subapDiam, wfsWavelength, telDiam, gsAlt, gsPos, subapLoc, selector, nLayers, layerAlt, fitL0, L0,  output, styc):
         
        self.nwfs = nwfs
        self.subapDiam = subapDiam
        self.wfsWavelength = wfsWavelength
        self.telDiam = telDiam
        self.gsAlt = gsAlt
        self.gsPos = gsPos[:,::-1] # Swap X and Y GS positions to agree with legacy code
        self.subapLoc = subapLoc
        self.selector = selector
        self.L0 = L0
        self.nLayers = nLayers
        self.layerAlt = layerAlt
        self.radSqaured_to_arcsecSqaured = ((180./numpy.pi) * 3600)**2
        # print("n_layers: {}".format(n_layers))

        self.nSubaps = []
        self.total_subaps = 0
        for wfs_n in range(nwfs):
            self.nSubaps.append(subapLoc[0].shape[2])
            self.total_subaps += self.nSubaps[wfs_n]
        self.nSubaps = numpy.array(self.nSubaps, dtype="int")
        self.total_subaps = int(self.total_subaps)

        self.combs = subapLoc[0].shape[0]
        self.envelope = subapLoc[0].shape[1]
        self.sliceLength = subapLoc[0].shape[2]
        self.subap_layer_positions = numpy.zeros((nLayers, 2, self.combs, self.envelope, self.sliceLength, 2))
        self.subap_layer_diameters = numpy.zeros((nLayers, 2, self.combs, self.envelope))
        self.output = output
        self.styc = styc


        for layer_n, layer_altitude in enumerate(self.layerAlt):
            for wfs_n in range(2):
                for comb in range(self.combs):                        

                    if self.gsAlt[self.selector[comb, wfs_n]] != 0:
                        # print("Its an LGS!")
                        scale_factor = (1 - layer_altitude/self.gsAlt[self.selector[comb,wfs_n]])
                        positions = scale_factor * self.subapLoc[wfs_n][comb].copy()
                    else:
                        scale_factor = 1
                        positions = self.subapLoc[wfs_n][comb].copy()

                    #translate due to GS position
                    gs_pos_rad = numpy.array(self.gsPos[self.selector[comb,wfs_n]]) * numpy.pi/180/3600
                    # print("GS Positions: {} rad".format(gs_pos_rad))
                    translation = gs_pos_rad * layer_altitude
                    # print("Translation: {} m".format(translation))
                    # print("Max position before translation: {} m".format(abs(positions).max()))
                    positions += translation

                    
                    self.subap_layer_positions[layer_n, wfs_n, comb] = positions
                    self.subap_layer_diameters[layer_n, wfs_n, comb] = self.subapDiam * scale_factor


        self.xy_separations = self.subap_layer_positions[:,0,:,:] - self.subap_layer_positions[:,1,:,:]
        # self.xy_separations += 1e-20



        self.fitL0 = fitL0
        self.cov_xx = numpy.zeros((nLayers, self.combs, self.envelope, self.sliceLength))
        self.cov_yy = numpy.zeros((nLayers, self.combs, self.envelope, self.sliceLength))
        self.flatSF = []
        self.sfFinder = numpy.zeros((self.nLayers, self.combs, self.envelope)).astype('int')
        if self.styc == True and self.fitL0 == False:
            self.computeStyc(L0)
            self.fixedLayerParameters()

        if self.styc == False and self.fitL0 == False:
            self.computeButt(L0)
            self.fixedLayerParameters() 




    def _make_covariance_slice(self, r0, L0):
        if self.styc==True and self.fitL0==True:
            self.computeStyc(L0)
            self.fixedLayerParameters()
        if self.styc==False and self.fitL0==True:
            self.computeButt(L0)
            self.fixedLayerParameters()

        if self.output=='x' or self.output=='y' or self.output=='x+y':
            self.covariance_slice = numpy.zeros((self.combs*self.envelope, self.sliceLength))
        if self.output=='x and y':
            self.covariance_slice = numpy.zeros((self.combs*self.envelope, self.sliceLength*2))
    
        for layer_n in range(self.nLayers):
            r0_scale2 = (L0[layer_n]/r0[layer_n])**(5./3.)

            if self.output=='x+y':
                self.covariance_slice += self.covariance_slice_fixed[layer_n] * r0_scale2

            if self.output=='x':
                self.covariance_slice += self.covariance_slice_fixed[layer_n] * r0_scale2

            if self.output=='y':
                self.covariance_slice += self.covariance_slice_fixed[layer_n] * r0_scale2

            if self.output=='x and y':
                self.covariance_slice += self.covariance_slice_fixed[layer_n] * r0_scale2

        return self.covariance_slice * self.radSqaured_to_arcsecSqaured




    def fixedLayerParameters(self):
        if self.output=='x' or self.output=='y' or self.output=='x+y':
            self.covariance_slice_fixed = numpy.zeros((self.nLayers, self.combs*self.envelope, self.sliceLength))
        if self.output=='x and y':
            self.covariance_slice_fixed = numpy.zeros((self.nLayers, self.combs*self.envelope, self.sliceLength*2))

        for layer_n in range(self.nLayers):
    
            for comb in range(self.combs):

                for env in range(self.envelope):

                    cov_xx = self.cov_xx[layer_n, comb, env]
                    cov_yy = self.cov_yy[layer_n, comb, env]

                    # print("covmat coords: ({}: {}, {}: {})".format(cov_mat_coord_x1, cov_mat_coord_x2, cov_mat_coord_y1, cov_mat_coord_y2))
                    r0_scale1 = ((self.wfsWavelength[self.selector[comb,0]] * self.wfsWavelength[self.selector[comb,1]])
                            / (8 * numpy.pi**2 * self.subap_layer_diameters[layer_n, 0, comb, env] * self.subap_layer_diameters[layer_n, 0, comb, env] ))

                    if self.output=='x+y':
                        self.covariance_slice_fixed[layer_n, self.envelope*comb + env] += ((cov_xx+cov_yy)*(r0_scale1/2.))

                    if self.output=='x':
                        self.covariance_slice_fixed[layer_n, self.envelope*comb + env] += cov_xx*r0_scale1

                    if self.output=='y':
                        self.covariance_slice_fixed[layer_n, self.envelope*comb + env] += cov_yy*r0_scale1

                    if self.output=='x and y':
                        self.covariance_slice_fixed[layer_n, self.envelope*comb + env] += numpy.hstack(((cov_xx*r0_scale1), (cov_yy*r0_scale1)))



    def computeStyc(self, L0):#)(nLayers, combs, env, xy_separations, subap_layer_diameters, L0):
        for layer_n in range(self.nLayers):
            for comb in range(self.combs):
                for env in range(self.envelope):
                    if self.output!='y':
                        self.cov_xx[layer_n, comb, env] = compute_covariance_xx(self.xy_separations[layer_n, comb, env], self.subap_layer_diameters[layer_n, 0, comb, env], L0[layer_n])
                    if self.output!='x':
                        self.cov_yy[layer_n, comb, env] = compute_covariance_yy(self.xy_separations[layer_n, comb, env], self.subap_layer_diameters[layer_n, 0, comb, env], L0[layer_n])

    def computeButt(self, L0):
        for layer_n in range(self.nLayers):
            for comb in range(self.combs):
                for env in range(self.envelope):
                    wfs1_diam = self.subap_layer_diameters[layer_n, 0, comb, env]
                    wfs2_diam = self.subap_layer_diameters[layer_n, 1, comb, env]
                    n_subaps1 = self.nSubaps[self.selector[comb][0]]
                    n_subaps2 = self.nSubaps[self.selector[comb][1]]
                    xy_sep = self.xy_separations[layer_n, comb, env]

                    assert abs(wfs1_diam - wfs2_diam) < 1.e-10              # numerical integration code doesn't support different subap sizes
                    assert abs(n_subaps1 - n_subaps2) < 1.e-10              # numerical integration code doesn't support different num. subaps
                    maxDelta=0
                    for i in range(xy_sep.shape[0]):
                        if math.isnan(xy_sep[i,0]) != True:
                            if numpy.abs(xy_sep[i]).max()>maxDelta:
                                maxDelta = int(numpy.abs(xy_sep[i]).max()/wfs1_diam)+1
                    
                    sf_dx = wfs1_diam/100.
                    sf_n = int(4*maxDelta*(wfs1_diam/sf_dx))
                    sf = new_structure_function_vk(numpy.arange(sf_n)*sf_dx, L0[layer_n])
                    sf[0] = 0.    #get rid of nan
                    # self.sfFinder[layer_n, comb, env] = int(sf.shape[0])
                    # self.flatSF = numpy.hstack((self.flatSF, sf))
                    nSamp = 8    #hard-wired parameter
                    sf_dx = wfs1_diam/100.
                    self.cov_xx[layer_n, comb, env], self.cov_yy[layer_n, comb, env] = compute_ztilt_covariances(n_subaps1, xy_sep, sf, sf_dx, nSamp, wfs1_diam)



def wfs_covariance(n_subaps1, wfs1_diam, xy_separations, sf, styc):
    """
    Calculates the covariance between 2 WFSs

    Parameters:
        n_subaps1 (int): number of sub-apertures in WFS 1
        wfs1_diam: Diameter of WFS 1 sub-apertures
        stycMethod: If True use phase structure function differencing method, otherwise use numerical integration method

    Returns:
        slope covariance of X with X , slope covariance of Y with Y, slope covariance of X with Y
    """        

    if styc == True:
        # print("Min separation: {}".format(abs(xy_separations).min()))
        cov_xx = compute_covariance_xx(xy_separations, wfs1_diam)
        cov_yy = compute_covariance_yy(xy_separations, wfs1_diam)
        return cov_xx, cov_yy

    else:
        nSamp = 8    #hard-wired parameter
        sf_dx = wfs1_diam/100.
        cov_xx, cov_yy = compute_ztilt_covariances(n_subaps1, xy_separations, sf, sf_dx, nSamp, wfs1_diam)

        return cov_xx, cov_yy



def compute_ztilt_covariances(n_subaps1, xy_separations, sf, sf_dx, nSamp, wfs1_diam):

    scaling = 206265.*206265.*3.* (500.e-9/(numpy.pi*wfs1_diam))**2;   # scaling to arcsec^2
    fudgeFactor = (206265.**2) * (500.e-9**2) / ( 8. * (numpy.pi**2) * (wfs1_diam**2) )    #further scaling to get from arcsec^2 to units used elsewhere in this module

    rxy = (numpy.arange(nSamp) - float(nSamp)/2 + 0.5) / float(nSamp)
    tilt = 2.*(3.**0.5)*rxy
    
    nSamp2 = float(nSamp**2)
    nSamp4 = float(nSamp**4)

    ra_intgrl = numpy.zeros((nSamp,nSamp),numpy.float)
    rb_intgrl = numpy.zeros((nSamp,nSamp),numpy.float)
    Dphi = numpy.zeros((nSamp,nSamp,nSamp,nSamp),numpy.float)
    cov = numpy.zeros((2,n_subaps1),numpy.float)

    dbl_intgrl = 0.
    for n in range(n_subaps1):
        if math.isnan(xy_separations[n,0]) != True:
            for ia in range(nSamp):
                for ja in range(nSamp):
                    for ib in range(nSamp):
                        for jb in range(nSamp):
                            x = (xy_separations[n,1]/wfs1_diam) - rxy[ia] + rxy[ib]
                            y = (xy_separations[n,0]/wfs1_diam) - rxy[ja] + rxy[jb]
                            r = numpy.sqrt(x*x + y*y) * wfs1_diam / sf_dx
                            r1 = int(r)
                            Dphi[ia,ja,ib,jb] = (r - float(r1))*sf[r1+1] + (float(r1+1)-r)*sf[r1]
                            ra_intgrl[ib,jb] += Dphi[ia,ja,ib,jb]
                            rb_intgrl[ia,ja] += Dphi[ia,ja,ib,jb]
                            dbl_intgrl += Dphi[ia,ja,ib,jb]
            xxtiltcov = 0.
            yytiltcov = 0.
            for ia in range(nSamp):
                for ja in range(nSamp):
                    for ib in range(nSamp):
                        for jb in range(nSamp):
                            phiphi = 0.5*(ra_intgrl[ib,jb] + rb_intgrl[ia,ja])/nSamp2
                            phiphi -= 0.5*Dphi[ia,ja,ib,jb]
                            phiphi -= 0.5*dbl_intgrl/nSamp4
                            xxtiltcov += phiphi*tilt[ia]*tilt[ib]
                            yytiltcov += phiphi*tilt[ja]*tilt[jb]
            cov[0,n] = scaling*xxtiltcov/nSamp4
            cov[1,n] = scaling*yytiltcov/nSamp4

    return cov[0]/fudgeFactor, cov[1]/fudgeFactor




def compute_covariance_xx(separation, subap1_diam, L0):    
    cov_xx = numpy.zeros((separation.shape[0]),numpy.float)
    
    for n in range(cov_xx.shape[0]):
        if math.isnan(separation[n,0]) != True:

            x1 = separation[n,1] + (subap1_diam - subap1_diam) * 0.5
            r1 = numpy.array([numpy.sqrt(x1**2 + separation[n,0]**2)])

            x2 = separation[n,1] - (subap1_diam + subap1_diam) * 0.5
            r2 = numpy.array([numpy.sqrt(x2**2 + separation[n,0]**2)])

            x3 = separation[n,1] + (subap1_diam + subap1_diam) * 0.5
            r3 = numpy.array([numpy.sqrt(x3**2 + separation[n,0]**2)])

            cov_xx[n] = (-2 * new_structure_function_vk(r1, L0)
                    + new_structure_function_vk(r2, L0)
                    + new_structure_function_vk(r3, L0))

    return cov_xx




def compute_covariance_yy(separation, subap1_diam, L0):
    cov_yy = numpy.zeros((separation.shape[0]),numpy.float)
    
    for n in range(cov_yy.shape[0]):
        if math.isnan(separation[n,0]) != True:

            y1 = separation[n,0] + (subap1_diam - subap1_diam) * 0.5
            r1 = numpy.array([numpy.sqrt(separation[n,1]**2 + y1**2)])

            y2 = separation[n,0] - (subap1_diam + subap1_diam) * 0.5
            r2 = numpy.array([numpy.sqrt(separation[n,1]**2 + y2**2)])

            y3 = separation[n,0] + (subap1_diam + subap1_diam) * 0.5
            r3 = numpy.array([numpy.sqrt(separation[n,1]**2 + y3**2)])

            cov_yy[n] = (-2 * new_structure_function_vk(r1, L0)
                + new_structure_function_vk(r2, L0)
                + new_structure_function_vk(r3, L0))

    return cov_yy



def new_structure_function_vk(separation, L0):
    dprf0 = (2*numpy.pi/L0)*separation
    k1 = 0.1716613621245709486
    # print 
    res = numpy.zeros(dprf0.shape[0])
    
    for i in range(dprf0.shape[0]):
        if dprf0[i] > 4.71239:
            res[i] = asymp_macdo(dprf0[i])
        else:
            res[i] = -macdo_x56(dprf0[i])

    res *= k1
    return res


def asymp_macdo(dprf0):
    # k2 is the value for
    # gamma_R(5./6)*2^(-1./6)
    k2 = 1.00563491799858928388289314170833
    k3 = 1.25331413731550012081   #  sqrt(pi/2)
    a1 = 0.22222222222222222222   #  2/9
    a2 = -0.08641975308641974829  #  -7/89
    a3 = 0.08001828989483310284   # 175/2187

    x1 = 1./dprf0
    res = (	k2
            - k3 * numpy.exp(-dprf0) * dprf0**(1./3)
            * (1.0 + x1*(a1 + x1*(a2 + x1*a3)))
            )
    # print "oh hey"
    return res

def macdo_x56(dprf0):
    a = 5./6
    x2a = dprf0**(2.*a)
    x22 = dprf0 * dprf0/4.
    x2n = 0.5

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

    s = Gma[0] * x2a
    s*= x2n

    # Prepare recurrence iteration for next step
    x2n *= x22

    for n in range(10):
        s += (Gma[n+1]*x2a + Ga[n+1]) * x2n
        # Prepare recurrent iteration for next step
        x2n *= x22

    return s




if __name__ == '__main__':
    # """####################################################################"""
    # """########################### DEMONSTRATION ##########################"""
    # """####################################################################"""
    
    # nwfs = 3
    # nxSubaps = numpy.array([7]*nwfs)
    # pupilMasks = circle(nxSubaps[0]/2., nxSubaps[0])
    # pupilMasks[3,3]=0
    # nSubaps = numpy.int(pupilMasks.sum())
    # covMapDim = (2*nxSubaps[0])-1
    # pupilMasks = [pupilMasks]*nwfs
    # telDiam = 4.2
    # subapDiam = telDiam/nxSubaps
    # gsAlt = numpy.array([0]*nwfs)
    # gsPos = numpy.array([[0.,-20.], [0.,20], [35,0]])
    # wfsWavelength = numpy.array([500e-9]*nwfs)
    # nLayers = 1
    # layerAlt = numpy.array([9282.])
    # r0 = numpy.array([0.2]*nLayers)
    # L0 = numpy.array([30]*nLayers)
    # stycMethod = True



    # """##################### GENERATE COVARIANCE MAPS ######################"""
    # matrixStart = time.time()
    # matrixParams = slopecovariance.CovarianceMatrix(nwfs, pupilMasks, telDiam, subapDiam, gsAlt, gsPos, wfsWavelength, nLayers, 
    #         layerAlt, r0, L0, stycMethod)
    # covarianceMatrix = matrixParams.make_covariance_matrix()
    # covMaps = covMapsFromMatrix(covarianceMatrix, gsPos, nxSubaps[0], nSubaps, pupilMasks[0], 2)
    # matrixFinish = time.time()



    # """################## CALCULATE SLOPE COVARIANCE ROI ##################"""
    # nxSubaps = 7
    # pupilMask = pupil.circle(nxSubaps/2., nxSubaps)
    # pupilMask[3,3]=0
    # subapDiam = telDiam/nxSubaps
    # stycMethod = True
    # envelope = 1
    # output = 'x and y'
    # onesMat, wfsMat_1, wfsMat_2, subapLoc, allMapPos, selector, vectorMap = referenceArrays(pupilMask, gsPos, telDiam, envelope)
    # slopes = fits.getdata('fitsFiles/canary_1Layer_h9282_r0p2_L030_randScrns_gspos00n20a0020a3500.fits') * (1./3600)*(numpy.pi/180)*(4.5/16.)
    
    # slopeStart = time.time()
    # slopeSlice = mapSliceFromSlopes(slopes, allMapPos, covMapDim, nSubaps, onesMat, wfsMat_1, wfsMat_2, selector, output)
    # slopeFinish = time.time()



    # """################### GENERATE COVARIANCE MAP ROI ###################"""
    # sliceStart = time.time()
    # params = covarianceSlice(nwfs, subapDiam, wfsWavelength, telDiam, gsAlt, gsPos, subapLoc, selector, nLayers, layerAlt, r0, L0, output, False)
    # mapSlice = params.getCovarianceSlice()
    # sliceFinish = time.time()
    
    # print '\n', 'Time to Generate Matrix:', matrixFinish-matrixStart, '\n'
    # pyplot.figure('Generated Matrix')
    # pyplot.imshow(covMaps)
    # pyplot.figure('Map Vector')
    # pyplot.imshow(vectorMap)
    
    # print '\n', 'Time to Calculate Slope Slice:', slopeFinish-slopeStart, '\n'
    # pyplot.figure('Map Slice from Slopes')
    # pyplot.imshow(slopeSlice)
    
    # print '\n', 'Time to Generate Slice:', sliceFinish-sliceStart, '\n'
    # pyplot.figure('Generated Map Slice')
    # pyplot.imshow(mapSlice)

    nwfs = 3
    nxSubaps = 7
    pupilMask = circle(nxSubaps/2., nxSubaps)
    pupilMask[3,3]=0
    nSubaps = numpy.int(pupilMask.sum())
    covMapDim = (2*nxSubaps)-1
    telDiam = 4.2
    gsAlt = numpy.array([0]*nwfs)
    gsPos = numpy.array([[0.,-20.], [0.,20]])#, [50,50], [-5, -5]])
    wfsWavelength = numpy.array([500e-9]*nwfs)
    
    nLayers = 2
    layerAlt = numpy.array([0., 9281.91628112])
    r0 = numpy.array([0.1, 0.1])
    L0 = numpy.array([25]*nLayers)

    subapDiam = telDiam/nxSubaps
    stycMethod = True
    envelope = 0
    belowGround = 0
    styc = True
    output = 'x and y'
    onesMat, wfsMat_1, wfsMat_2, allMapPos, selector, xy_separations = referenceArrays(pupilMask, gsPos, telDiam, belowGround, envelope)

    fitRot = False
    fitOffset = False
    fitL0 = False
    rot = numpy.array((0,45))
    offset = 0

    params = CovarianceSlice(nwfs, subapDiam, wfsWavelength, telDiam, gsAlt, gsPos, subapLoc, selector, nLayers, layerAlt, fitL0, L0, output, styc)
    se = params._make_covariance_slice(r0, L0)

    pyplot.figure()
    pyplot.imshow(se)