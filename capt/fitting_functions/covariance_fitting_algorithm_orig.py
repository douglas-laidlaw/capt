import numpy
import itertools
import sys
import os
from scipy.special import comb
from scipy.optimize import minimize, root
from capt.map_functions.covMap_fromMatrix import covMap_fromMatrix
from capt.roi_functions.roi_trackMatrix import roi_trackMatrix
from capt.covariance_generation.generate_covariance_matrix import covariance_matrix
from capt.covariance_generation.generate_covariance_roi import covariance_roi
from capt.covariance_generation.generate_covariance_roi_l3s import covariance_roi_l3s
from capt.roi_functions.roi_referenceArrays import roi_referenceArrays
from matplotlib import pyplot; pyplot.ion()






class fitting_parameters(object):
    def __init__(self, tp, method, target_array, n_wfs, pupil_mask, subap_diam, wavelength, 
        tel_diam, nx_subap, n_subap, gs_alt, gs_pos, allMapPos, selector, xy_separations, n_layer, 
        layer_alt, track_present, offset_present, fit_track, fit_offset, fit_L0, L0, r0, 
        roi_belowGround, map_axis, roi_envelope, zeroSep_cov, zeroSep_locations, mm, mmc, md, 
        transform_matrix, matrix_xy, huge_matrix, styc_method, print_fitting=True):

        # d=no

        self.method = method
        self.target_array = target_array
        self.n_wfs = n_wfs
        self.pupil_mask = pupil_mask
        self.nx_subap = nx_subap
        self.n_subap = n_subap
        self.combs = int(comb(self.n_wfs, 2, exact=True))
        self.roi_width = int((2*roi_envelope) + 1)
        self.roi_length = int(pupil_mask.shape[0] + roi_belowGround)
        self.roi_envelope = roi_envelope
        self.roi_belowGround = roi_belowGround
        self.n_layer = n_layer
        self.layer_alt = layer_alt
        self.zeroSep_cov = zeroSep_cov
        self.zeroSep_locations = zeroSep_locations
        self.map_axis = map_axis
        self.mm = mm
        self.mmc = mmc
        self.md = md
        self.transform_matrix = transform_matrix
        self.print_fitting = print_fitting
        self.count = 0
        if self.target_array=='Covariance Map ROI':
            if self.method=='Direct Fit':
                self.generationParams = covariance_roi(tp.pupil_mask, tp.subap_diam, 
                    tp.wavelength, tp.tel_diam, tp.n_subap, tp.gs_alt, tp.gs_pos, tp.allMapPos, 
                    tp.xy_separations, n_layer, layer_alt, False, tp.track_present, 
                    tp.offset_present, fit_track, fit_offset, fit_L0, L0, tp.map_axis, tp.styc_method)
            if self.method=='L3S Fit' or self.method=='2SL Fit':
                onesMat, wfsMat_1, wfsMat_2, allMapPos_acrossMap, selector, xy_separations_acrossMap = roi_referenceArrays(pupil_mask, gs_pos, tel_diam, pupil_mask.shape[0]-1, roi_envelope)
                self.generationParams = covariance_roi_l3s(pupil_mask, subap_diam, wavelength, tel_diam, n_subap, gs_alt, gs_pos, allMapPos_acrossMap, xy_separations_acrossMap, 
                    self.n_layer, self.layer_alt, False, offset_present, self.roi_belowGround, self.roi_envelope, fit_offset, fit_L0, L0, map_axis, styc_method)
        else:
            self.generationParams = covariance_matrix(pupil_mask, subap_diam, wavelength, tel_diam, 
                n_subap, gs_alt, gs_pos, n_layer, layer_alt, False, track_present, 
                offset_present, fit_track, fit_offset, fit_L0, r0, L0, styc_method, matrix_xy=matrix_xy, 
                huge_matrix=huge_matrix)



    def covariance_fit(self,cov_meas,r0,L0,track,shwfs_shift,shwfs_rot,fit_r0=False,fit_track=False,fit_L0=False,fit_groundL0=False,
        fit_globalL0=False,fit_shift=False, fit_rot=False, callback=None):
        
        # print('\n')

        self.fittingGroundL0 = fit_groundL0
        self.fittingGlobalL0 = fit_globalL0
        fit_L0s = numpy.array([fit_L0, fit_groundL0, fit_globalL0])

        self.fit_track = fit_track

        if len(numpy.where(fit_L0s==True)[0])>1:
            raise Exception('Choose where to fit L0. Come on san.')

        try:
            len(fit_r0)
        except TypeError:
            fit_r0 = numpy.array([fit_r0]*self.n_layer)

        try:
            len(fit_L0)
        except TypeError:
            fit_L0 = numpy.array([fit_L0]*self.n_layer)

        try:
            len(fit_groundL0)
        except TypeError:
            fit_groundL0 = numpy.array([fit_groundL0]*1)

        try:
            len(fit_globalL0)
        except TypeError:
            fit_globalL0 = numpy.array([fit_globalL0]*1)

        try:
            len(fit_track)
        except TypeError:
            fit_track = numpy.array([fit_track]*track.shape[0])

        try:
            len(fit_shift)
        except TypeError:
            fit_shift = numpy.array([fit_shift]*(self.n_wfs*2)).reshape((self.n_wfs, 2))

        try:
            len(fit_rot)
        except TypeError:
            fit_rot = numpy.array([fit_rot]*(self.n_wfs))


        r0 = r0.copy().astype("object")
        track = track.copy().astype("object")
        groundL0 = numpy.array((L0[:1])).copy().astype("object")
        globalL0 = numpy.array((L0[:1])).copy().astype("object")
        L0 = L0.copy().astype("object")
        shwfs_shift = shwfs_shift.copy().astype("object")
        shwfs_rot = shwfs_rot.copy().astype("object")

        #Guess array to store fitted variables
        guess = numpy.array([])

        for i, fit in enumerate(fit_r0):
            if fit:
                guess = numpy.append(guess, r0[i])
                r0[i] = None

        for i, fit in enumerate(fit_L0):
            if fit:
                guess = numpy.append(guess, L0[i])
                L0[i] = None

        for i, fit in enumerate(fit_groundL0):
            if fit:
                guess = numpy.append(guess, groundL0[i])
                groundL0[i] = None

        for i, fit in enumerate(fit_globalL0):
            if fit:
                guess = numpy.append(guess, globalL0[i])
                globalL0[i] = None

        for i, fit in enumerate(fit_track):
            if fit:
                guess = numpy.append(guess, track[i])
                track[i] = None

        shwfs_shift = shwfs_shift.reshape(2*self.n_wfs)
        fit_shift = fit_shift.reshape(2*self.n_wfs)
        for i, fit in enumerate(fit_shift):
            if fit:
                guess = numpy.append(guess, shwfs_shift[i])
                shwfs_shift[i] = None
        shwfs_shift = shwfs_shift.reshape((self.n_wfs, 2))
        fit_shift = fit_shift.reshape((self.n_wfs, 2))

        for i, fit in enumerate(fit_rot):
            if fit==True:
                guess = numpy.append(guess, shwfs_rot[i])
                shwfs_rot[i] = None


        # for i, fit in enumerate(fit_rot[1:]):
        #     if fit==True:
        #         guess = numpy.append(guess, shwfs_rot[i])
        #         shwfs_rot[i+1] = None


        if len(guess)==0:
            raise Exception("Give me something to fit!")

        self.staticArgs = (cov_meas, r0, track, L0, groundL0, globalL0, shwfs_shift, shwfs_rot, callback)
        optResult = root(self.cov_opt, guess, self.staticArgs, method="lm", tol=0.)
        # print('\n'+"Fitting Successful:", optResult['status']==1)
        return optResult, self.r0, self.L0, self.track, self.shwfs_rot, self.shwfs_shift, self.theoCovSlice, self.count, self.generationParams.timingStart







    def cov_opt(self, covSliceParams, cov_meas, r0, track, L0, groundL0, globalL0, shwfs_shift, shwfs_rot, callback):

        self.theoCovSlice = self.cov_fit_fromParams(covSliceParams, r0, track, L0, groundL0, globalL0, shwfs_shift, shwfs_rot)
        residual = (self.theoCovSlice - cov_meas)**2

        if self.print_fitting==True:
            print("*** RMS: {} ***".format(numpy.sqrt((residual).mean())), '\n')
        if callback:
            callback(self.theoCovSlice)
        # pyplot.figure()
        # pyplot.imshow(self.theoCovSlice)
        # stop=pls

        return numpy.sqrt(residual).flatten()









    def cov_fit_fromParams(self, covSliceParams, r0, track, L0, groundL0, globalL0, shwfs_shift, shwfs_rot):

        #Make parameters guess values
        np = 0

        r0 = r0.copy()
        for i, val in enumerate(r0):
            if val==None:
                r0[i] = numpy.abs(covSliceParams[np])
                np+=1

        L0 = L0.copy()
        for i, val in enumerate(L0):
            if val==None:
                L0[i] = numpy.abs(covSliceParams[np])
                np+=1

        groundL0 = groundL0.copy()
        for i, val in enumerate(groundL0):
            if val==None:
                groundL0[i] = numpy.abs(covSliceParams[np])
                np+=1

        globalL0 = globalL0.copy()
        for i, val in enumerate(globalL0):
            if val==None:
                globalL0[i] = numpy.abs(covSliceParams[np])
                np+=1

        if self.fittingGroundL0==True:
            L0[0] = numpy.array([groundL0]).flatten()[0]

        if self.fittingGlobalL0==True:
            L0 = numpy.array([globalL0]*self.n_layer).flatten()

        track = track.copy()
        for i, val in enumerate(track):
            if val==None:
                track[i] = numpy.abs(covSliceParams[np])
                np+=1

        shwfs_shift = shwfs_shift.copy().reshape((2*self.n_wfs))
        for i, val in enumerate(shwfs_shift):
            if val==None:
                shwfs_shift[i] = covSliceParams[np]
                np+=1
        shwfs_shift.resize((self.n_wfs, 2))

        shwfs_rot = shwfs_rot.copy()
        for i, val in enumerate(shwfs_rot):
            if val==None:
                shwfs_rot[i] = covSliceParams[np]
                np+=1

        # shwfs_rot = shwfs_rot.copy()
        # for i, val in enumerate(shwfs_rot[1:]):
        #     if val==None:
        #         shwfs_rot[i+1] = covSliceParams[np]
        #         np+=1


        # print(shwfs_rot)
        if self.target_array=='Covariance Map ROI':
            
            if self.method=='Direct Fit':
                covariance = self.generationParams._make_covariance_roi_(r0.astype('float'), L0.astype('float'), 
                    track=track.astype('float'), shwfs_shift=shwfs_shift.astype('float'), shwfs_rot=shwfs_rot.astype('float'))
            
            if self.method=='L3S Fit' or self.method=='2SL Fit':
                covariance = self.generationParams._make_covariance_roi_l3s_(r0.astype('float'), L0.astype('float'), shwfs_shift.astype('float'), 
                        shwfs_rot.astype('float'))

            if self.zeroSep_cov==False:
                covariance[self.zeroSep_locations] = 0.
        
        else:
            covariance = self.generationParams._make_covariance_matrix_(r0.astype('float'), L0.astype('float'), 
                track=track, shwfs_shift=shwfs_shift.astype('float'), shwfs_rot=shwfs_rot.astype('float')) 
            
            if self.method=='L3S Fit' or self.method=='2SL Fit':
                covariance = numpy.matmul(numpy.matmul(self.transform_matrix, covariance), self.transform_matrix.T)    
            
            if self.zeroSep_cov==False:
                covariance[self.zeroSep_locations] = 0.

            if self.target_array=='Covariance Map':
                covariance = covMap_fromMatrix(covariance, self.n_wfs, self.nx_subap, self.n_subap, 
                    self.pupil_mask, self.map_axis, self.mm, self.mmc, self.md)
                
        self.count+=1


        # print("\n", "Running: Covariance Slice", self.method, 'with roi_envelope = ',self.roi_envelope,'; roi_belowGround = ',self.roi_belowGround)
        if self.print_fitting==True:
            print("Iteration:", self.count)
            print("Target Array:", self.target_array)
            print("Method:", self.method)
            print("Layer Altitudes:", self.layer_alt)
            print("L0:", L0)
            print("Track:", track)
            print("r0:", r0)
            print("Shift: {}".format(shwfs_shift))
            print("Rotation: {}".format(shwfs_rot))

        self.r0 = r0.astype('float')
        self.L0 = L0.astype('float')
        self.track = track.astype('float')
        self.shwfs_rot = shwfs_rot.astype('float')
        self.shwfs_shift = shwfs_shift.astype('float')

        return covariance 
