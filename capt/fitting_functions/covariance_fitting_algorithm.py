import os
import sys
import numpy
import itertools
from scipy.special import comb
from scipy.optimize import minimize, root
from matplotlib import pyplot; pyplot.ion()
from capt.map_functions.covMap_fromMatrix import covMap_fromMatrix
from capt.roi_functions.roi_referenceArrays import roi_referenceArrays
from capt.covariance_generation.generate_covariance_roi import covariance_roi
from capt.covariance_generation.generate_covariance_matrix import covariance_matrix
from capt.covariance_generation.generate_covariance_roi_l3s import covariance_roi_l3s





class fitting_parameters(object):
    def __init__(self, tp, method, target_array, n_layer, layer_alt, tt_track_present, lgs_track_present, offset_present, 
            fit_unsensed, fit_tt_track, fit_lgs_track, fit_offset, fit_L0, L0, r0, roi_belowGround, roi_envelope, zeroSep_locations, 
            allMapPos, xy_separations):

        self.method = method
        self.zeroSep_locations = zeroSep_locations
        self.remove_tt = tp.remove_tt
        self.target_array = target_array
        self.zeroSep_cov = tp.zeroSep_cov
        self.print_fitting = tp.print_fitting
        self.gs_combs = tp.combs
        self.theo_cov = False
        self.output_fitted_array = tp.output_fitted_array
        self.count = 0
        self.fit_unsensed = fit_unsensed

        if lgs_track_present==True or fit_lgs_track==True:
            self.using_lgs = True
        else:
            self.using_lgs=False

        if self.fit_unsensed == True:
            n_layer += 1
            layer_alt = numpy.append(layer_alt, 1e20)
            r0 = numpy.append(r0, r0[0])
            L0 = numpy.append(L0, L0[0])

        if self.target_array=='Covariance Map ROI':
            if self.method=='Direct Fit':
                self.generationParams = covariance_roi(tp.pupil_mask, tp.subap_diam, 
                    tp.wavelength, tp.tel_diam, tp.n_subap, tp.gs_dist, tp.gs_pos, n_layer, 
                    layer_alt, L0, allMapPos, xy_separations, tp.map_axis, styc_method=tp.styc_method, 
                    wind_profiling=False, tt_track_present=tt_track_present, lgs_track_present=lgs_track_present,
                    offset_present=offset_present, fit_layer_alt=False, fit_tt_track=fit_tt_track, 
                    fit_lgs_track=fit_lgs_track, fit_offset=fit_offset, fit_L0=fit_L0)
            
            if self.method=='L3S Fit':
                onesMat, wfsMat_1, wfsMat_2, allMapPos_acrossMap, selector, xy_separations_acrossMap = roi_referenceArrays(
                    tp.pupil_mask, tp.gs_pos, tp.tel_diam, tp.pupil_mask.shape[0]-1, tp.roi_envelope)
                
                self.generationParams = covariance_roi_l3s(tp.pupil_mask, tp.subap_diam, tp.wavelength, 
                    tp.tel_diam, tp.n_subap, tp.gs_dist, tp.gs_pos, n_layer, layer_alt, L0, allMapPos_acrossMap, 
                    xy_separations_acrossMap, tp.map_axis, tp.roi_belowGround, tp.roi_envelope, 
                    styc_method=tp.styc_method, wind_profiling=False, lgs_track_present=lgs_track_present, 
                    offset_present=offset_present, fit_layer_alt=False, fit_lgs_track=fit_lgs_track, 
                    fit_offset=fit_offset, fit_L0=fit_L0)

        else:
            rem_tt = False
            l3s1_trans = False
            self.gs_combs += tp.n_wfs
            if self.method=='L3S Fit':
                l3s1_trans = True
            if tp.remove_tt==True:
                rem_tt = True

            self.generationParams = covariance_matrix(tp.pupil_mask, tp.subap_diam, tp.wavelength, 
                tp.tel_diam, tp.n_subap, tp.gs_dist, tp.gs_pos, n_layer, layer_alt, r0, L0, 
                styc_method=tp.styc_method, wind_profiling=False, tt_track_present=tt_track_present, 
                lgs_track_present=lgs_track_present, offset_present=offset_present, fit_layer_alt=False, 
                fit_tt_track=fit_tt_track, fit_lgs_track=fit_lgs_track, fit_offset=fit_offset, 
                fit_L0=fit_L0, matrix_xy=tp.matrix_xy, huge_matrix=tp.huge_matrix, 
                l3s1_transform=l3s1_trans, l3s1_matrix=tp.l3s1_matrix, remove_tt=rem_tt, 
                remove_tt_matrix=tp.remove_tt_matrix)

            if self.target_array=='Covariance Map':
                self.n_wfs = tp.n_wfs
                self.nx_subap = tp.nx_subap
                self.n_subap = tp.n_subap
                self.n_subap_from_pupilMask = tp.n_subap_from_pupilMask
                self.pupil_mask = tp.pupil_mask
                self.map_axis = tp.map_axis
                self.mm = tp.mm
                self.mmc = tp.mmc
                self.md = tp.md







    def covariance_fit(self, tp, cov_meas, layer_alt, r0, L0, tt_track, lgs_track, shwfs_shift, shwfs_rot, 
        fit_layer_alt=False, fit_r0=False, fit_tt_track=False, fit_lgs_track=False, fit_L0=False, 
        fit_groundL0=False, fit_globalL0=False, fit_shift=False, fit_rot=False, callback=None):

        if self.fit_unsensed == True:
            layer_alt = numpy.append(layer_alt, 1e20)
            r0 = numpy.append(r0, r0[0])
            L0 = numpy.append(L0, L0[0])

        self.fittingGroundL0 = fit_groundL0
        self.fittingGlobalL0 = fit_globalL0
        fit_L0s = numpy.array([fit_L0, fit_groundL0, fit_globalL0])

        self.fit_tt_track = fit_tt_track

        if len(numpy.where(fit_L0s==True)[0])>1:
            raise Exception('Choose where to fit L0. Come on son.')

        try:
            len(fit_layer_alt)
        except TypeError:
            fit_layer_alt = numpy.array([fit_layer_alt]*len(layer_alt))

        try:
            len(fit_r0)
        except TypeError:
            fit_r0 = numpy.array([fit_r0]*len(r0))

        try:
            len(fit_L0)
        except TypeError:
            fit_L0 = numpy.array([fit_L0]*len(L0))

        try:
            len(fit_groundL0)
        except TypeError:
            fit_groundL0 = numpy.array([fit_groundL0]*1)

        try:
            len(fit_globalL0)
        except TypeError:
            fit_globalL0 = numpy.array([fit_globalL0]*1)

        try:
            len(fit_tt_track)
        except TypeError:
            fit_tt_track = numpy.array([fit_tt_track]*tt_track.shape[0])

        try:
            len(fit_lgs_track)
        except TypeError:
            fit_lgs_track = numpy.array([fit_lgs_track]*(self.gs_combs*2)).reshape(self.gs_combs, 2)

        try:
            len(fit_shift)
        except TypeError:
            fit_shift = numpy.array([fit_shift]*(len(shwfs_shift)*2)).reshape((len(shwfs_shift), 2))

        try:
            len(fit_rot)
        except TypeError:
            fit_rot = numpy.array([fit_rot]*(len(shwfs_rot)))


        layer_alt = layer_alt.copy().astype("object")
        r0 = r0.copy().astype("object")
        L0 = L0.copy().astype("object")
        tt_track = tt_track.copy().astype("object")
        lgs_track = lgs_track.copy().astype("object")
        groundL0 = numpy.array((L0[:1])).copy().astype("object")
        globalL0 = numpy.array((L0[:1])).copy().astype("object")
        shwfs_shift = shwfs_shift.copy().astype("object")
        shwfs_rot = shwfs_rot.copy().astype("object")

        #Guess array to store fitted variables
        guess = numpy.array([])

        for i, fit in enumerate(fit_layer_alt):
            if fit:
                guess = numpy.append(guess, layer_alt[i])
                layer_alt[i] = None

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

        for i, fit in enumerate(fit_tt_track):
            if fit:
                guess = numpy.append(guess, tt_track[i])
                tt_track[i] = None


        lgs_track = lgs_track.reshape(self.gs_combs*2)
        fit_lgs_track = fit_lgs_track.reshape(self.gs_combs*2)
        for i, fit in enumerate(fit_lgs_track):
            if fit:
                guess = numpy.append(guess, lgs_track[i])
                lgs_track[i] = None
        lgs_track = lgs_track.reshape(self.gs_combs, 2)
        fit_lgs_track = fit_lgs_track.reshape(self.gs_combs, 2)


        shwfs_shift = shwfs_shift.reshape(2*len(shwfs_rot))
        fit_shift = fit_shift.reshape(2*len(shwfs_rot))
        for i, fit in enumerate(fit_shift):
            if fit:
                guess = numpy.append(guess, shwfs_shift[i])
                shwfs_shift[i] = None
        shwfs_shift = shwfs_shift.reshape((len(shwfs_rot), 2))
        fit_shift = fit_shift.reshape((len(shwfs_rot), 2))


        for i, fit in enumerate(fit_rot):
            if fit==True:
                guess = numpy.append(guess, shwfs_rot[i])
                shwfs_rot[i] = None

        if len(guess)==0:
            raise Exception("Fit something motherf%$*er!")

        self.staticArgs = (cov_meas, layer_alt, r0, tt_track, lgs_track, L0, groundL0, globalL0, shwfs_shift, shwfs_rot, callback)
        theo_results = root(self.cov_opt, guess, self.staticArgs, method="lm", tol=0.)
        # stopthis
        return theo_results['status'], self.r0, self.L0, self.tt_track, self.lgs_track, self.shwfs_rot, self.shwfs_shift, self.theo_cov, self.count, self.generationParams.timingStart







    def cov_opt(self, covSliceParams, cov_meas, layer_alt, r0, tt_track, 
            lgs_track, L0, groundL0, globalL0, shwfs_shift, shwfs_rot, callback):

        theo_cov = self.cov_fit_fromParams(covSliceParams, layer_alt, r0, 
            tt_track, lgs_track, L0, groundL0, globalL0, shwfs_shift, shwfs_rot)
        
        residual = (theo_cov - cov_meas)**2

        if self.print_fitting==True:
            print("*** RMS: {} ***".format(numpy.sqrt((residual).mean())), '\n')
        if callback:
            callback(self.theoCovSlice)

        if self.output_fitted_array==True:
            self.theo_cov = theo_cov
        
        return numpy.sqrt(residual).flatten()









    def cov_fit_fromParams(self, covSliceParams, layer_alt, r0, tt_track, lgs_track, L0, 
            groundL0, globalL0, shwfs_shift, shwfs_rot):

        #Make parameters guess values
        np = 0

        layer_alt = layer_alt.copy()
        for i, val in enumerate(layer_alt):
            if val==None:
                layer_alt[i] = numpy.abs(covSliceParams[np])
                np+=1

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
            L0 = numpy.array([globalL0]*len(r0)).flatten()

        tt_track = tt_track.copy()
        for i, val in enumerate(tt_track):
            if val==None:
                tt_track[i] = numpy.abs(covSliceParams[np])
                np+=1

        lgs_track = lgs_track.copy().reshape(2*self.gs_combs)
        for i, val in enumerate(lgs_track):
            if val==None:
                lgs_track[i] = covSliceParams[np]
                np+=1
        lgs_track.resize(self.gs_combs, 2)

        shwfs_shift = shwfs_shift.copy().reshape((2*len(shwfs_rot)))
        for i, val in enumerate(shwfs_shift):
            if val==None:
                shwfs_shift[i] = covSliceParams[np]
                np+=1
        shwfs_shift.resize((len(shwfs_rot), 2))

        shwfs_rot = shwfs_rot.copy()
        for i, val in enumerate(shwfs_rot):
            if val==None:
                shwfs_rot[i] = covSliceParams[np]
                np+=1

        if self.target_array=='Covariance Map ROI':
            
            if self.method=='Direct Fit':
                covariance = self.generationParams._make_covariance_roi_(layer_alt.astype('float'), 
                    r0.astype('float'), L0.astype('float'), tt_track=tt_track.astype('float'), 
                    lgs_track=lgs_track.astype('float'), shwfs_shift=shwfs_shift.astype('float'), 
                    shwfs_rot=shwfs_rot.astype('float'))
            
            if self.method=='L3S Fit':
                covariance = self.generationParams._make_covariance_roi_l3s_(layer_alt.astype('float'), 
                    r0.astype('float'), L0.astype('float'), lgs_track=lgs_track.astype('float'), 
                    shwfs_shift=shwfs_shift.astype('float'), shwfs_rot=shwfs_rot.astype('float'))

            if self.zeroSep_cov==False:
                covariance[self.zeroSep_locations] = 0.
        
        else:
            covariance = self.generationParams._make_covariance_matrix_(layer_alt.astype('float'), 
                r0.astype('float'), L0.astype('float'), tt_track=tt_track, lgs_track=lgs_track.astype('float'), 
                shwfs_shift=shwfs_shift.astype('float'), shwfs_rot=shwfs_rot.astype('float'))  
            
            if self.zeroSep_cov==False:
                covariance[self.zeroSep_locations] = 0.

            if self.target_array=='Covariance Map':
                covariance = covMap_fromMatrix(covariance, self.n_wfs, self.nx_subap, self.n_subap_from_pupilMask, 
                    self.pupil_mask, self.map_axis, self.mm, self.mmc, self.md)

                
        self.count+=1


        if self.print_fitting==True:
            print("Iteration:", self.count)
            print("Target Array:", self.target_array)
            print("Method:", self.method)
            print("Layer Altitudes:", layer_alt)
            print("L0:", L0)
            print("TT Track:", tt_track)
            if self.using_lgs==True:
                print("LGS Track:", lgs_track)
            print("r0:", r0)
            print("Shift: {}".format(shwfs_shift))
            print("Rotation: {}".format(shwfs_rot))

        self.r0 = r0.astype('float')
        self.L0 = L0.astype('float')
        self.tt_track = tt_track.astype('float')
        self.lgs_track = lgs_track.astype('float')
        self.shwfs_rot = shwfs_rot.astype('float')
        self.shwfs_shift = shwfs_shift.astype('float')


        # f, a = pyplot.subplots()
        # a.set_title('$x_{1}x_{2} + y_{1}y_{2}$')
        # a.set_xlabel('Sub-aperture Separation, $y$')
        # a.set_yticks([])
        # a.set_xticks(numpy.arange(0,9))
        # a.set_xticklabels(numpy.arange(-1,8))
        # im = a.imshow(covariance, vmin=0.002249661313233884, vmax=0.014974801906957418)
        # pyplot.savefig('pngs_forVideos/turb_fit/turb'+str(self.count)+'.png', dpi=800)

        return covariance 
