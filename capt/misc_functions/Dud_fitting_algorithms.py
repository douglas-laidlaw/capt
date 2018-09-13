


"""A collection of my attempts to improve on L3S and direct fitting...I tried :(. Fitting offsets is hard! 
Even L3S can't achieve this. If there is an offset present, the turbulence profile is measured incorrectly; 
if the turbulence profile is measured incorrectly, the offsets can't be measured."""



    if self._2sl_fit==True:
        self.fit_method = '2SL Fit'
        self._2sl_fit_turbulence_(self.cov_meas, cov_aloft_meas, zeroSep_mult)
    if self.ml3s_fit==True:
        self.fit_method = 'L3S Fit'
        self._ml3s_fit_turbulence_(self.cov_meas, cov_aloft_meas, zeroSep_mult)





    #ml3s fitting specifications
    if config_file.configDict["FITTING ALGORITHM"]["type"]=='ml3s':
        self.transform_matrix = transform_matrix(self.n_subap, self.n_wfs)
        self.direct_fit = False
        self._2sl_fit = False
        self.l3s_fit = False
        self.ml3s_fit = True
        fit_conf = fit_conf["ml3s"]
        self.fit_1_r0 = fit_conf["fit_1_r0"]
        self.fit_1_L0 = fit_conf["fit_1_L0"]
        self.fit_1_globalL0 = fit_conf["fit_1_globalL0"]
        self.offset_present=True
        self.fit_1_offset=False
        self.fitting_1_L0 = False
        if self.fit_1_L0==True or self.fit_1_globalL0==True:
            self.fitting_1_L0 = True

        self.fit_2 = fit_conf["fit_2"]
        self.num_reps = fit_conf["num_reps"]
        self.fit_2_rot = fit_conf["fit_2_rot"]
        self.fit_2_shift = fit_conf["fit_2_shift"]
        self.fit_2_offset = True
        if self.fit_2==False:
            self.num_reps=1

        self.fit_3 = fit_conf["fit_3"]
        self.fit_3_r0 = fit_conf["fit_3_r0"]
        self.fit_3_L0 = fit_conf["fit_3_L0"]
        self.fit_3_globalL0 = fit_conf["fit_3_globalL0"]
        self.fit_3_rot = fit_conf["fit_3_rot"]
        self.fit_3_shift = fit_conf["fit_3_shift"]
        self.fit_3_track = fit_conf["fit_3_track"]
        self.fitting_3_L0 = False
        self.fit_3_offset = False
        self.fit_3_L0 = False
        if self.fit_3_rot==True or self.fit_3_shift==True:
            self.fit_3_offset=True
        if self.fit_3_L0==True or self.fit_3_globalL0==True:
            self.fitting_3_L0 = True




    def _ml3s_fit_turbulence_(self, cov_meas, cov_aloft_meas, zeroSep_mult):
        """L3S fitting method for optical turbulence profiling
        
        Parameters:
            cov_meas (ndarray): covariance target array for direct fitting.
            cov_aloft_meas (ndarray): covariance target array for L3S fitting.
            zeroSep_mult (ndarray/float): array to null covariance at sub-aperture separations of zero (if zeroSep_cov==False)."""
        
        # pyplot.figure()
        # pyplot.imshow(cov_aloft_meas)
        ######## L3S.1 ########
        # print('\n')
        # self.L0 = numpy.array([numpy.nan]*self.n_layer)
        # layer_alt_aloft = numpy.array(self.layer_alt[1:self.observable_bins])
        # n_layer_aloft = layer_alt_aloft.shape[0]
        # guess_r0_aloft = numpy.array(self.guess_r0[1:self.observable_bins])
        # guess_L0_aloft = numpy.array(self.guess_L0[1:self.observable_bins])
        # track_aloft = numpy.array([0., 0.])
        self.L0 = numpy.array([numpy.nan]*self.n_layer)
        layer_alt_aloft = numpy.array(self.layer_alt[:self.observable_bins])
        n_layer_aloft = layer_alt_aloft.shape[0]
        guess_r0_aloft = numpy.array(self.guess_r0[:self.observable_bins])
        guess_L0_aloft = numpy.array(self.guess_L0[:self.observable_bins])
        track_aloft = numpy.array([0., 0.])

        start_ml3s1 = time.time()
        totes_its_ml3s1 = 0
        totes_its_ml3s2 = 0

        for i in range(self.num_reps):
            
            params = fitting_parameters(self.fit_method, self.target_array, self.n_wfs, self.pupil_mask, 
                self.subap_diam, self.wavelength, self.tel_diam, self.nx_subap, self.n_subap, self.gs_alt, 
                self.gs_pos, self.allMapPos, self.selector, self.xy_separations, n_layer_aloft, 
                layer_alt_aloft*self.air_mass, False, self.offset_present, False, False, 
                self.fitting_1_L0, guess_L0_aloft, guess_r0_aloft, self.roi_belowGround, self.map_axis, 
                self.roi_envelope, zeroSep_mult, self.mm, self.mmc, self.md, self.transform_matrix, self.matrix_xy, 
                self.huge_matrix, self.styc_method, print_fitting=self.print_fitting)
            print('\n'+'###########################################################','\n')
            print('##################### FITTING mL3S.1 ######################','\n')
            theo_results, guess_r0_aloft, guess_L0_aloft, track_aloft, self.shwfs_rot, self.shwfs_shift, cov_l3s_fit, its_ml3s1, t = params.covariance_fit(cov_aloft_meas, 
                guess_r0_aloft, guess_L0_aloft, track_aloft, self.shwfs_shift, self.shwfs_rot, 
                fit_r0=self.fit_1_r0, fit_L0=self.fit_1_L0, fit_track=False, fit_shift=False, 
                fit_rot=False, fit_globalL0=self.fit_1_globalL0)

            its_ml3s2 = 0
            if self.fit_2==True:
                params = fitting_parameters(self.fit_method, self.target_array, self.n_wfs, self.pupil_mask, 
                self.subap_diam, self.wavelength, self.tel_diam, self.nx_subap, self.n_subap, self.gs_alt, 
                self.gs_pos, self.allMapPos, self.selector, self.xy_separations, n_layer_aloft, 
                layer_alt_aloft*self.air_mass, False, self.offset_present, False, self.fit_2_offset, 
                False, guess_L0_aloft, guess_r0_aloft, self.roi_belowGround, self.map_axis, 
                self.roi_envelope, zeroSep_mult, self.mm, self.mmc, self.md, self.transform_matrix, self.matrix_xy, 
                self.huge_matrix, self.styc_method, print_fitting=self.print_fitting)
                print('\n'+'###########################################################','\n')
                print('###################### FITTING mL3S.2 ######################','\n')
                theo_results, guess_r0_aloft, guess_L0_aloft, track_aloft, self.shwfs_rot, self.shwfs_shift, cov_l3s_fit, its_ml3s2, t = params.covariance_fit(cov_aloft_meas, 
                    guess_r0_aloft, guess_L0_aloft, track_aloft, self.shwfs_shift, self.shwfs_rot, 
                    fit_r0=False, fit_L0=False, fit_track=False, fit_shift=self.fit_2_shift, 
                    fit_rot=self.fit_2_rot, fit_globalL0=False)

            totes_its_ml3s1 += its_ml3s1
            totes_its_ml3s2 += its_ml3s2

        finish_ml3s1 = time.time()
        self.ml3s1_time = finish_ml3s1 - start_ml3s1

        r0_aloft = guess_r0_aloft
        L0_aloft = guess_L0_aloft

        ######## mL3S.3 ########
        start_ml3s3 = time.time()
        if self.target_array=='Covariance Map ROI':
            cov_fit_aloft = self.make_covariance_roi(self.air_mass, self.gs_pos, layer_alt_aloft[1:], 
                r0_aloft[1:], L0_aloft[1:], track_aloft, self.shwfs_shift, self.shwfs_rot, track_present=False, 
                offset_present=self.offset_present, fitting_L0=self.fitting_1_L0) * zeroSep_mult
        else:
            cov_fit_aloft = self.make_covariance_matrix(self.air_mass, self.gs_pos, layer_alt_aloft[1:], 
                r0_aloft[1:], L0_aloft[1:], track_aloft, self.shwfs_shift, self.shwfs_rot, 
                target_array=self.target_array, track_present=False, offset_present=self.offset_present, 
                fitting_L0=self.fitting_1_L0, matrix_xy=self.matrix_xy) * zeroSep_mult
        # pyplot.imshow(cov_fit_aloft)
        # f=op

        layer_alt_ground = numpy.array(self.layer_alt[:1])
        guess_r0_ground = numpy.array(self.guess_r0[:1])
        guess_L0_ground = numpy.array(self.guess_L0[:1])

        params = fitting_parameters('Direct Fit', self.target_array, self.n_wfs, self.pupil_mask, self.subap_diam, 
            self.wavelength, self.tel_diam, self.nx_subap, self.n_subap, self.gs_alt, self.gs_pos, self.allMapPos, 
            self.selector, self.xy_separations, 1, layer_alt_ground*self.air_mass, self.track_present, self.offset_present, 
            self.fit_3_track, self.fit_3_offset, self.fit_3_L0, guess_L0_ground, guess_r0_ground, self.roi_belowGround, 
            self.map_axis, self.roi_envelope, zeroSep_mult, self.mm, self.mmc, self.md, self.transform_matrix,
            self.matrix_xy, self.huge_matrix, self.styc_method, print_fitting=self.print_fitting)
        print('##################### FITTING mL3S.3 ######################','\n')
        theo_results, r0_ground, L0_ground, self.track, self.shwfs_rot, self.shwfs_shift, self.cov_fit, its_ml3s3, trash = params.covariance_fit(cov_meas-cov_fit_aloft, 
            guess_r0_ground, guess_L0_ground, self.track, self.shwfs_shift, self.shwfs_rot, 
            fit_r0=self.fit_3_r0, fit_L0=self.fit_3_L0, fit_track=self.fit_3_track, 
            fit_shift=self.fit_3_shift, fit_rot=self.fit_3_rot)
        finish_ml3s3 = time.time()
        self.ml3s3_time = finish_ml3s3 - start_ml3s3

        r0 = numpy.append(r0_ground, r0_aloft[1:])
        L0 = numpy.append(L0_ground, L0_aloft[1:])
        layer_alt = numpy.append(layer_alt_ground, layer_alt_aloft[1:])
        n_layer = layer_alt.shape[0]

        self.r0 = r0
        self.L0[:self.observable_bins] = L0

        self.cov_meas = cov_meas
        self.cov_aloft_meas = cov_aloft_meas
        self.total_its = totes_its_ml3s1 + totes_its_ml3s2 + its_ml3s3
        self.total_fitting_time = self.ml3s1_time + self.ml3s3_time

        print('###########################################################','\n')
        print('#################### mL3S FIT COMPLETE ####################','\n')
        print('################## FITTING SUCCESS :', theo_results['status']==1, '#################','\n')
        print('################ TIME TAKEN L3S.1 : '+"%6.4f" % self.ml3s1_time +' ################','\n')
        print('################ TIME TAKEN L3S.3 : '+"%6.4f" % self.ml3s3_time+' ################','\n')











    #2sl fitting specifications
    if config_file.configDict["FITTING ALGORITHM"]["type"]=='2sl':
        self.transform_matrix = transform_matrix(self.n_subap, self.n_wfs)
        self.direct_fit = False
        self._2sl_fit = True
        self.l3s_fit = False
        self.ml3s_fit = False
        fit_conf = fit_conf["2sl"]
        self.fit_1_r0 = fit_conf["fit_1_r0"]
        self.fit_1_L0 = fit_conf["fit_1_L0"]
        self.fit_1_globalL0 = fit_conf["fit_1_globalL0"]
        self.fit_1_rot = fit_conf["fit_1_rot"]
        self.fit_1_shift = fit_conf["fit_1_shift"]
        self.fit_1_track = False
        self.fit_1_offset=False
        self.fitting_1_L0 = False
        if self.fit_1_rot==True or self.fit_1_shift==True:
            self.fit_1_offset=True
            self.offset_present = True
        if self.fit_1_L0==True or self.fit_1_globalL0==True:
            self.fitting_1_L0 = True

        self.fit_2_r0 = fit_conf["fit_2_r0"]
        self.fit_2_L0 = fit_conf["fit_2_L0"]
        self.fit_2_rot = fit_conf["fit_2_rot"]
        self.fit_2_shift = fit_conf["fit_2_shift"]
        self.fit_2_track = fit_conf["fit_2_track"]
        self.fit_2_offset = False
        if self.fit_2_rot==True or self.fit_2_shift==True:
            self.fit_2_offset=True
            self.offset_present=True
        if self.fit_2_track==True:
            self.track_present=True




    def _2sl_fit_turbulence_(self, cov_meas, cov_aloft_meas, zeroSep_mult):
        """L3S fitting method for optical turbulence profiling
        
        Parameters:
            cov_meas (ndarray): covariance target array for direct fitting.
            cov_aloft_meas (ndarray): covariance target array for L3S fitting.
            zeroSep_mult (ndarray/float): array to null covariance at sub-aperture separations of zero (if zeroSep_cov==False)."""
        

        ######## 2SL.1 ########
        self.L0 = numpy.array([numpy.nan]*self.n_layer)
        layer_alt = numpy.array(self.layer_alt[:self.observable_bins])
        n_layer = layer_alt.shape[0]
        guess_r0 = numpy.array(self.guess_r0[:self.observable_bins])
        guess_L0 = numpy.array(self.guess_L0[:self.observable_bins])
        track_aloft = numpy.array([0., 0.])

        if self.offset_present==False and self.fit_1_offset==False:
            guess_shwfs_shift = self.guess_shwfs_shift * 0
            guess_shwfs_rot = self.guess_shwfs_rot * 0
        else:
            if self.fit_1_offset==True:
                if self.fit_1_shift==True:
                    guess_shwfs_shift = self.guess_shwfs_shift
                else:
                    guess_shwfs_shift = self.shwfs_shift
                
                if self.fit_1_rot==True:
                    guess_shwfs_rot = self.guess_shwfs_rot
                else:
                    guess_shwfs_rot = self.shwfs_rot
            else:
                guess_shwfs_shift = self.shwfs_shift
                guess_shwfs_rot = self.shwfs_rot

        params = fitting_parameters(self.fit_method, self.target_array, self.n_wfs, self.pupil_mask, self.subap_diam, 
            self.wavelength, self.tel_diam, self.nx_subap, self.n_subap, self.gs_alt, self.gs_pos, self.allMapPos, 
            self.selector, self.xy_separations, n_layer, layer_alt*self.air_mass, False, self.offset_present, 
            self.fit_1_track, self.fit_1_offset, self.fitting_1_L0, guess_L0, guess_r0, self.roi_belowGround, 
            self.map_axis, self.roi_envelope, zeroSep_mult, self.mm, self.mmc, self.md, self.transform_matrix, 
            self.matrix_xy, self.huge_matrix, self.styc_method, print_fitting=self.print_fitting)
        print('\n'+'###########################################################','\n')
        print('###################### FITTING 2SL.1 ######################','\n')
        theo_results, r0_aloft, L0_aloft, track_aloft, self.shwfs_rot, self.shwfs_shift, cov_2sl_fit, its_2sl1, start_2sl1 = params.covariance_fit(cov_aloft_meas, 
            guess_r0, guess_L0, track_aloft, guess_shwfs_shift, guess_shwfs_rot, fit_r0=self.fit_1_r0, 
            fit_L0=self.fit_1_L0, fit_track=False, fit_shift=self.fit_1_shift, fit_rot=self.fit_1_rot, 
            fit_globalL0=self.fit_1_globalL0)
        finish_2sl1 = time.time()
        self._2sl1_time = finish_2sl1 - start_2sl1

        ######## 2SL.2 ########
        start_2sl2 = time.time()
        if self.target_array=='Covariance Map ROI':
            cov_fit_aloft = self.make_covariance_roi(self.air_mass, self.gs_pos, layer_alt[1:], r0_aloft[1:], 
                L0_aloft[1:], track_aloft, self.shwfs_shift, self.shwfs_rot, track_present=False, 
                offset_present=self.offset_present, fitting_L0=self.fitting_1_L0) * zeroSep_mult
        else:
            cov_fit_aloft = self.make_covariance_matrix(self.air_mass, self.gs_pos, layer_alt[1:], r0_aloft[1:], 
                L0_aloft[1:], track_aloft, self.shwfs_shift, self.shwfs_rot, target_array=self.target_array, 
                track_present=False, offset_present=self.offset_present, fitting_L0=self.fitting_1_L0) * zeroSep_mult

        layer_alt_ground = numpy.array(layer_alt[:1])
        guess_r0_ground = numpy.array(guess_r0[:1])
        guess_L0_ground = numpy.array(guess_L0[:1])

        params = fitting_parameters('Direct Fit', self.target_array, self.n_wfs, self.pupil_mask, self.subap_diam, 
            self.wavelength, self.tel_diam, self.nx_subap, self.n_subap, self.gs_alt, self.gs_pos, self.allMapPos, 
            self.selector, self.xy_separations, 1, layer_alt_ground*self.air_mass, self.track_present, self.offset_present, 
            self.fit_2_track, self.fit_2_offset, self.fit_2_L0, guess_L0_ground, guess_r0_ground, self.roi_belowGround, 
            self.map_axis, self.roi_envelope, zeroSep_mult, self.mm, self.mmc, self.md, self.transform_matrix, 
            self.matrix_xy, self.huge_matrix, self.styc_method, print_fitting=self.print_fitting)
        print('###################### FITTING 2SL.2 ######################','\n')
        theo_results, r0_ground, L0_ground, self.track, self.shwfs_rot, self.shwfs_shift, cov_fit_ground, its_2sl2, trash = params.covariance_fit(cov_meas-cov_fit_aloft, 
            guess_r0_ground, guess_L0_ground, self.track, self.shwfs_shift, self.shwfs_rot, fit_r0=self.fit_2_r0, 
            fit_L0=self.fit_2_L0, fit_track=self.fit_2_track, fit_shift=self.fit_2_shift, fit_rot=self.fit_2_rot)
        finish_2sl2 = time.time()
        self._2sl2_time = finish_2sl2 - start_2sl2

        r0 = numpy.append(r0_ground, r0_aloft[1:])
        L0 = numpy.append(L0_ground, L0_aloft[1:])

        self.total_its = its_2sl1 + its_2sl2
        self.total_fitting_time = self._2sl1_time + self._2sl2_time

        if self.target_array=='Covariance Map ROI':
            self.cov_fit = self.make_covariance_roi(self.air_mass, self.gs_pos, layer_alt*self.air_mass, 
                r0, L0, self.track, self.shwfs_shift, self.shwfs_rot, track_present=self.track_present, 
                offset_present=self.offset_present, fitting_L0=True) * zeroSep_mult
        else:
            self.cov_fit = self.make_covariance_matrix(self.air_mass, self.gs_pos, layer_alt*self.air_mass, 
                r0, L0, self.track, self.shwfs_shift, self.shwfs_rot, target_array=self.target_array, 
                track_present=self.track_present, offset_present=self.offset_present, 
                fitting_L0=True, matrix_xy=self.matrix_xy) * zeroSep_mult

        self.r0 = r0
        self.L0[:self.observable_bins] = L0

        print('###########################################################','\n')
        print('##################### 2SL FIT COMPLETE ####################','\n')
        print('################## FITTING SUCCESS :', theo_results['status']==1, '#################','\n')
        print('################ TIME TAKEN 2SL.1 : '+"%6.4f" % self._2sl1_time+' ################','\n')
        print('################ TIME TAKEN 2SL.2 : '+"%6.4f" % self._2sl2_time+' ################','\n')