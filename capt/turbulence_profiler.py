import yaml
import time
import numpy
import itertools
from scipy.misc import comb
from matplotlib import pyplot; pyplot.ion()
from capt.misc_functions.cross_cov import cross_cov
from capt.fixed_telescope_arrays import grab_fixed_array
from capt.roi_functions.roi_from_map import roi_from_map
from capt.roi_functions.roi_from_map import roi_from_map
from capt.misc_functions.calc_Cn2_r0 import calc_Cn2, calc_r0
from capt.misc_functions.make_pupil_mask import make_pupil_mask
from capt.misc_functions.r0_centVariance import r0_centVariance
from capt.misc_functions.transform_matrix import transform_matrix
from capt.map_functions.covMap_fromMatrix import covMap_fromMatrix
from capt.misc_functions.dasp_cents_reshape import dasp_cents_reshape
from capt.roi_functions.roi_referenceArrays import roi_referenceArrays
from capt.misc_functions.calc_hmax_NGSandLGS import calc_hmax_NGSandLGS
from capt.roi_functions.roi_zeroSep_locations import roi_zeroSep_locations
from capt.matrix_functions.matrix_zeroSep_false import matrix_zeroSep_false
from capt.covariance_generation.generate_covariance_roi import covariance_roi
from capt.roi_functions.calculate_roi_covariance import calculate_roi_covariance
from capt.misc_functions.remove_tt_cents import remove_tt_cents, remove_tt_matrix
from capt.fitting_functions.covariance_fitting_algorithm import fitting_parameters
from capt.misc_functions.mapping_matrix import get_mappingMatrix, covMap_superFast
from capt.covariance_generation.generate_covariance_matrix import covariance_matrix
from capt.covariance_generation.generate_covariance_roi_l3s import covariance_roi_l3s






class configuration(object):
    """First function to be called. The specified configuration file is imported and 
    exceptions called if there are variable inconsistencies."""

    def __init__(self, config_file):
        """Reads in configuration file and checks for variable inconsistencies
        
        Parameters:
            config_file (yaml): configuration file for turbulence profiling."""

        self.loadYaml(config_file)

        #raise exception if exact target array isn't specified.
        target_conf = self.configDict["TARGET ARRAY"]
        array_target_conf = numpy.array([target_conf["covariance_matrix"], target_conf["covariance_matrix_no_xy"],
            target_conf["covariance_map"], target_conf["covariance_map_roi"]])
        if array_target_conf.sum()!=1:
            if array_target_conf.sum()>1:
                raise Exception('Must specify target array.')
            if array_target_conf.sum()==0:
                raise Exception('Must choose target array.')


        fit_type = self.configDict["FITTING ALGORITHM"]
        #raise exception if specified fitting algorithm is not known.
        if fit_type["type"]!='direct' and fit_type["type"]!='l3s':
            raise Exception('Specified fitting algorithm not known.')


        #raise exception if SHWFS parameters have unequal lengths.
        shwfs_params = self.configDict["SHWFS"]
        n_wfs = shwfs_params["n_wfs"]
        nx_subap = shwfs_params["nx_subap"]
        n_subap = shwfs_params["n_subap"]
        subap_diam = shwfs_params["subap_diam"]
        gs_alt = shwfs_params["gs_alt"]
        wavelength = shwfs_params["wavelength"]
        subap_fov = shwfs_params["subap_fov"]
        pxls_per_subap = shwfs_params["pxls_per_subap"]
        shift_offset_x = shwfs_params["shift_offset_x"]
        shift_offset_y = shwfs_params["shift_offset_y"]
        rotational_offset = shwfs_params["rotational_offset"]
        len_shwfs_params = numpy.array([n_wfs, len(nx_subap), len(n_subap), len(subap_diam), 
            len(gs_alt), len(wavelength), len(subap_fov), len(pxls_per_subap), len(shift_offset_x),
            len(shift_offset_y), len(rotational_offset)])
        for i in range(len_shwfs_params.shape[0]):
            test_len = len_shwfs_params[i]
            for j in range(len_shwfs_params.shape[0]):
                if test_len!=len_shwfs_params[j]:
                    raise Exception('Check lengths of SHWFS parameters.')
        for i in range(n_wfs):
            alt = gs_alt[i]
            for j in range(n_wfs):
                if alt!=gs_alt[j]:
                    raise Exception('GS altitudes must be equal. Need to update xy_seps*self.scale_factor in generate_covariance.')


        #raise exception if atmospheric parameters have unequal lengths.
        atmos_params = self.configDict["FITTING PARAMETERS"]
        n_layer = atmos_params["n_layer"]
        layer_alt = atmos_params["layer_alt"]
        guess_r0 = atmos_params["guess_r0"]
        guess_L0 = atmos_params["guess_L0"]
        len_atmos_params = numpy.array([n_layer, len(layer_alt), len(guess_r0), len(guess_L0)])
        # test_len = len_atmos_params[0]
        for i in range(len_atmos_params.shape[0]):
            test_len = len_atmos_params[i]
            for j in range(len_atmos_params.shape[0]):
                if test_len!=len_atmos_params[j]:
                    raise Exception('Check lengths of atmospheric parameters.')

        #raise exception if L3S requirements are not satisfied.
        fit_params = self.configDict["FITTING ALGORITHM"]
        if fit_params["type"]=='l3s':
            if n_layer<2:
                raise Exception("Need to fit at least 2 layers to perform L3S.")
            if layer_alt[0]!=0:
                raise Exception("For L3S, must fit ground layer at 0km.")
        if fit_params["remove_tt"]==True:
            if target_conf["covariance_map_roi"]==True:
                raise Exception('Tip-tilt removal not yet available for ROI processing.')

    def loadYaml(self, yaml_file):
        with open(yaml_file) as _file:
            self.configDict = yaml.load(_file)







class turbulence_profiler(object):
    """Master node for performing turbulence profiling
    NOTE: LGS configurations have not been fully tested."""

    def __init__(self, config_file):
        """Fixes variables from config_file that are to be used to complete turbulence profiling.

        Parameters:
        config_file (dict): fitting specifications set in imported yaml configuration file."""

        telescope_conf = config_file.configDict["TELESCOPE"]
        shwfs_conf = config_file.configDict["SHWFS"]
        atmos_conf = config_file.configDict["FITTING PARAMETERS"]
        fit_conf = config_file.configDict["FITTING ALGORITHM"]
        target_conf = config_file.configDict["TARGET ARRAY"] 

        #system configuration
        self.n_wfs = shwfs_conf["n_wfs"]
        self.tel_diam = telescope_conf["tel_diam"]
        self.ao_system = telescope_conf["ao_system"]
        #Determine no. of GS combinations
        self.combs = int(comb(self.n_wfs, 2, exact=True))
        #Selection array for all combinations
        selector = numpy.array((range(self.n_wfs)))
        self.selector = numpy.array((list(itertools.combinations(selector, 2))))
        self.known_ao_system = False
        if self.ao_system=='CANARY' or self.ao_system=='AOF' or self.ao_system=='HARMONI':
            self.known_ao_system = True

        #fitting parameters
        self.n_layer = int(atmos_conf["n_layer"])
        self.layer_alt = numpy.array(atmos_conf["layer_alt"]).astype('float64')
        self.guess_r0 = numpy.array(atmos_conf["guess_r0"]).astype('float64')
        self.guess_L0 = numpy.array(atmos_conf["guess_L0"]).astype('float64')

        #shwfs parameters
        self.plate_scale = shwfs_conf["plate_scale"]
        self.nx_subap = shwfs_conf["nx_subap"]
        self.n_subap = shwfs_conf["n_subap"]
        self.loop_time = shwfs_conf["loop_time"]
        self.subap_diam = shwfs_conf["subap_diam"]
        self.wavelength = numpy.array(shwfs_conf["wavelength"]).astype('float64')
        self.offset_present = shwfs_conf["offset_present"]
        self.gs_alt = numpy.array(shwfs_conf["gs_alt"]).astype('float64')
        self.pix_arc = numpy.array(shwfs_conf["subap_fov"]) / numpy.array(shwfs_conf["pxls_per_subap"])
        self.shwfs_rot = numpy.array(shwfs_conf["rotational_offset"])
        self.shwfs_shift = numpy.hstack((numpy.array([shwfs_conf["shift_offset_x"]]).T, numpy.array([
            shwfs_conf["shift_offset_y"]]).T))
        self.offset_present = numpy.array(shwfs_conf["offset_present"])
        self.mask = telescope_conf["mask"]
        if self.mask=='circle':
            self.pupil_mask = make_pupil_mask(self.mask, self.n_subap, self.nx_subap[0], 
                telescope_conf["obs_diam"], self.tel_diam)
            self.n_subap_from_pupilMask = numpy.array([int(self.pupil_mask.sum())]*self.n_wfs)


        #target array parameters
        self.data_type = target_conf["data_type"]
        self.tt_track = numpy.array(target_conf["tt_track"])
        self.tt_track_present = numpy.array(target_conf["tt_track_present"])
        self.zeroSep_cov = target_conf["zeroSep_cov"]
        self.mapping_type = target_conf["mapping_type"]
        self.cent_start = target_conf["cent_start"]
        self.input_matrix = target_conf["input_matrix"]
        self.covariance_matrix = target_conf["covariance_matrix"]
        self.covariance_matrix_no_xy = target_conf["covariance_matrix_no_xy"]
        self.covariance_map = target_conf["covariance_map"]
        self.covariance_map_roi = target_conf["covariance_map_roi"]
        self.zeroSep_locations = False
        self.zeroSep_locations_l3s3 = False
        self.map_axis = target_conf["map_axis"]
        self.mm = numpy.nan
        self.mmc = numpy.nan
        self.md = numpy.nan
        if self.covariance_map_roi==True:
            self.lgs_track = numpy.zeros((self.combs, 2))
        else:
            self.lgs_track = numpy.zeros((self.combs+self.n_wfs, 2))

        #fitting alogirithm parameters
        self.repetitions = fit_conf["repetitions"]
        self.styc_method = fit_conf["styc_method"]
        self.print_fitting = fit_conf["print_fitting"]
        self.huge_matrix = fit_conf["huge_matrix"]
        self.force_altRange = fit_conf["force_altRange"]
        self.remove_tt = fit_conf["remove_tt"]
        self.cn2_noiseFloor = numpy.array(fit_conf["cn2_noiseFloor"]).astype(float)
        self.output_fitted_array = fit_conf["output_fitted_array"]
        self.fit_unsensed = fit_conf["fit_unsensed"]
        self.r0_unsensed = numpy.nan

        #build matrix to remove tip-tilt for each shwfs
        self.remove_tt_matrix = None
        if self.remove_tt==True:
            if self.mask=='circle':
                self.remove_tt_matrix = remove_tt_matrix(self.n_subap[0], self.n_wfs)
        
        #direct fitting specifications
        if config_file.configDict["FITTING ALGORITHM"]["type"]=='direct':
            self.l3s1_matrix = None
            self.direct_fit = True
            self.l3s_fit = False
            fit_conf = fit_conf["direct"]
            self.fit_r0 = fit_conf["fit_r0"]
            self.fit_L0 = fit_conf["fit_L0"]
            self.fit_globalL0 = fit_conf["fit_globalL0"]
            self.fit_groundL0 = fit_conf["fit_groundL0"]
            self.fit_tt_track = fit_conf["fit_tt_track"]
            self.fit_rot = fit_conf["fit_rot"]
            self.fit_shift = fit_conf["fit_shift"]
            self.fitting_L0 = False
            self.fit_offset = False
            self.fit_lgs_track = fit_conf["fit_lgs_track"]
            if self.fit_L0==True or self.fit_globalL0==True or self.fit_groundL0==True:
                self.fitting_L0 = True
            if self.fit_rot==True or self.fit_shift==True:
                self.fit_offset=True
            if self.fit_tt_track==True:
                self.tt_track_present = True
            self.lgs_track_present = False
            if self.fit_lgs_track==True:
                self.lgs_track_present = True


        #l3s fitting specifications
        if config_file.configDict["FITTING ALGORITHM"]["type"]=='l3s':
            self.l3s1_matrix = transform_matrix(self.n_subap, self.n_wfs)
            self.direct_fit = False
            self.l3s_fit= True
            fit_conf = fit_conf["l3s"]
            self.fit_1_r0 = fit_conf["fit_1_r0"]
            self.fit_1_L0 = fit_conf["fit_1_L0"]
            self.fit_1_globalL0 = fit_conf["fit_1_globalL0"]
            self.fit_1_rot = fit_conf["fit_1_rot"]
            self.fit_1_shift = fit_conf["fit_1_shift"]
            self.fit_1_tt_track = False
            self.fit_1_lgs_track = fit_conf["fit_1_lgs_track"]
            self.fit_1_offset=False
            self.fitting_1_L0 = False
            if self.fit_1_rot==True or self.fit_1_shift==True:
                self.fit_1_offset=True
                self.offset_present=True
            if self.fit_1_L0==True or self.fit_1_globalL0==True:
                self.fitting_1_L0 = True

            self.fit_2_r0 = fit_conf["fit_2_r0"]
            self.fit_2_L0 = fit_conf["fit_2_L0"]
            self.fit_2_rot = fit_conf["fit_2_rot"]
            self.fit_2_shift = fit_conf["fit_2_shift"]
            self.fit_2_tt_track = fit_conf["fit_2_tt_track"]
            self.fit_2_lgs_track = fit_conf["fit_2_lgs_track"]
            self.fit_2_offset = False
            if self.fit_2_rot==True or self.fit_2_shift==True:
                self.fit_2_offset=True
                self.offset_present=True
            if self.fit_2_tt_track==True:
                self.tt_track_present=True

            self.fit_3 = fit_conf["fit_3"]
            self.fit_3_r0 = fit_conf["fit_3_r0"]
            self.fit_3_L0 = fit_conf["fit_3_L0"]
            self.fit_3_globalL0 = False
            self.fit_3_groundL0 = False
            self.fit_3_rot = fit_conf["fit_3_rot"]
            self.fit_3_shift = fit_conf["fit_3_shift"]
            self.fit_3_tt_track = fit_conf["fit_3_tt_track"]
            self.fit_3_lgs_track = fit_conf["fit_3_lgs_track"]
            self.fitting_3_L0 = False
            self.fit_3_offset=False
            self.fit_3_L0 = False
            if self.fit_3_rot==True or self.fit_3_shift==True:
                self.fit_3_offset=True
            if self.fit_3_L0==True or self.fit_3_globalL0==True or self.fit_3_groundL0==True:
                self.fitting_3_L0 = True
            self.lgs_track_present = False
            if self.fit_1_lgs_track==True or self.fit_2_lgs_track==True:
                self.lgs_track_present = True




        if self.covariance_map==True or self.covariance_map_roi==True:
            self.map_axis = target_conf["map_axis"]

            self.mm_built = False
            if self.mask=='circle':
                if self.known_ao_system==True:
                    self.mm, self.mmc, self.md = grab_fixed_array.get_mappingMatrix(self.ao_system)
                else:
                    #get map of ones
                    matrix_region_ones = numpy.ones((self.n_subap[0], self.n_subap[0]))
                    self.mm, self.mmc, self.md = get_mappingMatrix(self.pupil_mask, matrix_region_ones)
                self.mm_built = True

            # if self.covariance_map_roi==True:
            self.roi_belowGround = target_conf["roi_belowGround"]
            self.roi_envelope = target_conf["roi_envelope"]
            if self.l3s_fit==True:
                if self.fit_3==True:
                    self.roi_belowGround_l3s3 = fit_conf["roi_belowGround_l3s3"]
                    self.roi_envelope_l3s3 = fit_conf["roi_envelope_l3s3"]

        print('\n'+'###########################################################','\n')
        print('######### TURBULENCE PROFILING PARAMETERS SECURE ##########')







    def perform_turbulence_profiling(self, air_mass, tas, pix_arc, shwfs_centroids, input_pupilMask=False, cov_matrix=False):
        """Performs tubrulence profiling by first processing shwfs_centroids/cov_matrix into covariance target array.
        Target array is then iteratively fitted to using Levenberg-Marquardt algorithm.
        
        Parameters:
            air_mass (float): air mass of observation.
            tas (ndarray): pick off positions of target acquisition system.
            shwfs_centroids (ndarray): shwfs centroid positions in x and y over some time interval. 
            cov_matrix (ndarray): pre-computed covariance matrix to be fitted to (if input_matrix=='True')"""
        
        
        if self.mask=='custom':
            if str(input_pupilMask)=='False':
                raise Exception('Custom pupil mask has not been set.')
            if numpy.sum(input_pupilMask)==0:
                raise Exception('Custom pupil mask has zero sub-apertures.')
            
            self.pupil_mask = input_pupilMask
            self.n_subap_from_pupilMask = numpy.array([int(self.pupil_mask.sum())]*self.n_wfs)
            matrix_region_ones = numpy.ones((self.n_subap_from_pupilMask[0], self.n_subap_from_pupilMask[0]))
            self.mm, self.mmc, self.md = get_mappingMatrix(self.pupil_mask, matrix_region_ones)
            self.mm_built = True
            if self.l3s_fit==True:
                self.l3s1_matrix = transform_matrix(self.n_subap_from_pupilMask, self.n_wfs)
            if self.remove_tt==True:
                self.remove_tt_matrix = remove_tt_matrix(self.n_subap_from_pupilMask[0], self.n_wfs)



        if self.zeroSep_cov==False:
            self.get_zeroSep_locations()

        if self.data_type=='canary':
            self.pix_arc = pix_arc
        if self.data_type=='dasp' and self.input_matrix==False:
            shwfs_centroids = dasp_cents_reshape(shwfs_centroids, self.pupil_mask, self.n_wfs)

        self.matrix_xy=False
        if self.covariance_matrix==True:
            self.target_array = 'Covariance Matrix'
            self.matrix_xy = True
        if self.covariance_matrix_no_xy==True:
            self.target_array = 'Covariance Matrix (No XY)'
        if self.covariance_map==True:
            self.target_array = 'Covariance Map'
        if self.covariance_map_roi==True:
            self.target_array = 'Covariance Map ROI'

        self.time_calc_cov = 0.
        self.import_tools(air_mass, tas, shwfs_centroids)
        if self.input_matrix==False:
            
            print('\n'+'###########################################################','\n')
            print('############### PROCESSING SHWFS CENTROIDS ################','\n')
            print('###########################################################','\n')

            if self.covariance_map_roi==True:
                self.cov_meas, cov_meas_l3s3, cov_aloft_meas = self.process_roi_cents()
            else:
                self.cov_meas, cov_meas_l3s3, cov_aloft_meas = self.process_matrix_cents()        
        else:
            
            print('\n'+'###########################################################','\n')
            print('################# PROCESSING INPUT MATRIX #################','\n')
            print('###########################################################')

            self.cov_meas, cov_meas_l3s3, cov_aloft_meas = self.process_input_matrix(cov_matrix)

        if self.direct_fit==True:
            self.fit_method = 'Direct Fit'
            self.r0 = numpy.array(self.guess_r0[:self.observable_bins])
            self.L0 = numpy.array(self.guess_L0[:self.observable_bins])
            for i in range(self.repetitions):
                self._direct_fit_turbulence_(self.cov_meas, i)

        if self.l3s_fit==True:
            self.fit_method = 'L3S Fit'
            self.r0 = numpy.array(self.guess_r0[:self.observable_bins])
            self.L0 = numpy.array(self.guess_L0[:self.observable_bins])
            for i in range(self.repetitions):
                self._l3s_fit_turbulence_(self.cov_meas, cov_meas_l3s3, cov_aloft_meas, i)

        return self




    def print_results(self, fitting_iteration):
        print('\n'+'###########################################################','\n')
        print('##################### FITTING RESULTS #####################','\n')
        print('Repetition No.: {}'.format(fitting_iteration))
        print('Target Array: '+self.target_array)
        print('Fitting Method: '+self.fit_method)
        print('Total Iterations : {}'.format(self.total_its))
        if self.input_matrix==False:
            print('Variance r0 (m) : {}'.format(r0_centVariance(self.shwfs_centroids[:, :self.n_subap[0]*2], self.wavelength[0], self.subap_diam[0])))
        print('Integrated r0 (m) : {}'.format( calc_r0(numpy.sum(self.Cn2[:self.observable_bins]), self.wavelength[0])))
        print('Unsensed r0 (m) : {}'.format(self.r0_unsensed), '\n')
        print('Layer Altitudes (m): {}'.format(self.layer_alt))
        print('r0 (m) : {}'.format(self.r0))
        print('Cn2 (m^-1/3) : {}'.format(self.Cn2))
        print('L0 (m) : {}'.format(self.L0))
        print('TT Track (arcsec^2) : {}'.format(self.tt_track))
        print('LGS X Track (arcsec^2) : {}'.format(self.lgs_track.T[0]))
        print('LGS Y Track (arcsec^2) : {}'.format(self.lgs_track.T[1]))
        print('SHWFS x Shift (m) : {}'.format(self.shwfs_shift.T[0]))
        print('SHWFS y Shift (m) : {}'.format(self.shwfs_shift.T[1]))
        print('SHWFS Rotation (degrees) : {}'.format(self.shwfs_rot))

        print('\n'+'###########################################################','\n')
        print('##################### TOTAL TIME TAKEN ####################','\n')
        print('############# CALCULATING COVARIANCE : '+"%6.4f" % (self.time_calc_cov)+' #############','\n')
        print('################ FITTING COVARIANCE : '+"%6.4f" % (self.total_fitting_time)+' ##############', '\n')
        print('###########################################################')





    def _direct_fit_turbulence_(self, cov_meas, repetition_no):
        """Direct fitting method for optical turbulence profiling
        
        Parameters:
            cov_meas (ndarray): covariance target array for direct fitting.
            zeroSep_locations (ndarray/float): array to null covariance at sub-aperture separations of zero (if zeroSep_cov==False)."""
        
        layer_alt = numpy.array(self.layer_alt[:self.observable_bins])
        n_layer = layer_alt.shape[0]

        params = fitting_parameters(self, self.fit_method, self.target_array, n_layer, layer_alt*self.air_mass, 
            self.tt_track_present, False, self.offset_present, self.fit_unsensed, self.fit_tt_track, self.fit_lgs_track, self.fit_offset, self.fitting_L0, 
            self.L0, self.r0, self.roi_belowGround, self.roi_envelope, self.zeroSep_locations, 
            self.allMapPos, self.xy_separations)
        print('\n'+'###########################################################','\n')
        print('##################### DIRECT FITTING ######################')
        success, r0, L0, self.tt_track, self.lgs_track, self.shwfs_rot, self.shwfs_shift, self.cov_fit, self.total_its, self.start_fit_cov = params.covariance_fit(
            self, cov_meas, layer_alt*self.air_mass, self.r0, self.L0, self.tt_track, self.lgs_track, self.shwfs_shift, self.shwfs_rot, 
            fit_r0=self.fit_r0, fit_L0=self.fit_L0, fit_tt_track=self.fit_tt_track, fit_lgs_track=self.fit_lgs_track, fit_shift=self.fit_shift, 
            fit_rot=self.fit_rot, fit_globalL0=self.fit_globalL0, fit_groundL0=self.fit_groundL0)
        
        self.finish_fit_cov = time.time()
        self.total_fitting_time = self.finish_fit_cov - self.start_fit_cov

        self.r0 = numpy.array([numpy.nan]*self.n_layer)
        self.L0 = numpy.array([numpy.nan]*self.n_layer)
        self.r0[:self.observable_bins] = r0[:n_layer]
        self.L0[:self.observable_bins] = L0[:n_layer]

        print('\n'+'###########################################################','\n')
        print('################### DIRECT FIT COMPLETE ###################','\n')
        print('################# FITTING SUCCESS :', success, '##################','\n')
        print('################### TIME TAKEN : '+"%6.4f" % self.total_fitting_time+' ###################')
        self.amend_results()
        self.print_results(repetition_no)

        



    def _l3s_fit_turbulence_(self, cov_meas, cov_meas_l3s3, cov_aloft_meas, repetition_no):
        """L3S fitting method for optical turbulence profiling
        
        Parameters:
            cov_meas (ndarray): covariance target array for direct fitting.
            cov_aloft_meas (ndarray): covariance target array for L3S fitting."""
        

        ######## L3S.1 ########
        layer_alt_aloft = numpy.array(self.layer_alt[:self.observable_bins])
        n_layer_aloft = layer_alt_aloft.shape[0]
        tt_track_aloft = numpy.array([0., 0.])


        params = fitting_parameters(self, self.fit_method, self.target_array, n_layer_aloft, 
            layer_alt_aloft*self.air_mass, False, self.lgs_track_present, self.offset_present, self.fit_unsensed, 
            self.fit_1_tt_track, self.fit_1_lgs_track, self.fit_1_offset, self.fitting_1_L0, self.L0[:n_layer_aloft], 
            self.r0[:n_layer_aloft], self.roi_belowGround, self.roi_envelope, self.zeroSep_locations, self.allMapPos, 
            self.xy_separations)
        print('\n'+'###########################################################','\n')
        print('###################### FITTING L3S.1 ######################','\n')
        success, r0_aloft, L0_aloft, tt_track_aloft, lgs_track_aloft, self.shwfs_rot, self.shwfs_shift, cov_l3s_fit, its_l3s1, start_l3s1 = params.covariance_fit(
            self, cov_aloft_meas, layer_alt_aloft*self.air_mass, self.r0[:n_layer_aloft], self.L0[:n_layer_aloft], tt_track_aloft, self.lgs_track, self.shwfs_shift, 
            self.shwfs_rot, fit_r0=self.fit_1_r0, fit_L0=self.fit_1_L0, fit_tt_track=False, fit_lgs_track=self.fit_1_lgs_track, 
            fit_shift=self.fit_1_shift, fit_rot=self.fit_1_rot, fit_globalL0=self.fit_1_globalL0)
        finish_l3s1 = time.time()
        self.l3s1_time = finish_l3s1 - start_l3s1

        if self.fit_unsensed==True:
            self.r0_unsensed = r0_aloft[n_layer_aloft:]
            layer_alt_aloft = numpy.append(layer_alt_aloft, 1e20)            
            # r0_aloft = r0_aloft[:n_layer_aloft]
            # L0_aloft = L0_aloft[:n_layer_aloft]

        ######## L3S.2 ########
        if self.target_array=='Covariance Map ROI':
            cov_meas_ground = cov_meas - self.make_covariance_roi(self.pupil_mask, self.air_mass, self.gs_pos, self.gs_alt, layer_alt_aloft[1:], 
                r0_aloft[1:], L0_aloft[1:], tt_track_aloft, lgs_track_aloft, self.shwfs_shift, self.shwfs_rot, tt_track_present=False, 
                lgs_track_present=self.fit_1_lgs_track, offset_present=self.offset_present, fitting_L0=self.fitting_1_L0)
        else:
            cov_meas_ground = cov_meas - self.make_covariance_matrix(self.pupil_mask, self.air_mass, self.gs_pos, self.gs_alt, layer_alt_aloft[1:], 
                r0_aloft[1:], L0_aloft[1:], tt_track_aloft, lgs_track_aloft, self.shwfs_shift, self.shwfs_rot, 
                target_array=self.target_array, tt_track_present=False, lgs_track_present=self.fit_1_lgs_track, 
                offset_present=self.offset_present, fitting_L0=self.fitting_1_L0, matrix_xy=self.matrix_xy, 
                l3s1_transform=False)

        l3s_2_1_time = time.time() - self.timingStart

        layer_alt_ground = numpy.array(self.layer_alt[:1])
        guess_r0_ground = numpy.array(self.r0[:1])
        guess_L0_ground = numpy.array(self.L0[:1])

        # pyplot.figure()
        # pyplot.imshow(cov_fit_aloft)

        params = fitting_parameters(self, 'Direct Fit', self.target_array, 1, layer_alt_ground*self.air_mass, 
            self.tt_track_present, False, self.offset_present, False, self.fit_2_tt_track, self.fit_2_lgs_track, self.fit_2_offset, self.fit_2_L0, 
            guess_L0_ground, guess_r0_ground, self.roi_belowGround, self.roi_envelope, self.zeroSep_locations, 
            self.allMapPos, self.xy_separations)
        print('###################### FITTING L3S.2 ######################','\n')
        success, r0_ground, L0_ground, tt_track_ground, lgs_track_ground, self.shwfs_rot, self.shwfs_shift, cov_fit_ground, its_l3s2, start_l3s2_2 = params.covariance_fit(
            self, cov_meas_ground, layer_alt_ground*self.air_mass, guess_r0_ground, guess_L0_ground, self.tt_track, 
            self.lgs_track, self.shwfs_shift, self.shwfs_rot, fit_r0=self.fit_2_r0, fit_L0=self.fit_2_L0, 
            fit_tt_track=self.fit_2_tt_track, fit_lgs_track=self.fit_2_lgs_track, fit_shift=self.fit_2_shift, fit_rot=self.fit_2_rot)
        l3s_2_2_time = time.time() - start_l3s2_2
        self.l3s2_time = l3s_2_1_time + l3s_2_1_time

        r0 = numpy.append(r0_ground, r0_aloft[1:n_layer_aloft])
        Cn2 = calc_Cn2(r0, self.wavelength[0])/self.air_mass
        L0 = numpy.append(L0_ground, L0_aloft[1:n_layer_aloft])
        self.lgs_track = lgs_track_aloft + lgs_track_ground
        self.tt_track = tt_track_aloft + tt_track_ground
        layer_alt = numpy.append(layer_alt_ground, layer_alt_aloft[1:n_layer_aloft])
        n_layer = layer_alt.shape[0]


        ######## L3S. 3 ########
        if self.fit_3==True:
            
            layer_alt3 = layer_alt[Cn2>self.cn2_noiseFloor]
            r03 = r0[Cn2>self.cn2_noiseFloor]
            L03 = L0[Cn2>self.cn2_noiseFloor]
            n_layer3 = layer_alt3.shape[0]

            if self.covariance_map_roi==True:
                cov_meas = cov_meas_l3s3

            params = fitting_parameters(self, 'Direct Fit', self.target_array, n_layer3, layer_alt3*self.air_mass, 
                self.tt_track_present, self.lgs_track_present, self.offset_present, False, self.fit_3_tt_track, self.fit_3_lgs_track, 
                self.fit_3_offset, self.fitting_3_L0, L03, r03, self.roi_belowGround_l3s3, self.roi_envelope_l3s3, self.zeroSep_locations_l3s3, 
                self.allMapPos_l3s3, self.xy_separations_l3s3)
            print('###################### FITTING L3S.3 ######################','\n')
            success, r03, L03, self.tt_track, self.lgs_track, self.shwfs_rot, self.shwfs_shift, self.cov_fit, its_l3s3, start_l3s3 = params.covariance_fit(
                    self, cov_meas, layer_alt3*self.air_mass, r03, L03, self.tt_track, self.lgs_track, self.shwfs_shift, self.shwfs_rot, 
                    fit_r0=self.fit_3_r0, fit_L0=self.fit_3_L0, fit_tt_track=self.fit_3_tt_track, fit_lgs_track=self.fit_3_lgs_track, 
                    fit_shift=self.fit_3_shift, fit_rot=self.fit_3_rot, fit_globalL0=self.fit_3_globalL0, fit_groundL0=self.fit_3_groundL0)
            finish_l3s3 = time.time()
            self.l3s3_time = finish_l3s3 - start_l3s3 
        
        if self.fit_3==False:
            if self.output_fitted_array==True:
                if self.target_array=='Covariance Map ROI':
                    self.cov_fit = self.make_covariance_roi(self.pupil_mask, self.air_mass, self.gs_pos, self.gs_alt, layer_alt, 
                        r0, L0, self.tt_track, self.lgs_track, self.shwfs_shift, self.shwfs_rot, 
                        l3s1_transform=False, tt_track_present=self.tt_track_present, lgs_track_present=self.lgs_track_present, 
                        offset_present=self.offset_present, fitting_L0=True)
                else:
                    self.cov_fit = self.make_covariance_matrix(self.pupil_mask, self.air_mass, self.gs_pos, self.gs_alt, layer_alt*self.air_mass, 
                        r0, L0, self.tt_track, self.lgs_track, self.shwfs_shift, self.shwfs_rot, target_array=self.target_array, 
                        tt_track_present=self.tt_track_present, lgs_track_present=self.lgs_track_present, offset_present=self.offset_present, 
                        fitting_L0=True)
                its_l3s3 = 0
                self.l3s3_time = 0
            else:
                self.cov_fit = False
                its_l3s3 = 0
                self.l3s3_time = 0

        self.r0 = numpy.array([numpy.nan]*self.n_layer)
        self.L0 = numpy.array([numpy.nan]*self.n_layer)
        self.r0[:self.observable_bins] = r0
        self.L0[:self.observable_bins] = L0

        self.total_its = its_l3s1 + its_l3s2 + its_l3s3
        self.total_fitting_time = self.l3s1_time + self.l3s2_time + self.l3s3_time

        print('###########################################################','\n')
        print('##################### L3S FIT COMPLETE ####################','\n')
        print('################## FITTING SUCCESS : '+str(success==True)+' #################','\n')
        print('################ TIME TAKEN L3S.1 : '+"%6.4f" % self.l3s1_time+' ################','\n')
        print('################ TIME TAKEN L3S.2 : '+"%6.4f" % self.l3s2_time+' ################','\n')
        print('################ TIME TAKEN L3S.3 : '+"%6.4f" % self.l3s3_time+' ################')

        self.amend_results()
        self.print_results(repetition_no)





    def get_zeroSep_locations(self):
        """Locations of zero sub-aperture separation within either matrix or map ROI
        
        Returns:
            ndarray: locations of zero sub-aperture separation"""

        if self.covariance_map_roi==True:
            roi_width = (2*self.roi_envelope) + 1
            roi_length = self.pupil_mask.shape[0] + self.roi_belowGround
            self.zeroSep_locations = roi_zeroSep_locations(self.combs, roi_width, 
                roi_length, self.map_axis, self.roi_belowGround)
            if self.l3s_fit==True:
                if self.fit_3==True:
                    roi_width_l3s3 = (2*self.roi_envelope_l3s3) + 1
                    roi_length_l3s3 = self.pupil_mask.shape[0] + self.roi_belowGround_l3s3
                    self.zeroSep_locations_l3s3 = roi_zeroSep_locations(self.combs, roi_width_l3s3, 
                        roi_length_l3s3, self.map_axis, self.roi_belowGround_l3s3)

        else:
            zeroSep_zeros = matrix_zeroSep_false(self.n_subap_from_pupilMask)
            self.zeroSep_locations = numpy.where(zeroSep_zeros==0)
            if self.l3s_fit==True:
                if self.fit_3==True:
                    self.zeroSep_locations_l3s3 = self.zeroSep_locations
            del zeroSep_zeros
        

        



    def make_covariance_matrix(self, pupil_mask, air_mass, gs_pos, gs_alt, layer_alt, r0, L0, tt_track, lgs_track, shwfs_shift, shwfs_rot, remove_tt=False, 
        l3s1_transform=False, target_array=False, tt_track_present=False, lgs_track_present=False, offset_present=False, fitting_L0=False, matrix_xy=True):
        """Analytically generate covariance matrix.
        
        Parameters:
            layer_alt (ndarray): altitudes of turbulent layers (m).
            r0 (ndarray): r0 profile.
            L0 (ndarray): L0 profile.
            track (ndarray): tracking addition to covariance.
            shwfs_shift (ndarray): SHWFS shift in x and y (m).
            shwfs_rot (ndarray): SHWFS rotation.
            target_array (str): set 'Covariance Map' if you want covariance map.
            track_present (bool): generate covariance matrix with given linear additions to covariance (from vibrations/track).
            offset_present (bool): determines whether covariance matrix is to account for given SHWFS shift/rotation.
            fitting_L0 (bool): determines whether generated covariance map ROI is to be used for L0 profiling.
            
        Returns:
            ndarray: covariance matrix or map."""

        r0 = numpy.array(r0)
        L0 = numpy.array(L0)
        shwfs_shift = numpy.array(shwfs_shift)
        shwfs_rot = numpy.array(shwfs_rot)
        gs_dist = numpy.array(gs_alt) * air_mass
        layer_dist = numpy.array(layer_alt) * air_mass
        if gs_pos.shape[0]!=self.n_wfs:
            raise Exception("Check number of GSs vs. SHWFS parameters.")

        generationParams = covariance_matrix(pupil_mask, self.subap_diam, self.wavelength, 
            self.tel_diam, self.n_subap, gs_dist, gs_pos, len(layer_alt), layer_dist, 
            r0, L0, styc_method=self.styc_method, wind_profiling=False, tt_track_present=tt_track_present, 
            lgs_track_present=lgs_track_present, offset_present=offset_present, fit_layer_alt=False, 
            fit_tt_track=False, fit_lgs_track=False, fit_offset=False, fit_L0=fitting_L0, 
            matrix_xy=matrix_xy, huge_matrix=self.huge_matrix, l3s1_transform=l3s1_transform, 
            l3s1_matrix=self.l3s1_matrix, remove_tt=remove_tt, remove_tt_matrix=self.remove_tt_matrix)
        self.timingStart = generationParams.timingStart

        cov = generationParams._make_covariance_matrix_((layer_dist).astype('float'), 
            r0.astype('float'), L0.astype('float'), tt_track=tt_track, lgs_track=lgs_track, 
            shwfs_shift=shwfs_shift.astype('float'), shwfs_rot=shwfs_rot.astype('float'))

        if self.zeroSep_cov==False:
            cov[self.zeroSep_locations] = 0.

        if target_array=='Covariance Map' or target_array=='Covariance Map ROI':
            if self.mm_built==False:
                matrix_region_ones = numpy.ones((int(pupil_mask.sum()), int(pupil_mask.sum())))
                self.mm, self.mmc, self.md = get_mappingMatrix(pupil_mask, matrix_region_ones)
                self.mm_built = True

            reduced_n_subap = int(pupil_mask.sum())
            self.n_subap_from_pupilMask = numpy.array([reduced_n_subap]*self.n_wfs)
            cov = covMap_fromMatrix(cov, self.n_wfs, self.nx_subap, self.n_subap_from_pupilMask, pupil_mask, 
                self.map_axis, self.mm, self.mmc, self.md)

            # stop
            if target_array=='Covariance Map ROI':
                cov = roi_from_map(cov, gs_pos, pupil_mask, self.selector, 
                    self.roi_belowGround, self.roi_envelope)

        return cov







    def make_covariance_roi(self, pupil_mask, air_mass, gs_pos, gs_alt, layer_alt, r0, L0, tt_track, lgs_track, shwfs_shift, shwfs_rot, l3s1_transform=False,
        tt_track_present=False, lgs_track_present=False, offset_present=False, fitting_L0=False):
        """Analytically generate covariance map ROI.
        
        Parameters:
            layer_alt (ndarray): altitudes of turbulent layers (m).
            r0 (ndarray): r0 profile.
            L0 (ndarray): L0 profile.
            tt_track (ndarray): tracking addition to covariance.
            shwfs_shift (ndarray): SHWFS shift in x and y (m).
            shwfs_rot (ndarray): SHWFS rotation.
            tt_track_present (bool): generate covariance matrix with given linear additions to covariance (from vibrations/track).
            offset_present (bool): determines whether covariance matrix is to account for given SHWFS shift/rotation.
            fitting_L0 (bool): determines whether generated covariance map ROI is to be used for L0 profiling.
            
        Returns:
            ndarray: covariance map ROI."""

        r0 = numpy.array(r0)
        L0 = numpy.array(L0)
        gs_pos = numpy.array(gs_pos)
        shwfs_shift = numpy.array(shwfs_shift)
        shwfs_rot = numpy.array(shwfs_rot)
        gs_dist = numpy.array(gs_alt) * air_mass
        layer_dist = numpy.array(layer_alt) * air_mass
        if gs_pos.shape[0]!=self.n_wfs:
            raise Exception("Check number of GSs vs. SHWFS parameters.")

        if l3s1_transform==False:
            onesMat, wfsMat_1, wfsMat_2, allMapPos, selector, xy_separations = roi_referenceArrays(pupil_mask, 
                gs_pos, self.tel_diam, self.roi_belowGround, self.roi_envelope)

            generationParams = covariance_roi(pupil_mask, self.subap_diam, 
                    self.wavelength, self.tel_diam, self.n_subap, gs_dist, gs_pos, layer_alt.shape[0], layer_dist, 
                    L0, allMapPos, xy_separations, self.map_axis, styc_method=self.styc_method, wind_profiling=False, 
                    tt_track_present=tt_track_present, lgs_track_present=lgs_track_present,offset_present=offset_present, 
                    fit_layer_alt=False, fit_tt_track=False, fit_lgs_track=False, fit_offset=False, fit_L0=fitting_L0)

            cov_roi = generationParams._make_covariance_roi_((layer_dist).astype('float'), 
                r0.astype('float'), L0.astype('float'), tt_track=tt_track, lgs_track=lgs_track, 
                shwfs_shift=shwfs_shift.astype('float'), shwfs_rot=shwfs_rot.astype('float'))

        if l3s1_transform==True:
            onesMat, wfsMat_1, wfsMat_2, allMapPos_acrossMap, selector, xy_separations_acrossMap = roi_referenceArrays(
                pupil_mask, gs_pos, self.tel_diam, pupil_mask.shape[0]-1, self.roi_envelope)
            
            generationParams = covariance_roi_l3s(pupil_mask, self.subap_diam, self.wavelength, 
                    self.tel_diam, self.n_subap, gs_dist, gs_pos, layer_alt.shape[0], layer_dist, 
                    L0, allMapPos_acrossMap, xy_separations_acrossMap, self.map_axis, self.roi_belowGround, 
                    self.roi_envelope, styc_method=self.styc_method, wind_profiling=False, lgs_track_present=lgs_track_present, 
                    offset_present=offset_present, fit_layer_alt=False, fit_lgs_track=False, fit_offset=False, fit_L0=fitting_L0)

            cov_roi = generationParams._make_covariance_roi_l3s_((layer_dist).astype('float'), 
                r0.astype('float'), L0.astype('float'), lgs_track=lgs_track, shwfs_shift=shwfs_shift.astype('float'), 
                shwfs_rot=shwfs_rot.astype('float'))

        self.timingStart = generationParams.timingStart

        if self.zeroSep_cov==False:
            cov_roi[self.zeroSep_locations] = 0.

        return cov_roi





    def process_input_matrix(self, cov_matrix):
        """Takes input covariance matrix and if specified, calculates new target array i.e. covariance map or map ROI.
        If L3S is being performed, the target array is calculated both with and without ground-layer mitigation
        
        Returns:
            ndarray: measured covariance matrix / map / map ROI.
            ndarray: measured covariance matrix / map / map ROI with ground-layer mitigated (if l3s_fit==True).
            ndarray: array to multiply matrix by to make 0 sub-aperture separations 0."""


        cov_meas = cov_matrix

        cov_aloft_meas = numpy.nan

        if self.remove_tt==True:
            cov_meas = numpy.matmul(numpy.matmul(self.remove_tt_matrix, cov_meas), self.remove_tt_matrix.T)

        if self.l3s_fit==True:
            cov_aloft_meas = numpy.matmul(numpy.matmul(self.l3s1_matrix, cov_meas), self.l3s1_matrix.T)

        if self.zeroSep_cov==False and self.target_array!='Covariance Map ROI':
            cov_meas[self.zeroSep_locations] = 0.
            if self.l3s_fit==True:
                cov_aloft_meas[self.zeroSep_locations] = 0.

        if self.target_array=='Covariance Matrix (No XY)':
            xx_yy_locations = track_matrix(2*self.n_subap_from_pupilMask[0]*self.n_wfs, self.n_subap_from_pupilMask[0], 
                numpy.array((1,1)))
            cov_meas[xx_yy_locations==0] = 0.
            if self.l3s_fit==True:
                cov_aloft_meas[xx_yy_locations==0] = 0.

        cov_meas_l3s3 = numpy.nan
        if self.target_array=='Covariance Map' or self.target_array=='Covariance Map ROI':
            cov_meas = covMap_fromMatrix(cov_meas, self.n_wfs, self.nx_subap, self.n_subap_from_pupilMask, 
                self.pupil_mask, self.map_axis, self.mm, self.mmc, self.md)
            if self.l3s_fit==True:
                cov_aloft_meas = covMap_fromMatrix(cov_aloft_meas, self.n_wfs, self.nx_subap, 
                    self.n_subap_from_pupilMask, self.pupil_mask, self.map_axis, self.mm, self.mmc, self.md)
            
            # stop
            if self.target_array=='Covariance Map ROI':
                cov_map = cov_meas.copy()

                cov_meas = roi_from_map(cov_map, self.gs_pos, self.pupil_mask, self.selector, 
                    self.roi_belowGround, self.roi_envelope)
                if self.zeroSep_cov==False:
                    cov_meas[self.zeroSep_locations] = 0.

                if self.l3s_fit==True:
                    cov_aloft_meas = roi_from_map(cov_aloft_meas, self.gs_pos, self.pupil_mask, self.selector, 
                        self.roi_belowGround, self.roi_envelope)
                    cov_aloft_meas[self.zeroSep_locations] = 0.

                    if self.fit_3==True:
                        cov_meas_l3s3 = roi_from_map(cov_map, self.gs_pos, self.pupil_mask, self.selector, 
                            self.roi_belowGround_l3s3, self.roi_envelope_l3s3)
                        if self.zeroSep_cov==False:
                            cov_meas_l3s3[self.zeroSep_locations_l3s3] = 0.

        return cov_meas, cov_meas_l3s3, cov_aloft_meas






    def process_matrix_cents(self):
        """Take SHWFS centroid measurements and calculates covariance matrix/map.
        If L3S is being performed, the covariance map ROI is calculated both with 
        and without ground-layer mitigation
        
        Returns:
            ndarray: measured covariance matrix/map.
            ndarray: measured covariance matrix/map with ground-layer mitigated (if l3s_fit==True).
            ndarray: array to multiply matrix by to make 0 sub-aperture separations 0."""

        begin_time = time.time()
        if self.remove_tt==False:
            cov_meas = cross_cov(self.shwfs_centroids)
            self.time_calc_cov = time.time() - begin_time
            print('############### COVARIANCE MATRIX CALCULATED ##############','\n')
            print('################### TIME TAKEN : '+"%6.4f" % (self.time_calc_cov)+' ###################','\n')

        if self.remove_tt==True:
            start_time = time.time()
            shwfs_cents_removedTT = numpy.matmul(numpy.matmul(self.remove_tt_matrix, 
                self.shwfs_centroids.T).T, self.remove_tt_matrix.T)
            cov_meas = cross_cov(shwfs_cents_removedTT)
            print('############ NO TT COVARIANCE MATRIX CALCULATED ###########','\n')
            print('################### TIME TAKEN : '+"%6.4f" % (time.time()-start_time)+' ###################')
        
        cov_aloft_meas = numpy.nan
        if self.l3s_fit==True:
            start_time = time.time()
            if self.remove_tt==True:
                shwfs_cents_aloft = numpy.matmul(numpy.matmul(self.l3s1_matrix, 
                    shwfs_cents_removedTT.T).T, self.l3s1_matrix.T)
                cov_aloft_meas = cross_cov(shwfs_cents_aloft)
            else:
                shwfs_cents_aloft = numpy.matmul(numpy.matmul(self.l3s1_matrix, 
                    self.shwfs_centroids.T).T, self.l3s1_matrix.T)
                cov_aloft_meas = cross_cov(shwfs_cents_aloft)
            print('############# L3S COVARIANCE MATRIX CALCULATED ############','\n')
            print('################### TIME TAKEN : '+"%6.4f" % (time.time()-start_time)+' ###################')

        self.time_calc_cov += time.time()-begin_time

        if self.zeroSep_cov == False:
            cov_meas[self.zeroSep_locations] = 0.
            if self.l3s_fit==True:
                cov_aloft_meas[self.zeroSep_locations] = 0.

        if self.target_array=='Covariance Matrix (No XY)':
            xx_yy_locations = track_matrix(2*self.n_subap[0]*self.n_wfs, self.n_subap[0], numpy.array((1,1)))
            cov_meas[xx_yy_locations==0] = 0.
            if self.l3s_fit==True:
                cov_aloft_meas[xx_yy_locations==0] = 0.

        if self.target_array=='Covariance Map':
            cov_meas = covMap_fromMatrix(cov_meas, self.n_wfs, self.nx_subap, self.n_subap, self.pupil_mask, 
                self.map_axis, self.mm, self.mmc, self.md)
            if self.l3s_fit==True:
                cov_aloft_meas = covMap_fromMatrix(cov_aloft_meas, self.n_wfs, self.nx_subap, self.n_subap, 
                    self.pupil_mask, self.map_axis, self.mm, self.mmc, self.md)

        return cov_meas, numpy.nan, cov_aloft_meas




    def process_roi_cents(self):
        """Takes SHWFS centroid measurements and calculates covariance map ROI. 
        If L3S is being performed, the covariance map ROI is calculated both with and without ground-layer mitigation.

        Returns:
            ndarray: measured covariance map ROI.
            ndarray: measured covariance map ROI with ground-layer mitigated (if l3s_fit==True).
            ndarray: array to multiply ROI by to make 0 sub-aperture separations 0."""
        
        begin_time = time.time()
        if self.remove_tt==False:
            cov_roi, calc_time = calculate_roi_covariance(self.shwfs_centroids, 
                self.gs_pos, self.pupil_mask, self.tel_diam, self.roi_belowGround, 
                self.roi_envelope, self.map_axis, self.mapping_type)
            print('################ COVARIANCE ROI CALCULATED ################','\n')
            print('################### TIME TAKEN : '+"%6.4f" % (calc_time)+' ###################','\n')

        if self.remove_tt==True:
            shwfs_cents_removedTT = numpy.matmul(numpy.matmul(self.remove_tt_matrix, 
                self.shwfs_centroids.T).T, self.remove_tt_matrix.T)
            cov_roi, calc_time = calculate_roi_covariance(shwfs_cents_removedTT, 
                self.gs_pos, self.pupil_mask, self.tel_diam, self.roi_belowGround, 
                self.roi_envelope, self.map_axis, self.mapping_type)
            print('########## NO TT COVARIANCE ROI CALCULATED ##########','\n')
            print('################### TIME TAKEN : '+"%6.4f" % (calc_time)+' ###################')

        cov_roi_aloft = numpy.nan
        cov_roi_l3s3 = numpy.nan
        if self.l3s_fit==True:
            if self.remove_tt==True:
                shwfs_cents_aloft = numpy.matmul(numpy.matmul(self.l3s1_matrix, 
                    shwfs_cents_removedTT.T).T, self.l3s1_matrix.T)
                cov_roi_aloft, calc_time = calculate_roi_covariance(shwfs_cents_aloft, 
                    self.gs_pos, self.pupil_mask, self.tel_diam, self.roi_belowGround, 
                    self.roi_envelope, self.map_axis, self.mapping_type)
            else:
                shwfs_cents_aloft = numpy.matmul(numpy.matmul(self.l3s1_matrix, 
                    self.shwfs_centroids.T).T, self.l3s1_matrix.T)
                cov_roi_aloft, calc_time = calculate_roi_covariance(shwfs_cents_aloft, 
                    self.gs_pos, self.pupil_mask, self.tel_diam, self.roi_belowGround, 
                    self.roi_envelope, self.map_axis, self.mapping_type)
            print('############# L3S.1 COVARIANCE ROI CALCULATED #############','\n')
            print('################### TIME TAKEN : '+"%6.4f" % (calc_time)+' ###################', '\n')

            if self.fit_3==True:
                if self.remove_tt==True:
                    cov_roi_l3s3, calc_time2 = calculate_roi_covariance(shwfs_cents_removedTT, 
                        self.gs_pos, self.pupil_mask, self.tel_diam, self.roi_belowGround_l3s3, 
                        self.roi_envelope_l3s3, self.map_axis, self.mapping_type)
                else:
                    cov_roi_l3s3, calc_time2 = calculate_roi_covariance(self.shwfs_centroids, 
                        self.gs_pos, self.pupil_mask, self.tel_diam, self.roi_belowGround_l3s3, 
                        self.roi_envelope_l3s3, self.map_axis, self.mapping_type)
                print('############# L3S.3 COVARIANCE ROI CALCULATED #############','\n')
                print('################### TIME TAKEN : '+"%6.4f" % (calc_time)+' ###################', '\n')

        if self.zeroSep_cov == False:
            cov_roi[self.zeroSep_locations] = 0.

            if self.l3s_fit==True:
                cov_roi_aloft[self.zeroSep_locations] = 0.

                if self.fit_3==True:
                    cov_roi_l3s3[self.zeroSep_locations_l3s3] = 0.

        self.time_calc_cov = time.time() - begin_time

        print('################## CALCULATIONS COMPLETE ##################','\n')
        print('################### TIME TAKEN : '+"%6.4f" % (self.time_calc_cov)+' ###################')


        return cov_roi, cov_roi_l3s3, cov_roi_aloft







    def amend_results(self):
        """Convert units of fitted results."""

        self.Cn2 = numpy.array([numpy.nan]*self.n_layer)
        self.Cn2[:self.observable_bins] = calc_Cn2(self.r0[:self.observable_bins], self.wavelength[0])
        self.Cn2 /= self.air_mass
        self.Cn2[:self.observable_bins][self.Cn2[:self.observable_bins]<=self.cn2_noiseFloor] = self.cn2_noiseFloor
        self.r0[:self.observable_bins] = calc_r0(self.Cn2[:self.observable_bins], self.wavelength[0])
        # self.shwfs_shift -= self.shwfs_shift[0]





    def import_tools(self, air_mass, tas, shwfs_centroids):
        """Converts tas into GS positons [arcsecs] (only if data_type=='canary' - simulated tas is 
        assumed to equal GS posistion). SHWFS centroid positions have units converted to arcsecs 
        by pix_arc (only if data_type!='canary'). Maximum altitude range is also printed and 
        observable_bins set (if force_altRange==True).
        
        Parameters:
            air_mass (float): air mass of observation (1/cos(theta)).
            tas (ndarray): position of targetting acquisition system's SHWFS pick-offs.
            shwfs_centroids (ndarray): SHWFS centroid measurements."""

        self.shwfs_centroids = numpy.nan
        self.air_mass = float(air_mass)
        self.gs_pos = tas.astype('float')
        self.gs_dist = self.gs_alt.copy() * self.air_mass

        if self.gs_pos.shape[0]!=self.n_wfs:
            raise Exception("Check number of GSs vs SHWFS parameters.")
        
        if self.data_type=='canary':
            self.gs_pos *= self.plate_scale
        
        if self.input_matrix==False:
            self.shwfs_centroids = numpy.zeros((shwfs_centroids.shape[0], 2 * numpy.sum(self.n_subap_from_pupilMask)))
            for wfs_n in range(self.n_wfs):
                step = int(2*self.n_subap_from_pupilMask[wfs_n])
                self.shwfs_centroids[:, (wfs_n*step):((wfs_n+1)*step)] =  shwfs_centroids[:, 
                    (wfs_n*step)+self.cent_start:((wfs_n+1)*step)+self.cent_start] * self.pix_arc[wfs_n]


        #solve h_max for the observation
        # print('Range of Maximum Observable Altitudes (km):')
        print('\n'+'###########################################################','\n')
        print('######################## h_max (km) #######################','\n')

        self.observable_bins = self.n_layer		
        self.min_alt, self.max_alt = calc_hmax_NGSandLGS(self.pupil_mask, self.n_subap, self.gs_pos, self.gs_dist, self.tel_diam, self.air_mass)

        #if force_altRange==True, only fit layers below h_max
        # if self.force_altRange == True:
        if self.gs_dist[0]==0:
            for i in range(self.n_layer):
                if self.layer_alt[i]!=0:
                    if (self.max_alt/self.layer_alt[i]) < 1:
                        self.observable_bins = i
                        break
        else:
            #account for lgs cone effect
            # gs_alt_zen = self.gs_dist[0].copy() / air_mass
            # self.max_alt = (self.max_alt*gs_alt_zen)/(self.max_alt+gs_alt_zen)
            for i in range(self.n_layer):
                if self.layer_alt[i]!=0:
                    if (self.max_alt/self.layer_alt[i]) < 1:
                        self.observable_bins = i
                        break
        
        if self.force_altRange==False:
            self.observable_bins = self.n_layer		

        print('###################### '+"%.3f" % (self.min_alt/1000.)+' -> '+"%.3f" % (self.max_alt/1000.)+' ####################')

        if self.target_array=='Covariance Map ROI':
            #These imported tools are the key to calculating the covariance map ROI and its 
            # analytically generated model during fitting
            self.onesMat, self.wfsMat_1, self.wfsMat_2, self.allMapPos, self.selector, self.xy_separations = roi_referenceArrays(
                self.pupil_mask, self.gs_pos, self.tel_diam, self.roi_belowGround, self.roi_envelope)
            if self.l3s_fit==True:
                if self.fit_3==True:
                    self.onesMat_l3s3, self.wfsMat_1_l3s3, self.wfsMat_2_l3s3, self.allMapPos_l3s3, self.selector, self.xy_separations_l3s3 = roi_referenceArrays(
                        self.pupil_mask, self.gs_pos, self.tel_diam, self.roi_belowGround_l3s3, self.roi_envelope_l3s3)
                
        else:
            self.allMapPos = self.selector = self.xy_separations = self.roi_belowGround = self.roi_envelope = 1.
            self.allMapPos_l3s3 = self.selector = self.xy_separations_l3s3 = self.roi_belowGround_l3s3 = self.roi_envelope_l3s3 = 1.