import numpy
import time
import math
import itertools
from scipy.special import comb
from matplotlib import pyplot; pyplot.ion()
import capt.misc_functions.matplotlib_format 
from capt.misc_functions.make_pupil_mask import make_pupil_mask
from capt.misc_functions.mapping_matrix import get_mappingMatrix
from capt.roi_functions.roi_referenceArrays import roi_referenceArrays

class covariance_roi_l3s(object):
		
	def __init__(self, pupil_mask, subap_diam, wavelength, tel_diam, n_subap, gs_alt, gs_pos, 
			n_layer, layer_alt, L0, allMapPos_acrossMap, xy_separations_acrossMap, roi_axis, 
			roi_belowGround, roi_envelope, styc_method=True, wind_profiling=False, 
			lgs_track_present=False, offset_present=False, fit_layer_alt=False, 
			fit_lgs_track=False, fit_offset=False, fit_L0=False):
		
		"""Configuration used to generate a covariance map ROI with the ground-layer mitigated.
		
		Parameters:
			pupil_mask (ndarray): mask of SHWFS sub-apertures within the telescope's pupil.
			subap_diam (float): diameter of SHWFS sub-aperture in telescope's pupil.
			wavelength (ndarray): SHWFS centroid wavelengh (nm).
			tel_diam (float): diameter of telescope pupil.
			gs_alt (ndarray): GS altitude. 0 for NGS (LGS not tested!).
			gs_pos (ndarray): GS asterism in telescope FoV.
			xy_separations_acrossMap (ndarray): x and y sub-aperture separation distance for ROI that has length cov_map.shape[0].
			n_layer (int): number of turbulent layers.
			layer_alt (ndarray): altitudes of turbulent layers (m).
			wind_profiling (bool): determines whether generated covariance map ROI is to be used for wind profiling.
			roi_belowGround (int): number of sub-aperture separations the ROI expresses 'below-ground'.
			roi_envelope (int): number of sub-aperture separations the ROI expresses either side of stellar separation.
			fit_L0 (bool): determines whether generated covariance map ROI is to be used for L0 profiling.
			L0 (ndarray): L0 profile (m).
			roi_axis (str): in which axis to express ROI ('x', 'y', 'x+y' or 'x and y')
			styc_method (bool): use styc method of analytically generating covariance."""

		#load ROI parameters
		self.pupil_mask = pupil_mask
		self.wind_profiling = wind_profiling
		self.aligned_xy_separations = xy_separations_acrossMap
		self.roi_envelope = roi_envelope
		selector = numpy.array((range(gs_pos.shape[0])))
		self.selector = numpy.array((list(itertools.combinations(selector, 2))))
		self.n_wfs = gs_pos.shape[0]
		self.subap_diam = subap_diam
		self.wavelength = numpy.append(wavelength, wavelength[0])
		self.tel_diam = tel_diam
		self.n_subap = n_subap
		self.nx_subap = pupil_mask.shape[0]
		self.gs_alt = numpy.append(gs_alt, gs_alt[0])
		self.gs_pos = gs_pos[:,::-1] # Swap X and Y GS positions to agree with legacy code
		self.L0 = L0
		self.n_layer = n_layer
		self.radSqaured_to_arcsecSqaured = ((180./numpy.pi) * 3600)**2  
		self.combs = int(comb(gs_pos.shape[0], 2, exact=True))
		self.roi_width = 1 + (2*self.roi_envelope)
		self.map_length = int((self.nx_subap * 2) - 1)
		self.subap_layer_positions = numpy.zeros((self.combs, n_layer, 2, 
			self.combs+1, self.roi_width, self.map_length, 2))
		self.subap_layer_diameters = numpy.ones((self.n_layer, 2, self.combs+1)) * self.subap_diam[0]
		self.translation = numpy.zeros((n_layer, 2, self.combs, 2))
		self.selector = numpy.vstack((self.selector, numpy.array([0, 0])))
		self.zeroBelowGround = pupil_mask.shape[0]-1
		self.roi_belowGround = roi_belowGround
		self.roi_axis = roi_axis
		self.styc_method = styc_method
		self.fit_L0 = fit_L0
		self.fit_layer_alt = fit_layer_alt
		self.offset_present = offset_present
		self.fit_offset = fit_offset
		self.offset_set = False
		self.map_centre_width = int(-1 + ((self.roi_width+1)/2.))
		self.map_centre_belowGround = int((self.map_length-1)/2.)
		self.scale_factor = numpy.zeros((n_layer, self.combs))

		#self.length_mult sets the number of roi axes generated.
		self.length_mult = 2
		if self.roi_axis != 'x and y' and self.roi_axis != 'x+y':
			self.length_mult = 1

		#build transformation matrices for roi
		self.covariance_slice_transformMatrix1 = numpy.ones((self.combs, self.n_wfs * self.roi_width, 
			self.map_length * self.n_wfs * self.length_mult)) * - 1./self.n_wfs
		self.covariance_slice_transformMatrix2 = numpy.ones((self.combs, self.roi_width, 
			self.map_length * self.n_wfs * self.length_mult)) * - 1./self.n_wfs
		count1 = 1
		count2 = 0
		startCount = 1
		for i in range(self.combs):
			self.covariance_slice_transformMatrix1[i, count1*self.roi_width: 
				count1*self.roi_width + self.roi_width] = (1 - (1./self.n_wfs))
			self.covariance_slice_transformMatrix2[i, :, count2*self.map_length*self.length_mult:
				(count2 * self.map_length * self.length_mult) + 
				(self.map_length * self.length_mult)] = (1 - (1./self.n_wfs))
			count1 += 1
			if count1 == self.n_wfs:
				startCount += 1
				count1 = startCount
				count2 += 1

		self.lgs_track_present = lgs_track_present
		self.fit_lgs_track = fit_lgs_track
		self.lgs_track_set = False


		#generate arrays to be filled with analytically generated covariance
		if self.roi_axis!='y':
			if wind_profiling==False:
				self.cov_xx = numpy.zeros((1, self.combs, self.combs+1, 
					self.roi_width, self.map_length, n_layer)).astype('float64')
			if wind_profiling==True:
				self.cov_xx = numpy.zeros((2, self.combs, self.combs+1, 
					self.roi_width, self.map_length, n_layer)).astype('float64')

		if self.roi_axis!='x':
			if wind_profiling==False:
				self.cov_yy = numpy.zeros((1, self.combs, self.combs+1, 
					self.roi_width, self.map_length, n_layer)).astype('float64')
			if wind_profiling==True:
				self.cov_yy = numpy.zeros((2, self.combs, self.combs+1, 
					self.roi_width, self.map_length, n_layer)).astype('float64')


		if fit_lgs_track==True or lgs_track_present==True:
			roi_xyseps = xy_separations_acrossMap.copy()[:, :, pupil_mask.shape[0]-1-roi_belowGround:]
			self.lgs_trackMatrix_locs = lgs_trackMatrix_locs(roi_xyseps, roi_axis)
			self.lgs_track = numpy.zeros(self.lgs_trackMatrix_locs.shape).astype('float64')

		if self.offset_present==False and self.fit_offset==False:
			self.compute_cov_offset = False
			self.meanDenominator = numpy.array([1.]*(self.combs+1))
			self.xy_separations = xy_separations_acrossMap
			self.auto_xy_separations = self.xy_separations[0]

		if self.offset_present==True or self.fit_offset==True:
			self.compute_cov_offset = True
			self.allMapPos = allMapPos_acrossMap
			self.subap_layer_positions_atSeparations()

			x_seps = numpy.array([self.aligned_xy_separations.T[0]]*self.n_subap[0])
			y_seps = numpy.array([self.aligned_xy_separations.T[1]]*self.n_subap[0])
			self.xy_seps = numpy.stack((x_seps, y_seps)).T

			self.xy_separations=numpy.zeros((self.combs, self.combs, self.roi_width, 
				self.map_length, self.subap_sep_positions.shape[3], 2))

			self.subap_positions_wfsAlignment = numpy.zeros((self.combs, self.combs, 2, 
				self.roi_width, self.map_length, self.subap_sep_positions.shape[3], 2)).astype("float64")


		self.covariance_slice_matrix = numpy.zeros((self.combs, self.n_wfs * self.roi_width, 
			self.map_length * self.n_wfs * self.length_mult, n_layer))



		self.timingStart = time.time()

		#if L0 and wind are not being fitted, the 2D response function profile (over the entire ROI) only has to be generated once (for 
		# each layer). If fitting r0, response functions are then multiplied by some factor before being summed together i.e 
		# the shape of each response function doesn't change.  
				
		if self.fit_layer_alt==False:
			self.subap_parameters(layer_alt)
		
		if self.fit_L0 == self.offset_present == self.fit_offset == self.wind_profiling == False:
			if self.styc_method == True:
				self.computeStyc(L0)
			if self.styc_method == False:
				self.computeButt(L0)    
			self.fixedLayerParameters()

		# stopinside





	def _make_covariance_roi_l3s_(self, layer_alt, r0, L0, lgs_track=False, shwfs_shift=False, 
		shwfs_rot=False, delta_xSep=False, delta_ySep=False):
		"""Master node for generating covariance map ROI for L3S.1.
		
		Parameters:
			r0 (ndarray): r0 profile (m).
			L0 (ndarray): L0 profile (m).
			shwfs_shift (ndarray): SHWFS shift in x and y (m). Note: not yet implemented!
			shwfs_rot(ndarray): SHWFS rotation. Note: not yet implemented!
			delta_xSep (ndarray): shift x separation in covariance map ROI (developed for wind profiling).
			delta_ySep (ndarray): shift y separation in covariance map ROI (developed for wind profiling).
			
		Return:
			ndarray: analytically generated covariance map ROI with ground-layer mitigated."""
		
		if self.fit_L0==True or self.offset_present==True or self.fit_offset==True or self.wind_profiling==True or self.fit_layer_alt==True:
			
			if self.fit_layer_alt==True:
				self.subap_parameters(layer_alt)

			if self.wind_profiling==True:
				self.delta_xSep = delta_xSep.copy()
				self.delta_ySep = delta_ySep.copy()

			if self.fit_offset==True:
				self.subap_wfsAlignment(shwfs_shift, shwfs_rot)

			else:
				if self.offset_present==True and self.offset_set==False:
					self.subap_wfsAlignment(shwfs_shift, shwfs_rot)                    

			if self.offset_set==False:
				if self.styc_method == True:
					ct = time.time()
					self.computeStyc(L0)
					# print(' - Comp: {}'.format(time.time() - ct))

				self.fixedLayerParameters()
			
				if self.offset_present==True and self.fit_offset==self.wind_profiling==self.fit_L0==self.fit_layer_alt==False:
					self.offset_set=True

		#calculate covariance map ROI
		r0_scale2 = (L0/r0)**(5./3)
		roi_l3s = (self.covariance_slice_matrix * r0_scale2).sum(3) * self.radSqaured_to_arcsecSqaured

		#perform first stage of ground-layer mitigation
		roi_l3s =  (roi_l3s * self.covariance_slice_transformMatrix1).reshape(self.combs, 
			self.n_wfs, self.roi_width, self.map_length * self.length_mult * self.n_wfs).sum(1)

		# firstTransformCollapse =  (self.covariance_slice_array * -1./self.n_wfs).reshape(self.combs, 
		# 	self.n_wfs, self.roi_width, self.map_length * self.length_mult * self.n_wfs).sum(1)

		#perform second stage of ground-layer mitigation
		roi_l3s = (roi_l3s * self.covariance_slice_transformMatrix2).reshape((self.combs, 
			self.roi_width, self.n_wfs, self.map_length * self.length_mult)).sum(2)
		#covariance map ROI with ground-layer mitigated (with length = self.map_length).
		roi_l3s = roi_l3s.reshape(self.combs * self.roi_width, self.map_length * self.length_mult)

		#convert roi_l3s to its final specifications (axes and ROI length)
		if self.roi_axis=='x':
			roi_l3s = roi_l3s[:, self.zeroBelowGround-self.roi_belowGround:]
		if self.roi_axis=='y':
			roi_l3s = roi_l3s[:, self.zeroBelowGround-self.roi_belowGround:]  
		if self.roi_axis=='x+y':
			roi_l3s = (roi_l3s[:, self.zeroBelowGround-self.roi_belowGround:self.map_length] 
				+ roi_l3s[:, self.map_length + (self.zeroBelowGround-self.roi_belowGround):])/2.
		if self.roi_axis=='x and y':
			roi_l3s = numpy.hstack((roi_l3s[:, self.zeroBelowGround-self.roi_belowGround:self.map_length], 
				roi_l3s[:, self.map_length + (self.zeroBelowGround-self.roi_belowGround):]))


		if self.lgs_track_present==True or self.fit_lgs_track==True:
			if self.lgs_track_set==False:
				self.set_lgs_tracking_values(lgs_track)
				if self.fit_lgs_track!=True:
					self.lgs_track_set = True

			roi_l3s += self.lgs_track

		return roi_l3s





	def set_lgs_tracking_values(self, lgs_track):
		"""Generates tracking matirx - linear additon to covariance map ROI.

		Parameters:
			track (ndarray): 1s for locations of ROI within map"""

		if self.roi_axis=='x':
			for i in range(self.combs):
				self.lgs_track[self.lgs_trackMatrix_locs==i+1] = lgs_track[i,0]

		if self.roi_axis=='y':
			for i in range(self.combs):
				self.lgs_track[self.lgs_trackMatrix_locs==i+1] = lgs_track[i,1]

		if self.roi_axis=='x+y':
			for i in range(self.combs):
				self.lgs_track[self.lgs_trackMatrix_locs==i+1] = (lgs_track[i,0]+lgs_track[i,1])/2.

		if self.roi_axis=='x and y':
			counter = 1
			for i in range(self.combs):
				for j in range(2):
					self.lgs_track[self.lgs_trackMatrix_locs==counter] = lgs_track[i, j]
					counter += 1






	def subap_parameters(self, layer_alt):
		"""Calculate initial parameters that are fixed i.e. translation of sub-aperture positions with altitude."""

		for wfs_n in range(2):
			for comb in range(self.combs):

				gs_pos = self.gs_pos[self.selector[comb, wfs_n]]

				# Scale for LGS
				if self.gs_alt[self.selector[comb, wfs_n]] != 0:
					# print("Its a LGS!")
					self.scale_factor[:, comb] = (1 - layer_alt/self.gs_alt[self.selector[comb,wfs_n]])
				else:
					self.scale_factor[:, comb] = 1.

				# translate due to GS position
				gs_pos_rad = numpy.array(gs_pos) * (numpy.pi/180.) * (1./3600.)

				# print("GS Positions: {} rad".format(gs_pos_rad))
				self.translation[:, wfs_n, comb] = gs_pos_rad
				self.subap_layer_diameters[:, wfs_n, comb] = self.subap_diam[self.selector[comb, wfs_n]] * self.scale_factor[:, comb]

		self.translation = (self.translation.T * layer_alt).T





	def subap_layer_positions_atSeparations(self):
		"""Calculates the position of every sub-aperture pairing (within the covariance map ROI) in the telescope's pupil. 
		The number of sub-aperture pairings at each covariance map ROI data point is also calculated so that mean 
		covariance can be found."""


		covMapDim = (self.pupil_mask.shape[0] * 2) - 1
		onesMat = numpy.ones((int(self.pupil_mask.sum()), int(self.pupil_mask.sum())))
		wfs_subap_pos = (numpy.array(numpy.where(self.pupil_mask == 1)).T * self.tel_diam/self.pupil_mask.shape[0])
		onesMM, onesMMc, mapDensity = get_mappingMatrix(self.pupil_mask, onesMat)

		xPosMM, xMMc, d = get_mappingMatrix(self.pupil_mask, onesMat*wfs_subap_pos.T[0])
		yPosMM, yMMc, d = get_mappingMatrix(self.pupil_mask, onesMat*wfs_subap_pos.T[1])

		xPosMM[onesMM==0] = numpy.nan
		yPosMM[onesMM==0] = numpy.nan
		ySeps = numpy.ones((self.combs, self.roi_width, self.map_length, self.nx_subap**2)) * numpy.nan
		xSeps = numpy.ones((self.combs, self.roi_width, self.map_length, self.nx_subap**2)) * numpy.nan
		self.meanDenominator = numpy.zeros((self.combs, self.roi_width, self.map_length))
		self.subap_sep_positions = numpy.ones((self.combs, self.roi_width, self.map_length, self.nx_subap**2, 2))
		self.allMapPos[self.allMapPos>=covMapDim] = 0.

		for comb in range(self.combs):
			mmLocations = (covMapDim * self.allMapPos[comb,:,:,0]) + self.allMapPos[comb,:,:,1]
			self.meanDenominator[comb] = onesMM[:,mmLocations].sum(0)
			ySepsMM = -yPosMM[:,mmLocations]
			xSepsMM = -xPosMM[:,mmLocations]

			for env in range(self.roi_width):
				for l in range(self.map_length):

					ySeps[comb, env,l] = ySepsMM[:, env, l]
					xSeps[comb, env,l] = xSepsMM[:, env, l]


			self.subap_sep_positions[comb,:,:,:,0] = xSeps[comb]
			self.subap_sep_positions[comb,:,:,:,1] = ySeps[comb]

		self.meanDenominator[self.meanDenominator==0] = 1.
		self.meanDenominator = numpy.stack([self.meanDenominator]*self.n_layer, 3)



	def subap_wfsAlignment(self, shwfs_shift, shwfs_rot):
		"""Calculates x and y sub-aperture separations under some SHWFS shift and/or rotation.
		
		Parameters:
			shwfs_shift (ndarray): SHWFS shift in x and y (m).
			shwfs_rot (ndarray): SHWFS rotation."""

		for gs_comb in range(self.combs):
			
			for comb in range(self.combs):

				for wfs_i in range(2):

					theta = (shwfs_rot[self.selector[comb, [1,0][wfs_i]]]) * numpy.pi/180.

					xtp = self.subap_sep_positions[gs_comb, :, :, :, 1].copy()
					ytp = self.subap_sep_positions[gs_comb, :, :, :, 0].copy()

					uu = xtp * numpy.cos(theta) - ytp * numpy.sin(theta)
					vv = xtp * numpy.sin(theta) + ytp * numpy.cos(theta)

					self.subap_positions_wfsAlignment[gs_comb, comb,wfs_i,:,
						:,:,1] = uu - shwfs_shift[self.selector[comb, wfs_i],0]
					self.subap_positions_wfsAlignment[gs_comb, comb,wfs_i,:,
						:,:,0] = vv - shwfs_shift[self.selector[comb, wfs_i],1]

				self.xy_separations[gs_comb, comb] = -(self.subap_positions_wfsAlignment[gs_comb, comb,0,
					:,:,:] - self.subap_positions_wfsAlignment[gs_comb, comb, 1, self.map_centre_width,self.map_centre_belowGround])




	def fixedLayerParameters(self):
		"""Creates covariance map ROI matrix where each calculated ROI (e.g. self.cov_xx) has each 
		GS combinations positioned appropriately."""

		self.covariance_slice_matrix *= 0.

		for gs_comb in range(self.combs):
			
			#fill covariance regions
			marker = self.roi_width * self.n_wfs
			count = 1
			u = 0
			v = self.map_length * self.length_mult
			r = self.roi_width
			s = self.roi_width * 2

			rr = 0
			ss = self.roi_width
			uu = (self.map_length * self.length_mult)
			vv = (self.map_length * self.length_mult) * 2

			for comb in range(self.combs+1):

				r0_scale1 = ((self.wavelength[self.selector[comb,0]] * self.wavelength[self.selector[comb,1]])
					/ (8 * numpy.pi**2 * self.subap_diam[self.selector[comb, 0]]
					* self.subap_diam[self.selector[comb, 1]] ))


				if self.roi_axis=='x':

					if comb == self.combs:
						for i in range(self.n_wfs):
							self.covariance_slice_matrix[gs_comb, i*self.roi_width: 
								(i+1)*self.roi_width, i*self.map_length*self.length_mult: 
								(i+1)*self.map_length*self.length_mult] += self.cov_xx[0, gs_comb, comb] * r0_scale1
					else:
						self.covariance_slice_matrix[gs_comb, r:s, u:v] += self.cov_xx[0, gs_comb, comb] * r0_scale1
						if self.wind_profiling==True:
							self.covariance_slice_matrix[gs_comb, rr:ss, uu:vv] += self.cov_xx[1, gs_comb, comb] * r0_scale1

				if self.roi_axis=='y':

					if comb == self.combs:
						for i in range(self.n_wfs):
							self.covariance_slice_matrix[gs_comb, i*self.roi_width: 
								(i+1)*self.roi_width, i*self.map_length*self.length_mult: 
								(i+1)*self.map_length*self.length_mult] += self.cov_yy[0, gs_comb, comb] * r0_scale1
					else:
						self.covariance_slice_matrix[gs_comb, r:s, u:v] += self.cov_yy[0, gs_comb, comb] * r0_scale1
						if self.wind_profiling==True:
							self.covariance_slice_matrix[gs_comb, rr:ss, uu:vv] += self.cov_yy[1, gs_comb, comb] * r0_scale1


				if self.roi_axis=='x and y' or self.roi_axis=='x+y':
				
					#fill auto-covariance regions
					if comb == self.combs:
						for i in range(self.n_wfs):
							self.covariance_slice_matrix[gs_comb, i*self.roi_width: 
								(i+1)*self.roi_width, i*self.map_length*self.length_mult: 
								(i+1)*self.map_length*self.length_mult] += numpy.hstack((self.cov_xx[0, gs_comb, comb], self.cov_yy[0, gs_comb, comb])) * r0_scale1
				
					else:
						self.covariance_slice_matrix[gs_comb, r:s, u:v] += numpy.hstack((self.cov_xx[0, gs_comb, comb], self.cov_yy[0, gs_comb, comb])) * r0_scale1
						if self.wind_profiling==True:
							self.covariance_slice_matrix[gs_comb, rr:ss, uu:vv] += numpy.hstack((self.cov_xx[1, gs_comb, comb], self.cov_yy[1, gs_comb, comb])) * r0_scale1

				if comb!=self.combs:

					r += self.roi_width
					s += self.roi_width

					uu += self.map_length * self.length_mult
					vv += self.map_length * self.length_mult

					if r == marker:
						count += 1
						u += self.map_length * self.length_mult
						v += self.map_length * self.length_mult
						r = (self.roi_width * count)
						s = (self.roi_width * count) + self.roi_width

						rr += self.roi_width
						ss += self.roi_width
						uu = (self.map_length * self.length_mult * count)
						vv = (self.map_length * self.length_mult * count) + (self.map_length * self.length_mult)  

			if self.wind_profiling!=True:
				self.covariance_slice_matrix[gs_comb] = mirror_covariance_roi(
					self.covariance_slice_matrix[gs_comb], self.n_wfs, self.roi_axis)
		


	def computeStyc(self, L0):
		"""Uses styc method for analytically generating covariance.
		
		Parameters:
			L0 (ndarray): L0 profile (m)."""


		for gs_comb in range(self.combs):
			for comb in range(self.combs+1):
				
				if comb==self.combs:
					xy_seps = numpy.stack([self.aligned_xy_separations[gs_comb]]*self.n_layer, 2)
					xy_seps *= numpy.stack([self.scale_factor[:, gs_comb]]*2, 1)

					mean_denom = 1.
					offset_present = False
					translation = 0.
				
				else:
					translation = self.translation[:, 1, comb] - self.translation[:, 0, comb]
					if self.compute_cov_offset==True:
						xy_seps = numpy.stack([self.xy_separations[gs_comb, comb]]*self.n_layer, 3)
						xy_seps *= numpy.stack([self.scale_factor[:, comb]]*2, 1)

						offset_present = True
						mean_denom = self.meanDenominator[gs_comb]
					else:
						xy_seps = numpy.stack([self.xy_separations[gs_comb]]*self.n_layer, 2)
						xy_seps *= numpy.stack([self.scale_factor[:, comb]]*2, 1)

						offset_present = False
						mean_denom = 1.

				#if wind profiling, shift x and y SHWFS sub-aperture separations
				if self.wind_profiling==True:
					if self.offset_present==True or self.fit_offset==True:
						if comb!=self.combs:
							xy_seps[:,:,:,:,0] += self.delta_xSep
							xy_seps[:,:,:,:,1] += self.delta_ySep
							xy_seps = numpy.stack((xy_seps, -xy_seps))

						else:
							xy_seps[:,:,:,0] += self.delta_xSep
							xy_seps[:,:,:,1] += self.delta_ySep
							xy_seps = numpy.stack((xy_seps, -xy_seps))

					if self.offset_present==False and self.fit_offset==False:
						xy_seps[:,:,:,0] += self.delta_xSep
						xy_seps[:,:,:,1] += self.delta_ySep
						xy_seps = numpy.stack((xy_seps, -xy_seps))
				else:
					xy_seps = numpy.expand_dims(xy_seps, axis=0)

				#calculate required xx covariance
				if self.roi_axis != 'y':

					self.cov_xx[:, gs_comb, comb] = compute_covariance_xx(xy_seps, 
						self.subap_layer_diameters[:, 0, gs_comb]/2., 
						self.subap_layer_diameters[:, 1, gs_comb]/2., translation, 
						L0, offset_present)/mean_denom

				#calculate required yy covariance
				if self.roi_axis != 'x':
					self.cov_yy[:, gs_comb, comb] = compute_covariance_yy(xy_seps, 
						self.subap_layer_diameters[:, 0, gs_comb]/2., 
						self.subap_layer_diameters[:, 1, gs_comb]/2., translation, 
						L0, offset_present)/mean_denom




	def computeButt(self, L0):
		"""Uses T. Butterley method for analytically generating covariance.

		Parameters:
			L0 (ndarray): L0 profile (m)."""

		for gs_comb in range(self.combs):
			for layer_n in range(self.n_layer):
				for comb in range(self.combs+1):
					for env in range(self.roi_width):
						wfs1_diam = self.subap_layer_diameters[gs_comb, layer_n, 0, comb, env]
						wfs2_diam = self.subap_layer_diameters[gs_comb, layer_n, 1, comb, env]
						n_subaps1 = self.n_subap[self.selector[comb][0]]
						n_subaps2 = self.n_subap[self.selector[comb][1]]
						xy_sep = self.xy_separations[gs_comb, layer_n, comb, env]

						# numerical integration code doesn't support different subap sizes
						assert abs(wfs1_diam - wfs2_diam) < 1.e-10              
						# numerical integration code doesn't support different num. subaps
						assert abs(n_subaps1 - n_subaps2) < 1.e-10              
						maxDelta=0
						for i in range(xy_sep.shape[0]):
							if math.isnan(xy_sep[i,0]) != True:
								if numpy.abs(xy_sep[i]).max()>maxDelta:
									maxDelta = int(numpy.abs(xy_sep[i]).max()/wfs1_diam)+1
						
						sf_dx = wfs1_diam/100.
						sf_n = int(4*maxDelta*(wfs1_diam/sf_dx))
						sf = structure_function_vk(numpy.arange(sf_n)*sf_dx, L0[layer_n])
						sf[0] = 0.    #get rid of nan
						nSamp = 8    #hard-wired parameter
						sf_dx = wfs1_diam/100.
						self.cov_xx[gs_comb, layer_n, comb, env], self.cov_yy[gs_comb, layer_n, comb, 
							env] = compute_ztilt_covariances(n_subaps1, xy_sep, sf, sf_dx, nSamp, wfs1_diam)






def lgs_trackMatrix_locs(xy_separations, map_axis):
	combs = xy_separations.shape[0]
	width = xy_separations.shape[1]
	length = xy_separations.shape[2]
	
	if map_axis=='x and y':
		no_axes = 2
	else:
		no_axes = 1

	lgs_trackMatrix = numpy.zeros((combs*width, xy_separations.shape[2]*no_axes))
	tracker = 1
	for i in range(combs):
		for ax in range(no_axes):
			xy_separations[i, :, :, ax][numpy.isnan(xy_separations[i, :, :, ax])==False] = tracker
			lgs_trackMatrix[i*width: (i+1)*width, ax*length: (ax+1)*length] = xy_separations[i, :, :, ax]
			tracker += 1
	
	lgs_trackMatrix[numpy.isnan(lgs_trackMatrix)==True] = 0.
	
	return lgs_trackMatrix.astype(int)




def mirror_covariance_roi(cov_roi_mat, n_wfs, roi_axis):
    step_length = int(cov_roi_mat.shape[1]/float(n_wfs))
    step_width = int(cov_roi_mat.shape[0]/float(n_wfs))
    if roi_axis=='x and y' or 'x+y':
        roi_width = int(step_length/2.)

    n1 = 0
    mm1 = 0
    for n in range(n_wfs):
        m1 = 0
        nn1 = 0

        for m in range(n + 1):
            if n != m:
                n2 = n1 + step_width
                m2 = m1 + step_length

                nn2 = nn1 + step_width
                mm2 = mm1 + step_length

                if roi_axis=='x and y' or roi_axis=='x+y':
                    mirror_roi = numpy.hstack((
                        (numpy.rot90(cov_roi_mat[n1: n2, m1: m2-roi_width], 2)),
                        (numpy.rot90(cov_roi_mat[n1: n2, m1+roi_width: m2], 2)) ))
                else:
                    mirror_roi = (numpy.rot90(cov_roi_mat[n1: n2, m1: m2], 2))

                cov_roi_mat[nn1: nn2, mm1: mm2] = mirror_roi

                m1 += step_length
                nn1 += step_width

        n1 += step_width
        mm1 += step_length
    return cov_roi_mat





def compute_ztilt_covariances(n_subaps1, xy_separations, sf, sf_dx, nSamp, wfs1_diam):
    """Function used by computeButt to calculate covariance. Note: not implemented!
    
    Parameters:
        n_subaps1 (int): number of sub-apertures in SHWFS.
        xy_separations (ndarray): x and y sub-aperture separations.
        sf (ndarray): Von-Karman structure function.
        sf_dx (float): structure function spatial frequency.
        nSamp (int): how many times to sample covariance across sub-aperture diameter.
        wfs1_diam (float): SHWFS sub-aperture diameter in the telescope's pupil.
        
    Returns:
        ndarray: xx covariance.
        ndarray: yy covariance."""

    #scaling to arcsec^2
    scaling = 206265.*206265.*3.* (500.e-9/(numpy.pi*wfs1_diam))**2;   
    #further scaling to get from arcsec^2 to units used elsewhere in this module
    fudgeFactor = (206265.**2) * (500.e-9**2) / ( 8. * (numpy.pi**2) * (wfs1_diam**2) )    

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







def compute_covariance_xx(separation, subap1_rad, subap2_rad, trans, L0, offset_present):
    """Calculates xx covariance - x-axis covariance between centroids - for one turbulent 
    layer in a single GS combination.
    
    Parameters:
        separation (ndarray): x and y sub-aperture separations (m).
        subap1_rad (float): radius of SHWFS sub-apertures in SHWFS.1.
        subap2_rad (float): radius of SHWFS sub-apertures in SHWFS.2.
        trans_wfs1 (float): translation of SHWFS.1 sub-aperture separation due to GS position and layer altitude.
        trans_wfs2 (float): translation of SHWFS.2 sub-aperture separation due to GS position and layer altitude.
        L0 (float): L0 value for turbulent layer (m).
    
    Returns:
        ndarray: xx spatial covariance"""


    separation = (separation + trans).astype('float64')
    nan_store = (numpy.isnan(separation[..., 1])==True)

    # y1 = separation[..., 1] + (subap2_rad - subap1_rad)
    # r1 = numpy.array(numpy.sqrt(separation[..., 0]**2 + y1**2))

    # y2 = separation[..., 1] - (subap2_rad + subap1_rad)
    # r2 = numpy.array(numpy.sqrt(separation[..., 0]**2 + y2**2))

    # y3 = separation[..., 1] + (subap2_rad + subap1_rad)
    # r3 = numpy.array(numpy.sqrt(separation[..., 0]**2 + y3**2))

    # cov_xx = (- 2 * structure_function_vk(r1, L0)
    #        + structure_function_vk(r2, L0)
    #        + structure_function_vk(r3, L0))

    cov_xx = (- 2 * structure_function_vk(numpy.array(numpy.sqrt(separation[..., 0]**2 + (separation[..., 1] + (subap2_rad - subap1_rad))**2)), L0)
           + structure_function_vk(numpy.array(numpy.sqrt(separation[..., 0]**2 + (separation[..., 1] - (subap2_rad + subap1_rad))**2)), L0)
           + structure_function_vk(numpy.array(numpy.sqrt(separation[..., 0]**2 + (separation[..., 1] + (subap2_rad + subap1_rad))**2)), L0))

    cov_xx[nan_store] = 0.

    if offset_present==True:
        cov_xx = cov_xx.sum(3)



    return cov_xx






def compute_covariance_yy(separation, subap1_rad, subap2_rad, trans, L0, offset_present):
    """Calculates yy covariance - y-axis covariance between centroids - for one turbulent 
    layer in a single GS combination.
    
    Parameters:
        separation (ndarray): x and y sub-aperture separations (m).
        subap1_rad (float): radius of SHWFS sub-apertures in SHWFS.1.
        subap2_rad (float): radius of SHWFS sub-apertures in SHWFS.2.
        trans_wfs1 (float): translation of SHWFS.1 sub-aperture separation due to GS position and layer altitude.
        trans_wfs2 (float): translation of SHWFS.2 sub-aperture separation due to GS position and layer altitude.
        L0 (float): L0 value for turbulent layer (m).
    
    Returns:
        ndarray: yy spatial covariance"""

    separation = (separation + trans).astype('float64')
    nan_store = (numpy.isnan(separation[..., 0])==True)

    # x1 = separation[..., 0] + (subap2_rad - subap1_rad)
    # r1 = numpy.array(numpy.sqrt(x1**2 + separation[..., 1]**2))

    # x2 = separation[..., 0] - (subap2_rad + subap1_rad)
    # r2 = numpy.array(numpy.sqrt(x2**2 + separation[..., 1]**2))

    # x3 = separation[..., 0] + (subap2_rad + subap1_rad)
    # r3 = numpy.array(numpy.sqrt(x3**2 + separation[..., 1]**2))

    # cov_yy = (- 2 * structure_function_vk(r1, L0)
    #         + structure_function_vk(r2, L0)
    #         + structure_function_vk(r3, L0))

    cov_yy = (- 2 * structure_function_vk(numpy.array(numpy.sqrt((separation[..., 0] + (subap2_rad - subap1_rad))**2 + separation[..., 1]**2)), L0)
            + structure_function_vk(numpy.array(numpy.sqrt((separation[..., 0] - (subap2_rad + subap1_rad))**2 + separation[..., 1]**2)), L0)
            + structure_function_vk(numpy.array(numpy.sqrt((separation[..., 0] + (subap2_rad + subap1_rad))**2 + separation[..., 1]**2)), L0))

    cov_yy[nan_store] = 0.

    if offset_present==True:
        cov_yy = cov_yy.sum(3)

    return cov_yy






def structure_function_vk(separation, L0):
    """Von Karman structure function for analytically generating covariance between SHWFS sub-apertures.
    
    Parameters:
        separation (ndarray): separation of SHWFS sub-apertures (m).
        L0 (float): L0 value (m).
        
    Returns:
        ndarray: spatial covariance."""

    dprf0 = (2*numpy.pi/L0)*separation
    k1 = 0.1716613621245709486
    res = numpy.zeros(dprf0.shape)

    if dprf0[numpy.isnan(dprf0)==False].max() > 4.71239:
        dprf0[numpy.isnan(dprf0)==True] = 1e20
        res[dprf0>4.71239] = asymp_macdo(dprf0[dprf0>4.71239])
        res[dprf0<=4.71239] = -macdo_x56(dprf0[dprf0<=4.71239])

    else:
        res = -macdo_x56(dprf0)

    return res * k1







def asymp_macdo(x):
    """Computes a term involved in the computation of the phase struct function with a finite outer scale 
    according to the Von-Karman model. The term involves the MacDonald function (modified bessel function 
    of second kind) K_{5/6}(x), and the algorithm uses the asymptotic form for x ~ infinity.
    
    Warnings :
    - This function makes a doubleing point interrupt for x=0
    and should not be used in this case.
    - Works only for x>0.
    
    
    Parameters:
        x (ndarray): (2*numpy.pi/L0)*separation < 4.71239
        
    Returns:
        ndarray: spatial covariance"""

    # k2 is the value for
    # gamma_R(5./6)*2^(-1./6)
    k2 = 1.00563491799858928388289314170833
    k3 = 1.25331413731550012081   #  sqrt(pi/2)
    a1 = 0.22222222222222222222   #  2/9
    a2 = -0.08641975308641974829  #  -7/89
    a3 = 0.08001828989483310284   # 175/2187

    x1 = 1./x
    res = (	k2
        - k3 * numpy.exp(-x) * x**(1./3)
        * (1.0 + x1*(a1 + x1*(a2 + x1*a3)))
    )

    return res







def macdo_x56(x):
	"""Computation of the function f(x) = x^(5/6)*K_{5/6}(x) using a series for the esimation of K_{5/6}, taken from Rod Conan thesis: 
	K_a(x)=1/2 \sum_{n=0}^\infty \frac{(-1)^n}{n!}\left(\Gamma(-n-a) (x/2)^{2n+a} + \Gamma(-n+a) (x/2)^{2n-a} \right), with a = 5/6.

	Setting x22 = (x/2)^2, setting uda = (1/2)^a, and multiplying by x^a, this becomes: 
	x^a * Ka(x) = 0.5 $ -1^n / n! [ G(-n-a).uda x22^(n+a) + G(-n+a)/uda x22^n ] 
	Then we use the following recurrence formulae on the following quantities:
	G(-(n+1)-a) = G(-n-a) / -a-n-1
	G(-(n+1)+a) = G(-n+a) /  a-n-1
	(n+1)! = n! * (n+1)
	x22^(n+1) = x22^n * x22

	At each iteration on n, one will use the values already computed at step (n-1). For efficiency, the values of G(a) and G(-a) 
	are hardcoded instead of being computed. The first term of the series has also been skipped, as it vanishes with another term in 
	the expression of Dphi.

	Parameters:
		x (ndarray): (2*numpy.pi/L0)*separation > 4.71239

	Returns:
		ndarray: spatial covariance."""

	a = 5./6
	x2a = x**(2.*a)
	x22 = x * x/4.


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

	# stop

	for n in range(10):
		s += (Gma[n+1]*x2a + Ga[n+1]) * x2n
		# Prepare recurrent iteration for next step
		x2n *= x22


	return s








if __name__ == "__main__":
	n_wfs = 3
	gs_alt = numpy.array([90000]*n_wfs)
	gs_pos = numpy.array([[0.,-20.], [0.,20], [30,0]])
	wavelength = numpy.array([500e-9]*n_wfs)
	n_layer = 4
	# layer_alt = numpy.array([0])
	layer_alt = numpy.array([0, 2000, 4000, 9282])
	fit_L0 = False
	offset_present = False
	r0 = numpy.array([0.1]*n_layer)
	L0 = numpy.array([25.]*n_layer)
	shwfs_shift = numpy.array(([0,0],[0,0],[0,0]))
	shwfs_rot = numpy.array([0,0,0])
	roi_envelope = 6
	roi_belowGround = 6
	roi_axis = 'x and y'
	styc_method = True
	fit_tt_track = False
	tt_track_present = False
	tt_track = numpy.array([0.0,0.0])

	fit_lgs_track = False
	lgs_track_present = False
	lgs_track = numpy.array(([1.,2.],[3., 4.], [5., 6.]))

	fit_offset = False
	offset_present = True

	delta_xSep = numpy.array([1, 0, 0, 0])
	delta_ySep = numpy.array([2, 0, 0, 0])

	"""CANARY"""
	tel_diam = 4.2
	obs_diam = 1.0
	subap_diam = numpy.array([0.6]*n_wfs)
	n_subap = numpy.array([36]*n_wfs)
	nx_subap = numpy.array([7]*n_wfs)

	# """AOF"""
	# tel_diam = 8.2
	# obs_diam = 0.94
	# subap_diam = numpy.array([0.205]*n_wfs)
	# n_subap = numpy.array([1248]*n_wfs)
	# nx_subap = numpy.array([40]*n_wfs)

	# """HARMONI"""
	# tel_diam = 39.0
	# obs_diam = 4.0
	# subap_diam = numpy.array([0.507]*n_wfs)
	# n_subap = numpy.array([4260]*n_wfs)
	# nx_subap = numpy.array([74]*n_wfs)

	pupil_mask = make_pupil_mask('circle', n_subap, nx_subap[0], obs_diam, tel_diam)

	matrix_region_ones = numpy.ones((n_subap[0], n_subap[0]))
	mm, mmc, md = get_mappingMatrix(pupil_mask, matrix_region_ones)

	onesMat, wfsMat_1, wfsMat_2, allMapPos, selector, xy_separations = roi_referenceArrays(pupil_mask, gs_pos, tel_diam, roi_belowGround, roi_envelope)
	onesMat, wfsMat_1, wfsMat_2, allMapPos_acrossMap, selector, xy_separations_acrossMap = roi_referenceArrays(pupil_mask, 
		gs_pos, tel_diam, pupil_mask.shape[0]-1, roi_envelope)

	s1 = time.time()
	params = covariance_roi_l3s(pupil_mask, subap_diam, wavelength, tel_diam, n_subap, gs_alt, 
		gs_pos, n_layer, layer_alt, L0, allMapPos_acrossMap, xy_separations_acrossMap, roi_axis, 
		roi_belowGround, roi_envelope, styc_method=True, lgs_track_present=False, offset_present=False, 
		fit_layer_alt=True, fit_lgs_track=False, fit_offset=False, fit_L0=False, wind_profiling=True)
	print('Conf: {}'.format(time.time() - s1))
	
	s = time.time()
	r = params._make_covariance_roi_l3s_(layer_alt, r0, L0, 
		lgs_track=lgs_track, shwfs_shift=shwfs_shift, shwfs_rot=shwfs_rot, 
		delta_xSep=delta_xSep, delta_ySep=delta_ySep)
	f = time.time()
	print('Make: {}'.format(f-s))

	pyplot.figure()
	pyplot.imshow(r)






	# """AOF"""
	# n_wfs = 2
	# tel_diam = 8.2
	# obs_diam = 1.1
	# subap_diam = numpy.array([0.205]*n_wfs)
	# n_subap = numpy.array([1240]*n_wfs)
	# nx_subap = numpy.array([40]*n_wfs)
	# wavelength = numpy.array([589e-9]*n_wfs)
	# gs_pos = numpy.array([[-64.,0.], [64.,0.]])
	
	# # n_layer = 29
	# # gs_alt = numpy.array([95000.]*n_wfs)*1.0919539999999999
	# # layer_alt = numpy.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000])*1.0919539999999999

	# n_layer = 2
	# layer_alt = numpy.array([0, 8000])*1.0919539999999999

	# gs_alt = numpy.array([95000.]*n_wfs)*1.0919539999999999
	# fit_L0 = False
	# r0 = numpy.array([0.2, 0.2])
	# # r0 = numpy.array([0.2]*n_layer)
	# L0 = numpy.array([25.]*n_layer)
	# roi_envelope = 1
	# roi_belowGround = 1
	# roi_axis = 'x and y'

	# pupil_mask = make_pupil_mask('circle', n_subap, nx_subap[0], obs_diam, tel_diam)
	# matrix_region_ones = numpy.ones((n_subap[0], n_subap[0]))
	# mm, mmc, md = get_mappingMatrix(pupil_mask, matrix_region_ones)
	# onesMat, wfsMat_1, wfsMat_2, allMapPos_acrossMap, selector, xy_separations_acrossMap = roi_referenceArrays(pupil_mask, 
	# 	gs_pos, tel_diam, pupil_mask.shape[0]-1, roi_envelope)

	# params = covariance_roi_l3s(pupil_mask, subap_diam, wavelength, tel_diam, n_subap, gs_alt, 
	# 	gs_pos, n_layer, layer_alt, L0, allMapPos_acrossMap, xy_separations_acrossMap, 
	# 	roi_axis, roi_belowGround, roi_envelope, styc_method=True, wind_profiling=False, 
	# 	lgs_track_present=False, offset_present=False, fit_layer_alt=False, 
	# 	fit_lgs_track=False, fit_offset=False, fit_L0=False)

	# s = time.time()
	# cov_roi = params._make_covariance_roi_l3s_(layer_alt, r0, L0, lgs_track=lgs_track, shwfs_shift=shwfs_shift, 
	# 	shwfs_rot=shwfs_rot, delta_xSep=delta_xSep, delta_ySep=delta_ySep)
	# f = time.time()
	# print('Time Taken: {}'.format(f-s))

	# pyplot.figure()
	# pyplot.imshow(cov_roi)