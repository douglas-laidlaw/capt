import time
import math
import numpy
import itertools
from matplotlib import pyplot; pyplot.ion()
from capt.misc_functions.make_pupil_mask import make_pupil_mask
from capt.misc_functions.mapping_matrix import get_mappingMatrix
from capt.roi_functions.roi_referenceArrays import roi_referenceArrays

 
class covariance_roi(object):

	def __init__(self, pupil_mask, subap_diam, wavelength, tel_diam, n_subap, gs_alt, gs_pos, 
		n_layer, layer_alt, L0, allMapPos, xy_separations, roi_axis, styc_method=True, 
		wind_profiling=False, tt_track_present=False, lgs_track_present=False, 
		offset_present=False, fit_layer_alt=False, fit_tt_track=False, 
		fit_lgs_track=False, fit_offset=False, fit_L0=False):
		
		"""Configuration used to generate covariance map ROI.

		Parameters:
			pupil_mask (ndarray): mask of SHWFS sub-apertures within the telescope's pupil.
			subap_diam (ndarray): diameter of SHWFS sub-aperture in telescope's pupil.
			wavelength (ndarray): SHWFS centroid wavelengh (nm).
			wavelength (ndarray): SHWFS centroid wavelengh (nm).
			tel_diam (float): diameter of telescope pupil.
			n_subap (ndarray): number of sub-apertures within each SHWFS.
			gs_alt (ndarray): GS altitude. 0 for NGS (LGS not tested!).
			gs_pos (ndarray): GS asterism in telescope FoV.
			allMapPos (ndarray): position of each SHWFS sub-aperture within the telescope's pupil.
			xy_separations (ndarray): x and y SHWFS sub-aperture separations.
			n_layer (int): number of turbulent layers.
			layer_alt (ndarray): altitudes of turbulent layers (m).
			wind_profiling (bool): determines whether covariance map ROI is to be used for wind profiling.
			tt_track_present (bool): generate covariance map ROI with linear additions to covariance (from vibrations/track).
			offset_present (bool): determines whether covariance map ROI is to account for a SHWFS shift/rotation.
			fit_tt_track (bool): determines whether the generated covariance map ROI is to fit track.
			fit_offset (bool): determines whether the generated covariance map ROI is to fit SHWFS shift/rotation.
			fit_L0 (bool): determines whether generated covariance map ROI is to be used for L0 profiling.
			L0 (ndarray): L0 profile (m).
			roi_axis (str): in which axis to express ROI ('x', 'y', 'x+y' or 'x and y')
			styc_method (bool): use styc method of analytically generating covariance."""

		self.n_wfs = gs_pos.shape[0]
		self.wind_profiling = wind_profiling
		self.pupil_mask = pupil_mask
		self.subap_diam = subap_diam
		self.wavelength = wavelength
		self.tel_diam = tel_diam
		self.gs_alt = gs_alt
		self.L0 = L0
		self.n_layer = n_layer
		self.n_subap = n_subap
		self.n_subap_from_pupilMask = int(self.pupil_mask.sum())
		self.nx_subap = pupil_mask.shape[0]
		self.radSqaured_to_arcsecSqaured = ((180./numpy.pi) * 3600)**2
		selector = numpy.array((range(self.n_wfs)))
		self.selector = numpy.array((list(itertools.combinations(selector, 2))))
		#Determine no. of GS combinations
		self.combs = xy_separations.shape[0]
		self.roi_width = xy_separations.shape[1]
		self.roi_length = xy_separations.shape[2]
		self.roi_belowGround = self.roi_length-self.nx_subap
		self.roi_centre_width = int(-1 + ((self.roi_width+1)/2.))
		self.translation = numpy.zeros((n_layer, 2, self.combs, 2))
		self.subap_layer_diameters = numpy.zeros((n_layer, 2, self.combs))
		self.roi_axis = roi_axis
		self.offset_present = offset_present
		self.fit_offset = fit_offset
		self.offset_set = False
		self.styc_method = styc_method
		self.fit_L0 = fit_L0
		# Swap X and Y GS positions to agree with legacy code
		self.gs_pos = gs_pos[:,::-1] 
		self.fit_layer_alt = fit_layer_alt
		self.scale_factor = numpy.zeros((n_layer, self.combs))

		self.tt_track_present = tt_track_present
		self.fit_tt_track = fit_tt_track
		self.tt_track_set = False

		self.lgs_track_present = lgs_track_present
		self.fit_lgs_track = fit_lgs_track
		self.lgs_track_set = False

		#generate arrays to be filled with analytically generated covariance
		if self.roi_axis!='y':
			self.cov_xx = numpy.zeros((n_layer, self.combs, self.roi_width, self.roi_length)).astype('float64')
		#generate arrays to be filled with analytically generated covariance
		if self.roi_axis!='x':
			self.cov_yy = numpy.zeros((n_layer, self.combs, self.roi_width, self.roi_length)).astype('float64')

		if self.offset_present==False and self.fit_offset==False:
			self.compute_cov_offset = False
			self.meanDenominator = numpy.array([1.]*self.combs)
			self.xy_separations = xy_separations

		if fit_tt_track==True or tt_track_present==True:
			self.tt_trackMatrix_locs = tt_trackMatrix_locs(xy_separations.copy(), roi_axis)
			self.tt_track = numpy.zeros(self.tt_trackMatrix_locs.shape).astype('float64')

		if fit_lgs_track==True or lgs_track_present==True:
			self.lgs_trackMatrix_locs = lgs_trackMatrix_locs(xy_separations.copy(), roi_axis)
			self.lgs_track = numpy.zeros(self.lgs_trackMatrix_locs.shape).astype('float64')

		if self.offset_present==True or self.fit_offset==True:
			self.compute_cov_offset = True
			self.allMapPos = allMapPos
			self.subap_layer_positions_atSeparations()

			x_seps = numpy.array([xy_separations.T[0]]*self.n_subap[0])
			y_seps = numpy.array([xy_separations.T[1]]*self.n_subap[0])
			self.xy_seps = numpy.stack((x_seps, y_seps)).T

			self.xy_separations=numpy.zeros((self.combs, self.roi_width, self.roi_length, 
								self.subap_sep_positions.shape[3], 2))

			self.subap_positions_wfsAlignment = numpy.zeros((self.combs, 2, self.roi_width, self.roi_length, 
								self.subap_sep_positions.shape[3], 2)).astype("float64")

		#compile covariance ROIs
		if self.roi_axis=='x' or self.roi_axis=='y' or self.roi_axis=='x+y':
			self.covariance_slice_fixed = numpy.zeros((self.n_layer, self.combs*self.roi_width, self.roi_length)).astype('float64')
		
		if self.roi_axis=='x and y':
			self.covariance_slice_fixed = numpy.zeros((self.n_layer, self.combs*self.roi_width, self.roi_length*2)).astype('float64')

		self.timingStart = time.time()

		#calculate fixed parameters
		if self.fit_layer_alt==False:
			self.subap_parameters(layer_alt)

		if self.fit_L0 == offset_present == self.fit_offset == self.wind_profiling == False:
			if self.styc_method == True:
				self.computeStyc(L0)
			if self.styc_method == False:
				self.self.computeButt(L0)
			self.fixedLayerParameters()






	def _make_covariance_roi_(self, layer_alt, r0, L0, tt_track=False, lgs_track=False, shwfs_shift=False, shwfs_rot=False, 
		delta_xSep=False, delta_ySep=False):
		"""Master node for generating covariance map ROI for L3S.1.
		
		Parameters:
			r0 (ndarray): r0 profile (m).
			L0 (ndarray): L0 profile (m).
			shwfs_shift (ndarray): SHWFS shift in x and y (m).
			shwfs_rot(ndarray): SHWFS rotation.
			delta_xSep (ndarray): shift x separation in covariance map ROI (developed for wind profiling).
			delta_ySep (ndarray): shift y separation in covariance map ROI (developed for wind profiling).
			
		Return:
			ndarray: analytically generated covariance map ROI with ground-layer mitigated."""

		
		if self.fit_L0 == True or self.offset_present==True or self.fit_offset==True or self.wind_profiling==True or self.fit_layer_alt==True:
			
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
					self.computeStyc(L0)

				self.fixedLayerParameters()

				if self.offset_present==True and self.fit_offset==self.wind_profiling==self.fit_L0==self.fit_layer_alt==False:

					self.offset_set=True

		covariance_slice = ((self.covariance_slice_fixed.T[:,:] * (L0/r0)**(5./3.)).T).sum(0)
		covariance_slice *= self.radSqaured_to_arcsecSqaured

		if self.tt_track_present==True or self.fit_tt_track==True:
			if self.tt_track_set==False:
				self.set_tt_tracking_values(tt_track)
				if self.fit_tt_track!=True:
					self.tt_track_set = True

			covariance_slice += self.tt_track

		if self.lgs_track_present==True or self.fit_lgs_track==True:
			if self.lgs_track_set==False:
				self.set_lgs_tracking_values(lgs_track)
				if self.fit_lgs_track!=True:
					self.lgs_track_set = True

			covariance_slice += self.lgs_track

		return covariance_slice



	def set_tt_tracking_values(self, tt_track):
		"""Generates tracking matirx - linear additon to covariance map ROI.
		
		Parameters:
			track (ndarray): 1s for locations of ROI within map"""

		if self.roi_axis=='x':
			self.tt_track[self.tt_trackMatrix_locs==1] = tt_track[0]

		if self.roi_axis=='y':
			self.tt_track[self.tt_trackMatrix_locs==1] = tt_track[1]

		if self.roi_axis=='x+y':
			self.tt_track[self.tt_trackMatrix_locs==1] = (tt_track[0]+tt_track[1])/2.		

		if self.roi_axis=='x and y':
			self.tt_track[self.tt_trackMatrix_locs==1] = tt_track[0]
			self.tt_track[self.tt_trackMatrix_locs==2] = tt_track[1]




	def set_lgs_tracking_values(self, lgs_track):
		"""Generates lgs tracking matirx - linear additon to covariance map ROI.

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

		for layer_n in range(self.n_layer):
			for wfs_n in range(2):
				for comb in range(self.combs):

					gs_pos = self.gs_pos[self.selector[comb, wfs_n]]
					layer_altitude = layer_alt[layer_n]

					# Scale for LGS
					if self.gs_alt[self.selector[comb, wfs_n]] != 0:
						# print("Its an LGS!")
						self.scale_factor[layer_n, comb] = (1 - layer_altitude/self.gs_alt[self.selector[comb,wfs_n]])
					else:
						self.scale_factor[layer_n, comb] = 1.

					# translate due to GS position
					gs_pos_rad = numpy.array(gs_pos) * (numpy.pi/180.) * (1./3600.)

					# print("GS Positions: {} rad".format(gs_pos_rad))
					self.translation[layer_n, wfs_n, comb] = gs_pos_rad * layer_altitude
					self.subap_layer_diameters[layer_n, wfs_n, comb] = self.subap_diam[self.selector[comb, wfs_n]] * self.scale_factor[layer_n, comb]





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
		ySeps = numpy.ones((self.combs, self.roi_width, self.roi_length, self.nx_subap**2)) * numpy.nan
		xSeps = numpy.ones((self.combs, self.roi_width, self.roi_length, self.nx_subap**2)) * numpy.nan
		self.meanDenominator = numpy.zeros(self.cov_xx[0].shape)
		self.subap_sep_positions = numpy.ones((self.combs, self.roi_width, self.roi_length, self.nx_subap**2, 2))
		self.allMapPos[self.allMapPos>=covMapDim] = 0.

		for comb in range(self.combs):

			mmLocations = (covMapDim * self.allMapPos[comb,:,:,0]) + self.allMapPos[comb,:,:,1]
			self.meanDenominator[comb] = onesMM[:,mmLocations].sum(0)
			ySepsMM = -yPosMM[:,mmLocations]
			xSepsMM = -xPosMM[:,mmLocations]

			for env in range(self.roi_width):
				for l in range(self.roi_length):

					ySeps[comb, env,l] = ySepsMM[:, env, l]
					xSeps[comb, env,l] = xSepsMM[:, env, l]

			self.subap_sep_positions[comb,:,:,:,0] = xSeps[comb]
			self.subap_sep_positions[comb,:,:,:,1] = ySeps[comb]

		# stop
		# self.meanDenominator[0] = mapDensity
		self.meanDenominator[self.meanDenominator==0] = 1.




	def subap_wfsAlignment(self, shwfs_shift, shwfs_rot):
		"""Calculates x and y sub-aperture separations under some SHWFS shift and/or rotation.
		
		Parameters:
			shwfs_shift (ndarray): SHWFS shift in x and y (m).
			shwfs_rot (ndarray): SHWFS rotation."""

		for comb in range(self.combs):

			for wfs_i in range(2):	

				theta = (shwfs_rot[self.selector[comb, [1,0][wfs_i]]]) * numpy.pi/180.

				xtp = self.subap_sep_positions[comb, :, :, :, 1]
				ytp = self.subap_sep_positions[comb, :, :, :, 0]

				uu = xtp * numpy.cos(theta) - ytp * numpy.sin(theta)
				vv = xtp * numpy.sin(theta) + ytp * numpy.cos(theta)

				self.subap_positions_wfsAlignment[comb,wfs_i,:,
					:,:,1] = uu - shwfs_shift[self.selector[comb, wfs_i],0]
				self.subap_positions_wfsAlignment[comb,wfs_i,:,
					:,:,0] = vv - shwfs_shift[self.selector[comb, wfs_i],1]

			self.xy_separations[comb] = -(self.subap_positions_wfsAlignment[comb,0,
				:,:,:] - self.subap_positions_wfsAlignment[comb,1,self.roi_centre_width,self.roi_belowGround])




	def fixedLayerParameters(self):
		"""Creates covariance map ROI matrix where each combination/layer is fixed such that, if only r0 is being 
		fitted, the 2D response function is generated once whereafter its shape is iteratively multiplied."""

		self.covariance_slice_fixed *= 0.

		for layer_n in range(self.n_layer):

			for comb in range(self.combs):

				# print("covmat coords: ({}: {}, {}: {})".format(cov_mat_coord_x1, cov_mat_coord_x2, cov_mat_coord_y1, cov_mat_coord_y2))
				r0_scale1 = ((self.wavelength[self.selector[comb,0]] * self.wavelength[self.selector[comb,1]])
						/ (8 * numpy.pi**2 * self.subap_diam[self.selector[comb, 0]] * self.subap_diam[self.selector[comb, 1]] ))

				if self.roi_axis=='x+y':
					cov_xx = self.cov_xx[layer_n, comb]
					cov_yy = self.cov_yy[layer_n, comb]

					self.covariance_slice_fixed[layer_n, comb*self.roi_width: 
						(comb+1)*self.roi_width] += ((cov_xx+cov_yy)*(r0_scale1/2.))


				if self.roi_axis=='x':
					cov_xx = self.cov_xx[layer_n, comb]

					self.covariance_slice_fixed[layer_n, comb*self.roi_width: 
						(comb+1)*self.roi_width] += cov_xx*r0_scale1


				if self.roi_axis=='y':
					cov_yy = self.cov_yy[layer_n, comb]

					self.covariance_slice_fixed[layer_n, comb*self.roi_width: 
						(comb+1)*self.roi_width] += cov_yy*r0_scale1
				

				if self.roi_axis=='x and y':
					cov_xx = self.cov_xx[layer_n, comb]
					cov_yy = self.cov_yy[layer_n, comb]

					self.covariance_slice_fixed[layer_n, comb*self.roi_width: 
						(comb+1)*self.roi_width] += numpy.hstack(((cov_xx*r0_scale1), (cov_yy*r0_scale1)))



	def computeStyc(self, L0):
		"""Uses styc method for analytically generating covariance.

		Parameters:
			L0 (ndarray): L0 profile (m)."""

		for layer_n in range(self.n_layer):
			for comb in range(self.combs):
				
				xy_seps = self.xy_separations[comb].copy() * self.scale_factor[layer_n, comb]
				# print(xy_seps)
				if self.wind_profiling==True:
					if self.offset_present==True or self.fit_offset==True:
						xy_seps[:,:,:,0] += self.delta_xSep[layer_n]
						xy_seps[:,:,:,1] += self.delta_ySep[layer_n]
					if self.offset_present==False and self.fit_offset==False:
						xy_seps[:,:,0] += self.delta_xSep[layer_n]
						xy_seps[:,:,1] += self.delta_ySep[layer_n]


				if self.roi_axis!='y':
					self.cov_xx[layer_n, comb] = compute_covariance_xx(xy_seps, 
						self.subap_layer_diameters[layer_n, 0, comb], L0[layer_n], self.translation[layer_n, 0, comb], 
						self.translation[layer_n, 1, comb], self.compute_cov_offset)/self.meanDenominator[comb]

				if self.roi_axis!='x':
					self.cov_yy[layer_n, comb] = compute_covariance_yy(xy_seps, 
						self.subap_layer_diameters[layer_n, 0, comb], L0[layer_n], self.translation[layer_n, 0, comb], 
						self.translation[layer_n, 1, comb], self.compute_cov_offset)/self.meanDenominator[comb]




def tt_trackMatrix_locs(xy_separations, map_axis):
	combs = xy_separations.shape[0]
	width = xy_separations.shape[1]
	length = xy_separations.shape[2]
	
	if map_axis=='x and y':
		no_axes = 2
	else:
		no_axes = 1
	tt_trackMatrix = numpy.zeros((combs*width, xy_separations.shape[2]*no_axes))


	for i in range(combs):
		for ax in range(no_axes):
			if ax==0:
				xy_separations[i, :, :, ax][numpy.isnan(xy_separations[i, :, :, ax])==False] = 1
				tt_trackMatrix[i*width: (i+1)*width, ax*length: (ax+1)*length] = xy_separations[i, :, :, ax].copy()
			if ax==1:
				xy_separations[i, :, :, ax][numpy.isnan(xy_separations[i, :, :, ax])==False] = 2
				tt_trackMatrix[i*width: (i+1)*width, ax*length: (ax+1)*length] = xy_separations[i, :, :, ax].copy()
	
	tt_trackMatrix[numpy.isnan(tt_trackMatrix)==True] = 0.

	return tt_trackMatrix.astype(int)




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





def computeButt(self, L0):
	"""Uses T. Butterley method for analytically generating covariance.

	Parameters:
		L0 (ndarray): L0 profile (m)."""

	for layer_n in range(self.n_layer):
		for comb in range(self.combs):
			wfs1_diam = self.subap_layer_diameters[layer_n, 0, comb]
			wfs2_diam = self.subap_layer_diameters[layer_n, 1, comb]
			n_subaps1 = self.nSubaps[self.selector[comb][0]]
			n_subaps2 = self.nSubaps[self.selector[comb][1]]
			xy_sep = self.xy_separations[layer_n, comb]

			#numerical integration code doesn't support different subap sizes
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
			self.cov_xx[layer_n, comb], self.cov_yy[layer_n, comb] = compute_ztilt_covariances(n_subaps1, 
				xy_sep, sf, sf_dx, nSamp, wfs1_diam)








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







def compute_covariance_xx(separation, subap1_diam, L0, trans_wfs1, trans_wfs2, offset_present):
    """Calculates xx covariance - x-axis covariance between centroids - for one turbulent 
    layer in a single GS combination.
    
    Parameters:
        separation (ndarray): x and y sub-aperture separations (m).
        subap1_diam (float): radius of SHWFS sub-apertures in SHWFS.1.
        subap2_diam (float): radius of SHWFS sub-apertures in SHWFS.2.
        trans_wfs1 (float): translation of SHWFS.1 sub-aperture separation due to GS position and layer altitude.
        trans_wfs2 (float): translation of SHWFS.2 sub-aperture separation due to GS position and layer altitude.
        L0 (float): L0 value for turbulent layer (m).
        offset_present (bool): if True, covariance is summed to map mean as a function of sub-aperture spearation.
    
    Returns:
        ndarray: xx spatial covariance"""

    separation = separation + (trans_wfs2 - trans_wfs1)
    nan_store = (numpy.isnan(separation[...,1])==True)

    x1 = separation[...,1] + (subap1_diam - subap1_diam) * 0.5
    r1 = numpy.array(numpy.sqrt(x1**2 + separation[...,0]**2))

    x2 = separation[...,1] - (subap1_diam + subap1_diam) * 0.5
    r2 = numpy.array(numpy.sqrt(x2**2 + separation[...,0]**2))

    x3 = separation[...,1] + (subap1_diam + subap1_diam) * 0.5
    r3 = numpy.array(numpy.sqrt(x3**2 + separation[...,0]**2))

    # x4 = separation[...,1] - ((subap1_diam - subap1_diam) * 0.5)
    # r4 = numpy.array(numpy.sqrt(x4**2 + separation[...,0]**2))

    cov_xx = (-2 * structure_function_vk(r1, L0)
            + structure_function_vk(r2, L0)
            + structure_function_vk(r3, L0))
            #- structure_function_vk(r4, L0))

    cov_xx[nan_store] = 0.

    if offset_present==True:
        cov_xx = cov_xx.sum(2)

    return cov_xx




def compute_covariance_yy(separation, subap1_diam, L0, trans_wfs1, trans_wfs2, offset_present):
    """Calculates yy covariance - y-axis covariance between centroids - for one turbulent 
    layer in a single GS combination.
    
    Parameters:
        separation (ndarray): x and y sub-aperture separations (m).
        subap1_diam (float): radius of SHWFS sub-apertures in SHWFS.1.
        subap2_diam (float): radius of SHWFS sub-apertures in SHWFS.2.
        trans_wfs1 (float): translation of SHWFS.1 sub-aperture separation due to GS position and layer altitude.
        trans_wfs2 (float): translation of SHWFS.2 sub-aperture separation due to GS position and layer altitude.
        L0 (float): L0 value for turbulent layer (m).
        offset_present (bool): if True, covariance is summed to map mean as a function of sub-aperture spearation.
    
    Returns:
        ndarray: yy spatial covariance"""    

    separation = separation + (trans_wfs2 - trans_wfs1)
    nan_store = (numpy.isnan(separation[...,0])==True)

    y1 = separation[...,0] + (subap1_diam - subap1_diam) * 0.5
    r1 = numpy.array(numpy.sqrt(separation[...,1]**2 + y1**2))

    y2 = separation[...,0] - (subap1_diam + subap1_diam) * 0.5
    r2 = numpy.array(numpy.sqrt(separation[...,1]**2 + y2**2))

    y3 = separation[...,0] + (subap1_diam + subap1_diam) * 0.5
    r3 = numpy.array(numpy.sqrt(separation[...,1]**2 + y3**2))

    # y4 = separation[...,0] - ((subap1_diam - subap1_diam) * 0.5)
    # r4 = numpy.array(numpy.sqrt(y4**2 + separation[...,1]**2))

    cov_yy = (-2 * structure_function_vk(r1, L0)
        + structure_function_vk(r2, L0)
        + structure_function_vk(r3, L0))
        #- structure_function_vk(r4, L0))

    cov_yy[nan_store] = 0.

    if offset_present==True:
        cov_yy = cov_yy.sum(2)

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


    # dprf0 = (2*numpy.pi/L0)*separation
    # k1 = 0.1716613621245709486
    # res = numpy.zeros(dprf0.shape)

    # if dprf0[numpy.isnan(dprf0)==False].max() > 4.71239:
    #     res[dprf0>4.71239] = asymp_macdo(dprf0[dprf0>4.71239])
    #     if dprf0[numpy.isnan(dprf0)==False].min() <= 4.71239:
    #         res[dprf0<=4.71239] = -macdo_x56(dprf0[dprf0<=4.71239])

    # else:
    #     res = -macdo_x56(dprf0)

    # return res * k1







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

    for n in range(10):
        s += (Gma[n+1]*x2a + Ga[n+1]) * x2n
        # Prepare recurrent iteration for next step
        x2n *= x22

    return s


if __name__ == "__main__":
	n_wfs = 2
	gs_alt = numpy.array([0]*n_wfs)
	gs_pos = numpy.array([[0.,-20.], [0.,20]])
	wavelength = numpy.array([500e-9]*n_wfs)
	n_layer = 1
	# layer_alt = numpy.array([0])
	layer_alt = numpy.array([9282])
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

	delta_xSep = numpy.array([0, 0, 0])
	delta_ySep = numpy.array([1.2, 0, 0])

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
	
	
	
	params = covariance_roi(pupil_mask, subap_diam, wavelength, tel_diam, n_subap, gs_alt, 
		gs_pos, n_layer, layer_alt, L0, allMapPos, xy_separations, roi_axis, styc_method=True, 
		tt_track_present=True, lgs_track_present=False, offset_present=False, fit_layer_alt=False, 
		fit_tt_track=False, fit_lgs_track=False, fit_offset=False, fit_L0=False, wind_profiling=False)
	
	
	s = time.time()
	r = params._make_covariance_roi_(layer_alt, r0, L0, tt_track, lgs_track, shwfs_shift, shwfs_rot, delta_xSep=delta_xSep, delta_ySep=delta_ySep)
	f = time.time()
	print('Time Taken: {}'.format(f-s))

	pyplot.figure()
	pyplot.imshow(r)