import time
import numpy
import itertools
from scipy.special import comb
from matplotlib import pyplot; pyplot.ion()
from capt.misc_functions.make_pupil_mask import make_pupil_mask
from capt.misc_functions.transform_matrix import transform_matrix
from capt.map_functions.covMap_fromMatrix import covMap_fromMatrix
from capt.matrix_functions.tt_removal_matrix import tt_removal_matrix
from capt.misc_functions.mapping_matrix import get_mappingMatrix, covMap_superFast

class covariance_matrix(object):
    
	def __init__(self, pupil_mask, subap_diam, wavelength, tel_diam, n_subap, gs_alt, gs_pos, 
		n_layer, layer_alt, r0, L0, styc_method=True, wind_profiling=False, tt_track_present=False, 
		lgs_track_present=False, offset_present=False, fit_layer_alt=False, fit_tt_track=False, 
		fit_lgs_track=False, fit_offset=False, fit_L0=False, matrix_xy=True, huge_matrix=False, 
		l3s1_transform=False, l3s1_matrix=None, remove_tt=False, remove_tt_matrix=None):

		"""Configuration used to generate covariance matrix.

		Parameters:
			pupil_mask (ndarray): mask of SHWFS sub-apertures within the telescope's pupil.
			subap_diam (ndarray): diameter of SHWFS sub-aperture in telescope's pupil.
			wavelength (ndarray): SHWFS centroid wavelengh (nm).
			tel_diam (float): diameter of telescope pupil.
			n_subap (ndarray): number of sub-apertures within each SHWFS.
			gs_alt (ndarray): GS altitude. 0 for NGS (LGS not tested!).
			gs_pos (ndarray): GS asterism in telescope FoV.
			n_layer (int): number of turbulent layers.
			layer_alt (ndarray): altitudes of turbulent layers (m).
			wind_profiling (bool): determines whether covariance map ROI is to be used for wind profiling.
			tt_track_present (bool): generate covariance map ROI with linear additions to covariance (from vibrations/track).
			offset_present (bool): determines whether covariance map ROI is to account for a SHWFS shift/rotation.
			fit_tt_track (bool): determines whether the generated covariance matrix is to fit track.
			fit_offset (bool): determines whether the generated covariance matrix is to fit SHWFS shift/rotation.
			fit_L0 (bool): determines whether generated covariance map ROI is to be used for L0 profiling.
			r0 (ndarray): r0 profile (m).
			L0 (ndarray): L0 profile (m).
			styc_method (bool): use styc method of analytically generating covariance.
			matrix_xy (bool): generate orthogonal covariance e.g. x1y1.
			huge_matrix (bool): if array size n_layer * cov_matrix is too big, reduce to cov_matrix (takes longer to generate)."""


		self.n_wfs = gs_pos.shape[0]
		self.subap_diam = subap_diam
		self.wavelength = wavelength
		self.tel_diam = tel_diam
		self.pupil_mask = pupil_mask
		self.gs_alt = gs_alt
        # Swap X and Y GS positions to agree with legacy code
		self.gs_pos = gs_pos[:,::-1]
		self.n_layer = n_layer
		self.layer_alt = layer_alt
		self.combs = int(comb(gs_pos.shape[0], 2, exact=True))
		selector = numpy.array((range(self.n_wfs)))
		self.selector = numpy.array((list(itertools.combinations(selector, 2))))
		self.n_subap = n_subap
		self.total_subaps = int(numpy.sum(self.n_subap))
		self.styc_method = styc_method
		self.radSqaured_to_arcsecSqaured = ((180./numpy.pi) * 3600)**2
		self.fit_L0 = fit_L0
		self.translation = numpy.zeros((self.n_layer, self.n_wfs, 2))
		self.subap_layer_diameters = numpy.zeros((self.n_layer, self.n_wfs)).astype("float64")
		self.wind_profiling = wind_profiling
		self.offset_present = offset_present
		self.fit_layer_alt = fit_layer_alt
		self.fit_offset = fit_offset
		self.offset_set = False
		self.matrix_xy=matrix_xy
		self.scale_factor = numpy.zeros((n_layer, self.n_wfs))

		self.tt_track_set = False
		self.fit_tt_track = fit_tt_track
		self.tt_track_present = tt_track_present
		if self.fit_tt_track==True or self.tt_track_present==True:
			self.tt_trackMatrix_locs = tt_trackMatrix_locs(2 * self.n_subap[0] * self.n_wfs, self.n_subap[0])
			self.tt_track = numpy.zeros((self.tt_trackMatrix_locs.shape)).astype('float64')

		self.lgs_track_set = False
		self.fit_lgs_track = fit_lgs_track
		self.lgs_track_present = lgs_track_present
		if self.fit_lgs_track==True or self.lgs_track_present==True:
			self.lgs_trackMatrix_locs = matrix_lgs_trackMatrix_locs(2 * self.n_subap[0] * self.n_wfs, self.n_subap)
			self.lgs_track = numpy.zeros((self.lgs_trackMatrix_locs.shape)).astype('float64')

		#make l3s1 transform matrix
		self.l3s1_transform = l3s1_transform
		if l3s1_transform==True:
			if l3s1_matrix is None:
				self.l3s1_matrix = transform_matrix(self.n_subap, self.n_wfs)
			else:
				self.l3s1_matrix = l3s1_matrix

		#make matrix for removing tip-tilt
		self.remove_tt = remove_tt
		if remove_tt==True:
			if remove_tt_matrix is None:
				self.remove_tt_matrix = tt_removal_matrix(self.n_subap[0], self.n_wfs)
			else:
				self.remove_tt_matrix = remove_tt_matrix

		#parameters required for fast generation of aligned/auto covariance 
		self.mm, mmc, md = get_mappingMatrix(self.pupil_mask, 
			numpy.ones((self.n_subap[0], self.n_subap[0])))
		mapOnes = covMap_superFast((pupil_mask.shape[0]*2)-1, 
			numpy.ones((self.n_subap[0], self.n_subap[0])), self.mm, mmc, md)
		self.flatMM = mapOnes.flatten().astype('int')
		self.fill_xx = numpy.zeros(self.flatMM.shape)
		self.fill_yy = numpy.zeros(self.flatMM.shape)
		self.fill_xy = numpy.zeros(self.flatMM.shape)

		yMapSep = mapOnes.copy() * numpy.arange(-(pupil_mask.shape[0]-1), 
			(pupil_mask.shape[0])) * self.tel_diam/pupil_mask.shape[0]
		yMapSepFlat = yMapSep.flatten()
		xMapSep = yMapSep.copy().T
		xMapSepFlat = xMapSep.flatten()
		
		if self.offset_present==False and self.fit_offset==False:
			self.subap_positions = [False]*self.n_wfs
			self.xy_separations = numpy.array((xMapSepFlat[self.flatMM==1], 
				yMapSepFlat[self.flatMM==1])).T

		if self.offset_present==True or self.fit_offset==True:
			self.subap_positions_wfsAlignment = numpy.zeros((self.n_wfs, 
				self.n_subap[0], 2)).astype("float64")
			wfs_subap_pos = (numpy.array(numpy.where(self.pupil_mask== 1)).T * 
				self.tel_diam/self.pupil_mask.shape[0])
			self.subap_positions = numpy.array([wfs_subap_pos]*self.n_wfs)

			self.auto_xy_separations = numpy.array((xMapSepFlat[self.flatMM==1], 
							yMapSepFlat[self.flatMM==1])).T

		if huge_matrix==True:
			print('THIS MATRIX BE HUUUUGGGEEEE')
			self.huge_matrix = True
		else:
			self.huge_matrix = False

		if self.fit_layer_alt==False:
			#calculate fixed parameters
			self.subap_parameters()

		self.timingStart = time.time()
		if self.fit_L0 == self.offset_present == self.fit_offset == self.wind_profiling == self.huge_matrix == False:
			self.compile_matrix(r0, L0)
			# print('Boo!')




			

	def _make_covariance_matrix_(self, layer_alt, r0, L0, tt_track=False, lgs_track=False, shwfs_shift=False, shwfs_rot=False, 
		delta_xSep=False, delta_ySep=False):
		"""Master node for generating covariance matrix.
		
		Parameters:
			r0 (ndarray): r0 profile (m).
			L0 (ndarray): L0 profile (m).
			shwfs_shift (ndarray): SHWFS shift in x and y (m).
			shwfs_rot (ndarray): SHWFS rotation.
			delta_xSep (ndarray): shift x separation in covariance matrix (developed for wind profiling).
			delta_ySep (ndarray): shift y separation in covariance matrix (developed for wind profiling).

		Return:
			ndarray: analytically generated covariance matrix."""


		if self.fit_L0 == True or self.offset_present==True or self.fit_offset==True or self.wind_profiling==True or self.fit_layer_alt==True or self.huge_matrix==True:
			
			if self.fit_layer_alt==True:
				self.subap_parameters(layer_alt)

			if self.wind_profiling==True:
				self.delta_xSep = delta_xSep.copy()
				self.delta_ySep = delta_ySep.copy()

			if self.fit_offset==True:
				self.subapPos_wfsAlignment(shwfs_shift, shwfs_rot)

			else:
				if self.offset_present==True and self.offset_set==False:
					self.subapPos_wfsAlignment(shwfs_shift, shwfs_rot)                    

			if self.offset_set==False:
				self.compile_matrix(r0, L0)
			
				if self.offset_present==True and self.fit_L0 == self.fit_offset == self.wind_profiling == self.fit_layer_alt == self.huge_matrix == False:
				
					self.offset_set=True

		if self.huge_matrix==False:
			covariance_matrix_output = (((self.covariance_matrix.T * 
				(L0/r0)**(5./3.)).T).sum(0)) * self.radSqaured_to_arcsecSqaured
		else:
			covariance_matrix_output = self.covariance_matrix * self.radSqaured_to_arcsecSqaured


		if self.l3s1_transform==True:
			covariance_matrix_output = numpy.matmul(numpy.matmul(self.l3s1_matrix, covariance_matrix_output), self.l3s1_matrix.T)

		if self.remove_tt==True:
			covariance_matrix_output = numpy.matmul(numpy.matmul(self.remove_tt_matrix, covariance_matrix_output), self.remove_tt_matrix)



		if self.tt_track_present==True or self.fit_tt_track==True:
			if self.tt_track_set==False:
				self.tt_track[self.tt_trackMatrix_locs==1] = tt_track[0]
				self.tt_track[self.tt_trackMatrix_locs==2] = tt_track[1]
				if self.fit_tt_track!=True:
					self.tt_track_set = True

			covariance_matrix_output += self.tt_track

		if self.lgs_track_present==True or self.fit_lgs_track==True:
			if self.lgs_track_set==False:
				self.set_lgs_tracking_values(lgs_track)
				if self.fit_lgs_track!=True:
					self.lgs_track_set = True

			covariance_matrix_output += self.lgs_track

		return covariance_matrix_output




	def set_lgs_tracking_values(self, lgs_track):
		"""Generates lgs tracking matirx - linear additon to covariance map ROI.

		Parameters:
			track (ndarray): 1s for locations of ROI within map"""

		counter = 1
		for i in range(self.combs+self.n_wfs):
			for j in range(2):
				self.lgs_track[self.lgs_trackMatrix_locs==counter] = lgs_track[i, j]
				counter += 1




	def subap_parameters(self):
		"""Calculate initial parameters that are fixed i.e. translation of sub-aperture positions with altitude."""



		for layer_n, layer_altitude in enumerate(self.layer_alt):
			for wfs_n in range(self.n_wfs):

				# Scale for LGS
				if self.gs_alt[wfs_n] != 0:
					# print("Its an LGS!")
					self.scale_factor[layer_n, wfs_n] = (1. - layer_altitude/self.gs_alt[wfs_n])
				else:
					self.scale_factor[layer_n, wfs_n] = 1.
				
				# translate due to GS position
				gs_pos_rad = numpy.array(self.gs_pos[wfs_n]) * (numpy.pi/180.) * (1./3600.)
				# print("GS Positions: {} rad".format(gs_pos_rad))
				self.translation[layer_n, wfs_n] = gs_pos_rad * layer_altitude
				self.subap_layer_diameters[layer_n, wfs_n] = self.subap_diam[wfs_n] * (self.scale_factor[layer_n, wfs_n])




	def subapPos_wfsAlignment(self, shwfs_shift, shwfs_rot):
		"""Calculates x and y sub-aperture separations under some SHWFS shift and/or rotation.
		
		Parameters:
			shwfs_shift (ndarray): SHWFS shift in x and y (m).
			shwfs_rot (ndarray): SHWFS rotation."""

		for wfs_i in range(self.n_wfs):

			theta = shwfs_rot[wfs_i] * numpy.pi/180.

			xtp = self.subap_positions[wfs_i,:,1]
			ytp = self.subap_positions[wfs_i,:,0]

			uu = xtp * numpy.cos(theta) - ytp * numpy.sin(theta)
			vv = xtp * numpy.sin(theta) + ytp * numpy.cos(theta)

			self.subap_positions_wfsAlignment[wfs_i,:,1] = uu + shwfs_shift[wfs_i,1]
			self.subap_positions_wfsAlignment[wfs_i,:,0] = vv + shwfs_shift[wfs_i,0]

		# d=off

	def compile_matrix(self, r0, L0):
		"""Build covariance matrix.
		
		Parameters:
			r0 (ndarray): r0 profile (m).
			L0 (ndarray): L0 profile (m)."""


		# Compile covariance matrix
		if self.huge_matrix==False:
			self.covariance_matrix = numpy.zeros((self.n_layer, 2 * self.total_subaps, 
				2 * self.total_subaps)).astype("float64")
		else:
			cov_mat = numpy.zeros((2 * self.total_subaps, 
				2 * self.total_subaps)).astype("float64")
			self.covariance_matrix = numpy.zeros((2 * self.total_subaps, 
				2 * self.total_subaps)).astype("float64")

		for layer_n in range(self.n_layer):
			# print("Compute Layer {}".format(layer_n))

			subap_ni = 0
			for wfs_i in range(self.n_wfs):
				subap_nj = 0
				# Only loop over upper diagonal of covariance matrix (symmetrical).
				for wfs_j in range(wfs_i+1):
					
					if self.offset_present==True or self.fit_offset==True:
						if wfs_i!=wfs_j:
							xy_separations = calculate_wfs_separations(self.n_subap[wfs_i], 
								self.n_subap[wfs_j], self.subap_positions_wfsAlignment[wfs_i], 
								self.subap_positions_wfsAlignment[wfs_j])
						else:
							xy_separations = self.auto_xy_separations

					else:
						xy_separations = self.xy_separations

					#calculate auto-covariance - same for each wfs, therefore only calculate once
					if wfs_i==wfs_j and wfs_i==0:
						# print(xy_separations* self.scale_factor[layer_n, wfs_i])
						auto_cov_xx, auto_cov_yy, auto_cov_xy = wfs_covariance(
							self.subap_layer_diameters[layer_n, wfs_i], 
							self.subap_layer_diameters[layer_n, wfs_j],
							L0[layer_n], xy_separations * self.scale_factor[layer_n, wfs_i], 
							self.translation[layer_n, wfs_i], self.translation[layer_n, wfs_j], 
							self.styc_method, self.matrix_xy)

						auto_cov_xx, auto_cov_yy, auto_cov_xy = self.map_aligned_covariance(auto_cov_xx, auto_cov_yy, auto_cov_xy, 
								self.n_subap[wfs_i], self.n_subap[wfs_j])
					
					if wfs_i==wfs_j:
						cov_xx = auto_cov_xx
						cov_yy = auto_cov_yy
						cov_xy = auto_cov_xy

					#calculate covariance between wfss
					if wfs_i!=wfs_j:
						cov_xx, cov_yy, cov_xy = wfs_covariance(
							self.subap_layer_diameters[layer_n, wfs_i], 
							self.subap_layer_diameters[layer_n, wfs_j],
							L0[layer_n], xy_separations * self.scale_factor[layer_n, wfs_i], 
							self.translation[layer_n, wfs_i], self.translation[layer_n, wfs_j], 
							self.styc_method, self.matrix_xy)

						if self.offset_present==False and self.fit_offset==False:
							cov_xx, cov_yy, cov_xy = self.map_aligned_covariance(cov_xx, cov_yy, cov_xy, 
								self.n_subap[wfs_i], self.n_subap[wfs_j])

					subap_ni = int(numpy.sum(self.n_subap[:wfs_i]))
					subap_nj = int(numpy.sum(self.n_subap[:wfs_j]))

					#coordinates of xx covariance
					cov_mat_coord_x1 = subap_ni * 2
					cov_mat_coord_x2 = subap_ni * 2 + self.n_subap[wfs_i]

					#coordinates of yy covariance
					cov_mat_coord_y1 = subap_nj * 2
					cov_mat_coord_y2 = subap_nj * 2 + self.n_subap[wfs_j]
					
					# r0_scale = ((self.wavelength[wfs_i] * self.wavelength[wfs_j]) / 
					#     (8 * (numpy.pi**2) * self.subap_layer_diameters[layer_n][wfs_i] * 
					#     self.subap_layer_diameters[layer_n][wfs_j])) / self.scale_factor[layer_n, wfs_i]

					r0_scale = ((self.wavelength[wfs_i] * self.wavelength[wfs_j]) / 
						(8 * (numpy.pi**2) * self.subap_diam[wfs_i] * 
						self.subap_diam[wfs_j]))

					# print(r0_scale*(2*(0.1**(-5./3))))
					if self.huge_matrix==False:
						self.covariance_matrix[layer_n, cov_mat_coord_x1: cov_mat_coord_x2, 
							cov_mat_coord_y1: cov_mat_coord_y2] = cov_xx * r0_scale

						self.covariance_matrix[layer_n, cov_mat_coord_x1 + self.n_subap[wfs_i]: 
							cov_mat_coord_x2 + self.n_subap[wfs_i], cov_mat_coord_y1 + self.n_subap[wfs_j]: 
							cov_mat_coord_y2 + self.n_subap[wfs_j]] = cov_yy * r0_scale

						if self.matrix_xy==True:
							self.covariance_matrix[layer_n, cov_mat_coord_x1 + self.n_subap[wfs_i]: 
								cov_mat_coord_x2 + self.n_subap[wfs_i], cov_mat_coord_y1: 
								cov_mat_coord_y2] = cov_xy * r0_scale

							self.covariance_matrix[layer_n, cov_mat_coord_x1: cov_mat_coord_x2, 
								cov_mat_coord_y1 + self.n_subap[wfs_j]: cov_mat_coord_y2 + 
								self.n_subap[wfs_j]] = cov_xy * r0_scale

						self.covariance_matrix[layer_n] = mirror_covariance_matrix(self.covariance_matrix[layer_n], 
							self.n_subap)


					if self.huge_matrix==True:
						r0_scale *= (L0[layer_n]/r0[layer_n])**(5./3.)

						cov_mat[cov_mat_coord_x1: cov_mat_coord_x2, cov_mat_coord_y1: 
							cov_mat_coord_y2] = cov_xx * r0_scale

						cov_mat[cov_mat_coord_x1 + self.n_subap[wfs_i]: cov_mat_coord_x2 + 
							self.n_subap[wfs_i], cov_mat_coord_y1 + self.n_subap[wfs_j]: 
							cov_mat_coord_y2 + self.n_subap[wfs_j]] = cov_yy * r0_scale

						if self.matrix_xy==True:
							cov_mat[cov_mat_coord_x1 + self.n_subap[wfs_i]: cov_mat_coord_x2 + 
								self.n_subap[wfs_i], cov_mat_coord_y1: 
								cov_mat_coord_y2] = cov_xy * r0_scale
							
							cov_mat[cov_mat_coord_x1: cov_mat_coord_x2, cov_mat_coord_y1 + 
								self.n_subap[wfs_j]: cov_mat_coord_y2 + 
								self.n_subap[wfs_j]] = cov_xy * r0_scale

			if self.huge_matrix==True:
				self.covariance_matrix += mirror_covariance_matrix(cov_mat, self.n_subap)


	def map_aligned_covariance(self, flatMap_covXX, flatMap_covYY, flatMap_covXY, wfs1_n_subap, wfs2_n_subap):
		self.fill_xx[self.flatMM==1] = flatMap_covXX
		cov_xx = (self.mm * self.fill_xx)[self.mm==1].reshape(wfs1_n_subap, 
			wfs2_n_subap)

		self.fill_yy[self.flatMM==1] = flatMap_covYY
		cov_yy = (self.mm * self.fill_yy)[self.mm==1].reshape(wfs1_n_subap, 
			wfs2_n_subap)

		if self.matrix_xy==True:
			self.fill_xy[self.flatMM==1] = flatMap_covXY
			cov_xy = (self.mm * self.fill_xy)[self.mm==1].reshape(wfs1_n_subap, 
				wfs2_n_subap)
		else:
			cov_xy = 0.

		return cov_xx, cov_yy, cov_xy


def tt_trackMatrix_locs(size, subSize):
    """Creates a block matrix composed as three values. Number of sub-blocks
    is calculated by the integer multiple of size and subSize.

    Parameters:
        size (int): size of block matrix
        subSize (int): size of individual blocks
        values (ndarray): numpy array containing 2 values (xx, yy). 0 is assigned to xy.

    Returns:
        ndarray: track matrix."""

    track_matrix = numpy.zeros((size, size))
    ints = int(size/subSize)

    for i in range(ints):
        l = (i+1)*subSize

        for j in range(ints):
            h = (j+1)*subSize

            if i%2==0 and j%2==0:
                track_matrix[i*subSize:l, j*subSize:h] = 1

            if i%2==1 and j%2==1:
                track_matrix[i*subSize:l, j*subSize:h] = 2

    return track_matrix.astype(int)




def matrix_lgs_trackMatrix_locs(size, n_subap):
	"""Creates a block matrix composed as three values. Number of sub-blocks
	is calculated by the integer multiple of size and subSize.

	Parameters:
		size (int): size of block matrix
		subSize (int): size of individual blocks
		values (ndarray): numpy array containing 2 values (xx, yy). 0 is assigned to xy.

	Returns:
		ndarray: track matrix."""

	lgs_trackMatrix = numpy.zeros((size, size))
	n_wfs = len(n_subap)
	tracker = 1
	
	n1=0
	for i in numpy.arange(n_wfs):
		m1=0
		for j in range(i+1):
			n2 = n1 + 2 * n_subap[i]
			m2 = m1 + 2 * n_subap[j]

			lgs_trackMatrix[n1:n1+n_subap[i], m1:m1+n_subap[j]] = tracker
			tracker += 1

			lgs_trackMatrix[n1+n_subap[i]:n2, m1+n_subap[j]:m2] = tracker
			tracker += 1

			m1 += 2 * n_subap[i]
		n1 += 2 * n_subap[j]

	return mirror_covariance_matrix(lgs_trackMatrix, n_subap)




def mirror_covariance_matrix(cov_mat, n_subap):
    """Mirrors a covariance matrix around the diagonal.

    Parameters:
        cov_mat (ndarray): The covariance matrix to mirror.
        n_subap (ndarray): number of sub-apertures within each SHWFS.
    
    Returns:
        ndarray: complete covariance matrix."""

    total_slopes = cov_mat.shape[0]
    n_wfs = len(n_subap)

    n1 = 0
    for n in range(n_wfs):
        m1 = 0
        for m in range(n + 1):
            if n != m:
                n2 = n1 + 2 * n_subap[n]
                m2 = m1 + 2 * n_subap[m]

                cov_mat[m1: m2, n1: n2] = (numpy.swapaxes(cov_mat[n1: n2, 
                    m1: m2], 1, 0))

                m1 += 2 * n_subap[m]
        n1 += 2 * n_subap[n]
    return cov_mat






def calculate_wfs_separations(n_subap1, n_subap2, wfs1_positions, wfs2_positions):
    """Calculates the separation between all sub-apertures in two WFSs

    Parameters:
        n_subap1 (int): Number of sub-apertures in WFS 1.
        n_subap2 (int): Number of sub-apertures in WFS 2.
        wfs1_positions (ndarray): SHWFS.1 sub-aperture separation from centre of the telescope.
        wfs2_positions (ndarray): SHWFS.2 sub-aperture separation from centre of the telescope.

    Returns:
        ndarray: SHWFS sub-aperture separations"""

    xy_separations = numpy.zeros((n_subap1, n_subap2, 2)).astype("float64")

    for i, (x2, y2) in enumerate(wfs1_positions):
        for j, (x1, y1) in enumerate(wfs2_positions):
            xy_separations[i, j] = (x1-x2), (y1-y2)


    return xy_separations





def wfs_covariance(wfs1_diam, wfs2_diam, L0, xy_seps, trans_wfs1, trans_wfs2, styc_method, matrix_xy):
    """Calculates the covariance between 2 SHWFSs

    Parameters:
        wfs1_diam (float): Diameter of WFS 1 sub-apertures
        wfs2_diam (float): Diameter of WFS 2 sub-apertures
        L0 (ndarray): L0 profile (m).
        xy_seps (ndarray): SHWFS sub-aperture separations.
        trans_wfs1 (float): translation of SHWFS.1 sub-aperture separation due to GS position and layer altitude.
        trans_wfs2 (float): translation of SHWFS.2 sub-aperture separation due to GS position and layer altitude.
        styc_method (bool): use styc method of analytically generating covariance.
        matrix_xy (bool): generate orthogonal covariance.

    Returns:
        ndarray: xx spatial covariance
        ndarray: yy spatial covariance
        ndarray: xy spatial covariance (if matrix_xy==True)"""

    xy_separations = xy_seps + (trans_wfs2 - trans_wfs1)
    # print(trans_wfs1, trans_wfs2)

    if styc_method==True:
        # print("Min separation: {}".format(abs(xy_separations).min()))
        cov_xx = compute_covariance_xx(xy_separations, wfs1_diam, wfs2_diam, L0)
        cov_yy = compute_covariance_yy(xy_separations, wfs1_diam, wfs2_diam, L0)
        
        if matrix_xy==True:
            cov_xy = compute_covariance_xy(xy_separations, wfs1_diam, wfs2_diam, L0)
        else:
            cov_xy = 0.
        return cov_xx, cov_yy, cov_xy

    else:
        raise Exception('Only styc technique available...for now!')






def compute_covariance_xx(separation, subap1_diam, subap2_diam, L0):
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

    x1 = separation[...,1] + (subap1_diam - subap2_diam) * 0.5
    r1 = numpy.array(numpy.sqrt(x1**2 + separation[...,0]**2))

    x2 = separation[...,1] - (subap1_diam + subap2_diam) * 0.5
    r2 = numpy.array(numpy.sqrt(x2**2 + separation[...,0]**2))

    x3 = separation[...,1] + (subap1_diam + subap2_diam) * 0.5
    r3 = numpy.array(numpy.sqrt(x3**2 + separation[...,0]**2))

    cov_xx = (-2 * structure_function_vk(r1, L0)
            + structure_function_vk(r2, L0)
            + structure_function_vk(r3, L0))
    # print(cov_xx*(L0**(5./3.)))
    # print(cov_xx)

    return cov_xx






def compute_covariance_yy(separation, subap1_diam, subap2_diam, L0):
    """Calculates yy covariance - y-axis covariance between centroids - for one turbulent 
    layer in a single GS combination.
    
    Parameters:
        separation (ndarray): x and y sub-aperture separations (m).
        subap1_diam (float): diameter of SHWFS sub-apertures in SHWFS.1.
        subap2_diam (float): diameter of SHWFS sub-apertures in SHWFS.2.
        L0 (float): L0 value for turbulent layer (m).
    
    Returns:
        ndarray: yy spatial covariance"""    

    y1 = separation[...,0] + (subap1_diam - subap2_diam) * 0.5
    r1 = numpy.array(numpy.sqrt(separation[...,1]**2 + y1**2))

    # print(r1)
    # print(subap1_diam-subap2_diam)
    # print(separation[...,0])

    y2 = separation[...,0] - (subap1_diam + subap2_diam) * 0.5
    r2 = numpy.array(numpy.sqrt(separation[...,1]**2 + y2**2))

    y3 = separation[...,0] + (subap1_diam + subap2_diam) * 0.5
    r3 = numpy.array(numpy.sqrt(separation[...,1]**2 + y3**2))

    cov_yy = (-2 * structure_function_vk(r1, L0)
        + structure_function_vk(r2, L0)
        + structure_function_vk(r3, L0))

    return cov_yy






def compute_covariance_xy(seperation, subap1_diam, subap2_diam, L0):
    """Calculates xy covariance - covariance between orthogonal centroids - for one turbulent 
    layer in a single GS combination.
    
    Parameters:
        separation (ndarray): x and y sub-aperture separations (m).
        subap1_diam (float): diameter of SHWFS sub-apertures in SHWFS.1.
        subap2_diam (float): diameter of SHWFS sub-apertures in SHWFS.2.
        L0 (float): L0 value for turbulent layer (m).
    
    Returns:
        ndarray: xy spatial covariance"""

    s0 = numpy.sqrt((subap1_diam*0.5)**2 + (subap2_diam*0.5)**2)
    # print(subap1_diam, subap2_diam)

    x1 = seperation[..., 0] + s0
    y1 = seperation[..., 1] - s0
    r1 = numpy.sqrt(x1**2 + y1**2)

    x2 = seperation[..., 0] - s0
    y2 = seperation[..., 1] + s0
    r2 = numpy.sqrt(x2**2 + y2**2)

    x3 = seperation[..., 0] + s0
    y3 = seperation[..., 1] + s0
    r3 = numpy.sqrt(x3**2 + y3**2)

    x4 = seperation[..., 0] - s0
    y4 = seperation[..., 1] - s0
    r4 = numpy.sqrt(x4**2 + y4**2)

    Cxy = (structure_function_vk(r1, L0)
            + structure_function_vk(r2, L0)
            - structure_function_vk(r3, L0)
            - structure_function_vk(r4, L0)
           )

    return -Cxy * 0.5






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

    # print(dprf0)    

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
    # wavelengthma_R(5./6)*2^(-1./6)
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
    K_a(x)=1/2 \sum_{n=0}^\infty \frac{(-1)^n}{n!}\left(\wavelengthma(-n-a) (x/2)^{2n+a} + \wavelengthma(-n+a) (x/2)^{2n-a} \right), with a = 5/6.

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
	tel_diam = 4.2
	obs_diam = 1.
	subap_diam = numpy.array([0.6]*n_wfs)
	gs_alt = numpy.array([0]*n_wfs)
	gs_pos = numpy.array([[0.,-20.], [0.,20]])
	wavelength = numpy.array([500e-9]*n_wfs)
	n_layer = 1
	layer_alt = numpy.array([9282.])

	r0 = numpy.array([0.10]*n_layer)
	L0 = numpy.array([25.]*n_layer)
	tt_track = numpy.array([0.0,0.0])
	shwfs_shift = numpy.array(([0,0],[0,0]))
	shwfs_rot = numpy.array([0,0])
	n_subap = numpy.array([36]*n_wfs)
	nx_subap = numpy.array([7]*n_wfs)
	pupil_mask = make_pupil_mask('circle', n_subap, 7, obs_diam, tel_diam)
	matrix_region_ones = numpy.ones((n_subap[0], n_subap[0]))
	mm, mmc, md = get_mappingMatrix(pupil_mask, matrix_region_ones)
	lgs_track = numpy.array(([1.,2.],[3., 4.], [5., 6.], [7.,8.],[9., 10.], [11., 12.]))


	params = covariance_matrix(pupil_mask, subap_diam, wavelength, tel_diam, n_subap, gs_alt, gs_pos, 
		n_layer, layer_alt, r0, L0, styc_method=True, wind_profiling=False, tt_track_present=True, 
		lgs_track_present=False, offset_present=False, fit_layer_alt=False, fit_tt_track=False, 
		fit_lgs_track=False, fit_offset=False, fit_L0=False, matrix_xy=True, huge_matrix=False, 
		l3s1_transform=False, l3s1_matrix=None, remove_tt=False, remove_tt_matrix=None)

	s = time.time()
	cov_mat = params._make_covariance_matrix_(layer_alt, r0, L0, tt_track=tt_track, lgs_track=lgs_track, shwfs_shift=shwfs_shift, shwfs_rot=shwfs_rot)
	f = time.time()
	print('Time Taken: {}'.format(f-s))
	cov_map = covMap_fromMatrix(cov_mat, n_wfs, nx_subap, n_subap, pupil_mask, 'x and y', mm, mmc, md)

	# pyplot.figure()
	# pyplot.imshow(cov_mat)
	# pyplot.colorbar()

	cov_map = covMap_fromMatrix(cov_mat, n_wfs, nx_subap, n_subap, pupil_mask, 'x and y', mm, mmc, md)


	c = numpy.hstack(( numpy.rot90(cov_map[:,:13], 3), numpy.rot90(cov_map[:,13:], 3) ))

	pyplot.figure('fresh')
	pyplot.imshow(c)

	# pyplot.figure('fresh xx')
	# pyplot.plot(numpy.arange(-6,7), cov_map[:, 19]/cov_map[:, 19].max())

    # t, tt_matrix = removeTT_cents(numpy.zeros((100, int(n_subap.sum()*n_wfs))), n_subap, n_wfs)
    # rem_mat = numpy.matmul(numpy.matmul(tt_matrix, cov_mat), tt_matrix.T) 
    # cov_map = covMap_fromMatrix(rem_mat, n_wfs, nx_subap, n_subap, pupil_mask, 'x and y', mm, mmc, md)
    # pyplot.figure('removed')
    # pyplot.imshow(cov_map)
    # pyplot.figure('removed xx')
    # pyplot.plot(numpy.arange(-6,7), cov_map[:, 19]/cov_map[:, 19].max())