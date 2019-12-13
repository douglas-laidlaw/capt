
# -*- coding: utf-8 -*-
import numpy
import numba
import canaryConfig
import aofConfig
import myFont
from aotools.functions import circle
from matplotlib import pyplot; pyplot.ion()

# @numba.jit
def matcov(
    nWfs, pupilMask, telDiam, nSubaps, nxSubaps, subapDiam, gsAlt, gsPos, nLayers,
    layerHeights, r0, L0, data, gsMag, covMatPart=0 , pupilOffset=None,
    wfsRot=None):
	"""
	Generate a covariance matrix for a given parameter set.
	Uses code ported from LESIA, Gendron and co. from STYC, CANARY.

	Parameters:
	nWfs (int): Number of wavefront sensors present.
	nSubaps (ndarray): Number of sub-apertures for each WFS
	nxSubaps (ndarray): Number of sub-apertures in one direction for each WFS
	subapDiam (ndarray): The diameter of the sub-apertures for each WFS in metres
	gsAlt (ndarray): Reciprocal (1/metres) of the Guide star alitude for each WFS
	GSPos (ndarray): X,Y position of each WFS in arcsecs. Array shape (Wfs, 2)
	nLayers (int): The Number of atmospheric turbulence layers
	layerHeights (ndarray): The alitude of each turbulence layer in metres
	r0 (ndarray): The strength of each layer in metres
	L0 (ndarray): The outer-scale of each layer in metres
	data (ndarray): An empty array to fill, of size (2*totalSubaps, 2*totalSubaps)

	"""

	# print nWfs
	# print pupilMask
	# print telDiam
	# print nSubaps
	# print nxSubaps
	# print subapDiam
	# print gsAlt
	# print gsPos
	# print nLayers
	# print layerHeights
	# print r0
	# print L0
	# print gsMag

	subapPos = (numpy.array(numpy.where(pupilMask==1)).T * telDiam/nxSubaps[0]).T
	subapPos = numpy.tile(subapPos, (1,nWfs))


	subapLayerPos = subap_position(
		nWfs, nSubaps, nxSubaps, gsAlt, gsPos, subapPos, nLayers,
		layerHeights, gsMag, pupilOffset, wfsRot)

	# d=off
	# print 'hi', subapLayerPos
	# d=ki
	# Rescale the projected suze of all subapertures at the different altitudes

	subapSizes = numpy.zeros((nWfs, nLayers))
	for n in range(nWfs):
		for l in range(nLayers):
			# subapSizes[n, l] = subapDiam[n] * (1. - gsAlt[n]*layerHeights[l])
			if gsAlt[n]!=0:
				subapSizes[n, l] = subapDiam[n] * (1. - layerHeights[l]/gsAlt[n])
			else:
				subapSizes[n, l] = subapDiam[n]

	# Computation of the covariance matrix
	#####################################

	lambda2 = pow(0.5e-6/2./3.1415926535,2);

	# Truth Sensor no
	ts = nWfs - 1

	ioff = joff = 0
	units = numpy.zeros(nLayers)

	# Find the total number of slopes
	totSlopes = 2 * nSubaps.sum()

	# pyplot.imshow(subapLayerPos[1,:,:,0])
	# d=p
	# Loop over WFS 1
	for m in range(0, nWfs):
		Ni = nSubaps[m] + ioff

		# Loop over WFS 2. Dont loop over all WFSs, use symmetry
		for n in range(0, m+1):

			off_XY = nSubaps[n]
			off_YX = nSubaps[m] * totSlopes
			off_YY = off_XY + off_YX

			Nj = nSubaps[n] + joff

			kk = 1./( subapDiam[m] * subapDiam[n])

			for l in range(0, nLayers):
				# units[l] = ((kk/0.42))*r0[l]**(-5./3.)
				units[l] = kk * lambda2 * r0[l]**(-5./3.)
			# Loop through subap i on WFS 1
			for i in range(ioff, Ni):
				# Loop through subap j on WFS 2
				for j in range(joff, Nj):

					caa_xx = 0
					caa_yy = 0
					caa_xy = 0
					# Loop through altitude layers
					for l in range(0, nLayers):
						# Check layer is not above LGS
						if subapSizes[m, l]>0 and subapSizes[n,l]>0:
							# Distances in x and y between the subapertures i and j

							du = (subapLayerPos[0, m, i-ioff, l]
									- subapLayerPos[0, n, j-joff, l])
							dv = (subapLayerPos[1, m, i-ioff, l]
									- subapLayerPos[1,n, j-joff, l])

							# print du, dv

							s1 = subapSizes[m, l] * 0.5
							s2 = subapSizes[n, l] * 0.5

							# print(s1, s2)
							ac = s1 - s2
							ad = s1 + s2
							bc = -ad
							bd = -ac

							cov = compute_cov(
									du, dv, ac, ad, bc, bd, s1, s2, L0[l],
									units[l])

							caa_xx += cov[1]
							caa_yy += cov[0]
							caa_xy += cov[2]

					# print("i: {}, j: {}, i0: {}, NL: {}".format(i, j,i0, NL))
					# print("off_XY: {}, off_YX: {}, off_YY:{}".format(
					#		 off_XY, off_YX, off_YY))
					# print("du: {:.4f}, dv: {:4f}".format(du,dv))
					# print("caa_xx: {:.3f}, caa_yy: {:.3f}, caa_xy: {:.3f}\n".format(
					#		 caa_xx, caa_yy, caa_xy ))


					data[i, j] = caa_xx
					data[i+nSubaps[n], j] = caa_xy
					data[i, j+nSubaps[m]] = caa_xy
					data[i+nSubaps[n], j+nSubaps[m]] = caa_yy


			joff = joff + 2*nSubaps[n]
		ioff += 2*nSubaps[m]
		joff = 0

	data = mirrorCovMat(data, nSubaps)

	return data


# @numba.jit
def mirrorCovMat(covMat, nSubaps):

	totSlopes = covMat.shape[0]
	nWfs = nSubaps.shape[0]

	n1 = 0
	for n in range(nWfs):
		m1 = 0
		for m in range(n+1):
			if n!=m:
				n2 = n1 + 2*nSubaps[n]
				m2 = m1 + 2*nSubaps[m]

				nn1 = totSlopes - 2*nSubaps[n] - n1
				nn2 = nn1 + 2*nSubaps[n]

				mm1 = totSlopes - 2*nSubaps[m] - m1
				mm2 = mm1 + 2*nSubaps[m]

				# covMat[nn1: nn2, mm1: mm2] = (
				# 		numpy.swapaxes(covMat[n1: n2, m1: m2], 1, 0)
				# 		)

				covMat[nn1: nn2, mm1: mm2] = (
						numpy.swapaxes(covMat[mm1: mm2, nn1: nn2], 1, 0)
						)

				m1+=2*nSubaps[m]
		n1+=2*nSubaps[n]
	return covMat

# @numba.jit
def subap_position(
		nWfs, nSubaps, nxSubaps, gsAlt, gsPos, subapPos,
		nLayers, layerHeights, gsMag, pupilOffset=None, wfsRot=None):

	rad = numpy.pi / 180.


	# u, v arrays, contain subap coordinates of all WFSs
	# CURRENTLY BROKEN FOR WFSs WITH DIFFERENT nSUBAPS!!
	u = numpy.zeros((nWfs, nSubaps[0], nLayers))
	v = numpy.zeros((nWfs, nSubaps[0], nLayers))

	for l in range(0, nLayers):
		ioff = 0

		for n in range(0, nWfs):

			# dX = gsPos[n, 0] * layerHeights[l]
			# dY = gsPos[n, 1] * layerHeights[l]

			dY = gsPos[n, 0] * layerHeights[l]
			dX = gsPos[n, 1] * layerHeights[l]
			# print dX, dY
			# i=iff

			rr = 1.
			if gsAlt[n]!=0:
				rr = 1. - layerHeights[l] / gsAlt[n]
			print(rr)

			# Magnification
			if gsMag[n]!=None:
				G = float(gsMag[n]) / nxSubaps[n]
			else:
				G = 1./nxSubaps[n]

			# If Rotation angle required, find angle in radians
			if wfsRot[n]!=None:
				th = wfsRot[n] * rad
			else:
				th = 0

			for i in range(nSubaps[n]):
				xtp = subapPos[0, ioff + i] * G
				ytp = subapPos[1, ioff + i] * G

				# Correct for any rotation
				uu = xtp * numpy.cos(th) - ytp * numpy.sin(th)
				vv = xtp * numpy.sin(th) + ytp * numpy.cos(th)

				# If any pupil offset is required, adjust position
				if numpy.any(pupilOffset):
					uu += pupilOffset[0, n]
					vv += pupilOffset[1, n]

				u[n, i, l] = uu*rr + dX
				v[n, i, l] = vv*rr + dY

				# print uu, rr, dX
				# print uu*rr + dX
				# d=oh

			ioff += nSubaps[n]

	return numpy.array([u, v])

# @numba.jit(nopython=True)
def compute_cov(du, dv, ac, ad, bc, bd, s1, s2, L0, units):
	"""
	<du> & <dv>: X and Y coordinates of the distance between the two considered subapertures.
	<ac> & <ad> & <bc> & <bd>  : precomputed values
	<s1> & <s2>				: half size of each subapertures
	<L0>				  :
	<units>					:
	Computes the XX, XY and YY covariance values for two subapertures.
	"""

	cov_xx = cov_yy = cov_xy = 0

	cov_xx = cov_XX(du, dv, ac, ad, bc, bd, L0)
	cov_xx *= 0.5

	cov_yy = cov_YY(du, dv, ac, ad, bc, bd, L0)
	cov_yy *= 0.5

	s0 = numpy.sqrt(s1**2 + s2**2)

	cov_xy = cov_XY(du, dv, s0, L0)

	cov_xy *= 0.25

	if s1>s2:
		cc = 1 - (float(s2)/s1)
	else:
		cc = 1 - (float(s1)/s2)

	cov_xy *= (1. - cc**2)
	# print(units)


	cov_xx *= units
	cov_yy *= units
	cov_xy *= units

	# cov = numpy.zeros(3, dtype="float64")
	cov = [0., 0., 0.]
	cov[0] = cov_xx
	cov[1] = cov_yy
	cov[2] = cov_xy

	return cov


# @numba.jit(nopython=True)
def cov_XX(du, dv, ac, ad, bc, bd, L0):
	"""
	Compute the XX-covariance with the distance sqrt(du2+dv2). DPHI is
	precomputed on tabDPHI.
	"""

	cov =   (-1 * DPHI(du+ac, dv, L0)
			+ DPHI(du+ad, dv, L0)
			+ DPHI(du+bc, dv, L0)
			- DPHI(du+bd, dv, L0)
			)

	# print( numpy.sqrt((du+ac)**2 + dv**2) )
	# print(du)
	return cov


# @numba.jit(nopython=True)
def cov_YY(du, dv, ac, ad, bc, bd, L0):
	"""
	Compute the YY-covariance with the distance sqrt(du2+dv2). DPHI is
	precomputed on tabDPHI.
	"""

	cov =   (-1 * DPHI(du, dv+ac, L0)
			+ DPHI(du, dv+ad, L0)
			+ DPHI(du, dv+bc, L0)
			- DPHI(du, dv+bd, L0)
			)

	return cov

# @numba.jit(nopython=True)
def cov_XY(du, dv, s0, L0):
	"""
	Compute the XY-covariance with the distance sqrt(du2+dv2). DPHI is precomputed on tabDPHI.
   """

	cov = (  -1*DPHI(du + s0, dv - s0, L0)
			+ DPHI(du + s0, dv + s0, L0)
			+ DPHI(du - s0, dv - s0, L0)
			- DPHI(du - s0, dv + s0, L0)
			)
	return cov

# @numba.jit(nopython=True)
def DPHI(x, y, L0):
	"""
	Parameters:
		x (float): Seperation between apertures in X direction
		y (float): Separation between apertures in Y direction
		L0 (float): Outer scale

	Computes the phase structure function for a separation (x,y).
	The r0 is not taken into account : the final result of DPHI(x,y,L0)
	has to be scaled with r0^-5/3, with r0 expressed in meters, to get
	the right value.
	"""
	r = numpy.sqrt(x**2 + y**2)

	return rodconan(r, L0, 10)

# @numba.jit(nopython=True)
def rodconan(r, L0, k):
	"""
	The phase structure function is computed from the expression
	Dphi(r) = k1  * L0^(5./3) * (k2 - (2.pi.r/L0)^5/6 K_{5/6}(2.pi.r/L0))

	For small r, the expression is computed from a development of
	K_5/6 near 0. The value of k2 is not used, as this same value
	appears in the series and cancels with k2.
	For large r, the expression is taken from an asymptotic form.

	"""
	# k1 is the value of :
	# 2*gamma_R(11./6)*2^(-5./6)*pi^(-8./3)*(24*gamma_R(6./5)/5.)^(5./6);
	k1 = 0.1716613621245709486
	dprf0 = (2*numpy.pi/L0)*r
	# d=off
	# print(dprf0)

	if dprf0 > 4.71239:
		res = asymp_macdo(dprf0)
	else:
		res = -macdo_x56(dprf0)

	res *= k1 * L0**(5./3)

	return res


# @numba.jit(nopython=True)
def asymp_macdo(x):
	"""
	Computes a term involved in the computation of the phase struct
	function with a finite outer scale according to the Von-Karman
	model. The term involves the MacDonald function (modified bessel
	function of second kind) K_{5/6}(x), and the algorithm uses the
	asymptotic form for x ~ infinity.
	Warnings :
	- This function makes a doubleing point interrupt for x=0
	and should not be used in this case.
	- Works only for x>0.
	"""

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

# @numba.jit(nopython=True)
def macdo_x56(x):
	"""
	Computation of the function
	f(x) = x^(5/6)*K_{5/6}(x)
	using a series for the esimation of K_{5/6}, taken from Rod Conan thesis :
	K_a(x)=1/2 \sum_{n=0}^\infty \frac{(-1)^n}{n!}
	\left(\Gamma(-n-a) (x/2)^{2n+a} + \Gamma(-n+a) (x/2)^{2n-a} \right) ,
	with a = 5/6.

	Setting x22 = (x/2)^2, setting uda = (1/2)^a, and multiplying by x^a,
	this becomes :
	x^a * Ka(x) = 0.5 $ -1^n / n! [ G(-n-a).uda x22^(n+a) + G(-n+a)/uda x22^n ]
	Then we use the following recurrence formulae on the following quantities :
	G(-(n+1)-a) = G(-n-a) / -a-n-1
	G(-(n+1)+a) = G(-n+a) /  a-n-1
	(n+1)! = n! * (n+1)
	x22^(n+1) = x22^n * x22
	and at each iteration on n, one will use the values already computed at step (n-1).
	The values of G(a) and G(-a) are hardcoded instead of being computed.

	The first term of the series has also been skipped, as it
	vanishes with another term in the expression of Dphi.
	"""

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


if __name__=='__main__':
	telConfig = canaryConfig.telescopeParams

	GSPOS = numpy.array(([0,-20],[0,20])) * (1./3600) * (numpy.pi/180.)
	TEL_DIAM = telConfig["Telescope"]["diameter"]
	xSubaps = telConfig["WFS"]["nxSubaps"]

	NWFS = GSPOS.shape[0]
	NXSUBAPS = numpy.array([xSubaps]*NWFS)
	SUBAPDIAM = numpy.array([telConfig["WFS"]["subapDiam"]]*NWFS)
	GSALT = numpy.array([30000]*NWFS)
	GSTYPE = numpy.array([1]*NWFS)
	PUPILOFFSET = numpy.array(([0,0],[0,0]))
	PUPILMAG = numpy.array([NXSUBAPS[0]]*NWFS)
	PUPILROT = numpy.array([0]*NWFS)
	# PUPILROT[1]=10
	OBS = 0.285
	NCPU = 1
	PART = 0
	PUPIL_MASK = telConfig["WFS"]["pupilMask"]
	NSUBAPS = numpy.array([int(PUPIL_MASK.sum())]*NWFS)
	waveL = 500e-9
	gam = numpy.array([waveL]*NWFS)


	NLAYERS = 1
	r0 = numpy.array([0.1]*NLAYERS)
	L0 = numpy.array([10.]*NLAYERS)
	# LAYERHEIGHTS = numpy.array([0])
	LAYERHEIGHTS = numpy.array([0])

	covMat = numpy.zeros((2*NSUBAPS.sum(), 2*NSUBAPS.sum()), dtype="float64")
	c = matcov(NWFS, PUPIL_MASK, TEL_DIAM, NSUBAPS, NXSUBAPS, SUBAPDIAM, GSALT, GSPOS, NLAYERS, LAYERHEIGHTS, r0, L0, covMat, PUPILMAG, wfsRot=PUPILROT, pupilOffset=PUPILOFFSET) * ((180./numpy.pi) * 3600)**2
	pyplot.figure()
	pyplot.imshow(c)