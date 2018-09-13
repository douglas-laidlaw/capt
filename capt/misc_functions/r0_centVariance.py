import numpy
from astropy.io import fits
from matplotlib import pyplot; pyplot.ion()

def r0_centVariance(slopes, wavelength, subap_diam):

	slopes = slopes * (1./3600) * (numpy.pi/180)

	slope_variance = numpy.mean(numpy.var(slopes, 0))
	r0 = 0.336 * (wavelength**(6./5.)) * (subap_diam**(-1./5.)) * (slope_variance**(-3./5.))

	return r0