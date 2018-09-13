import numpy




def calc_r0(Cn2, lam):
	"""Calculate r0 from Cn2

	Parameters:
	Cn2 (ndarray): Cn2 profile (m^{-1/3}).
	lam (float): wavelength (nm).

	Returns:
	ndarray: r0."""

	r0 = (((((lam/(2*numpy.pi))**2)*(1/0.42))**-1.)*Cn2)**(-3./5.)
	return r0




def calc_Cn2(r0, lam):
	"""Calculate Cn2 from r0.

	Parameters:
	r0 (ndarray): r0 profile (m).
	lam (float): wavelength (m).

	Returns:
	ndarray: Cn2."""

	Cn2 = ((r0**(-5./3.))*((lam/(2*numpy.pi))**2)*(1/0.42))
	return Cn2





if __name__=='__main__':
	lam = 0.5e-6
	r0 = 0.1
	print(calc_Cn2(r0, lam))
	print(calc_R0(calcCn2(r0, lam), lam))
