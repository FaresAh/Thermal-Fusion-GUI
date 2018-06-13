import numpy as np
import math
from skimage import filters
from metrics import entr


def minim(coeff1, coeff2):
	"""
	Apply the minimum fusion strategy to the coefficients given in parameters
	
	coeff1 - coefficient of the RGB image
	coeff2 - coefficient of the IR image
	
	
	Returns fused coefficient 
	"""
	w, h, c = coeff1.shape
	
	coeff = np.zeros(coeff1.shape)
	
	a = 0.0
	b = 0.0
	
	for i in range(w):
		for j in range(h):
			for k in range(c):
				a = coeff1[i][j][k]
				b = coeff2[i][j][k]
				
				delt = abs(a) + abs(b) 
				
				if delt == 0:
					coeff[i][j][k] = 0.0
				elif math.isclose(a, 0.0, rel_tol=1e-3) or math.isclose(b, 0.0, rel_tol=1e-3):
					coeff[i][j][k] = min(a, b)
				elif (a < 0.0 and b < 0.0) or (a > 0.0 and b > 0.0):
					coeff[i][j][k] = (abs(b)/delt) * a + (abs(a)/delt) * b
				elif a < 0. and b > 0.:
					coeff[i][j][k] = (max(abs(a), abs(b))/delt) * a + (min(abs(a), abs(b))/delt) * b
				else:
					coeff[i][j][k] = (min(abs(a), abs(b))/delt) * a + (max(abs(a), abs(b))/delt) * b
	
	return coeff


def maxim(coeff1, coeff2):
	"""
	Apply the maximum fusion strategy to the coefficients given in parameters
	
	coeff1 - coefficient of the RGB image
	coeff2 - coefficient of the IR image
	
	
	Returns fused coefficient 
	"""
	w, h, c = coeff1.shape
	
	coeff = np.zeros(coeff1.shape)
	
	a = 0.0
	b = 0.0
	
	for i in range(w):
		for j in range(h):
			for k in range(c):
				a = coeff1[i][j][k]
				b = coeff2[i][j][k]
				delt = abs(a) + abs(b)
				
				if delt == 0:
					coeff[i][j][k] = 0.0
				elif math.isclose(a, 0.0, rel_tol=1e-2) or math.isclose(b, 0.0, rel_tol=1e-2):
					coeff[i][j][k] = max(a, b)
				elif (a < 0. and b < 0.) or (a > 0. and b > 0.) :
					coeff[i][j][k] = (abs(a)/delt) * a + (abs(b)/delt) * b
				elif a < 0. and b > 0.:
					coeff[i][j][k] = (min(abs(a), abs(b))/delt) * a + (max(abs(a), abs(b))/delt) * b
				else:
					coeff[i][j][k] = (max(abs(a), abs(b))/delt) * a + (min(abs(a), abs(b))/delt) * b

	return coeff


def coeffsEntropy(coeff1, coeff2):
	"""
	Apply the entropy fusion strategy to the coefficients given in parameters
	
	coeff1 - coefficient of the RGB image
	coeff2 - coefficient of the IR image
	
	More details available at https://www.iiitd.edu.in/~mayank/ICAPR09-MedicalFusion.pdf
	
	
	Returns fused coefficient 
	"""
	w, h, c = coeff1.shape
	
	sum1 = np.empty(coeff1.shape).astype(np.float64)
	sum2 = np.empty(coeff1.shape).astype(np.float64)

	for k in range(c):
		sum1 += coeff1[c]
		sum2 += coeff2[c]
	
	entropy1, entropy2 = 0., 0.

	sum1 = sum1[sum1 > 0.]
	sum1 = sum1/sum1.sum()
	sum1 = sum1 * np.log2(sum1)
	
	sum2 = sum2[sum2 > 0.]
	sum2 = sum2/sum2.sum()
	sum2 = sum2 * np.log2(sum2)

	entropy1 = -sum1.sum()
	entropy2 = -sum2.sum()

	delt = entropy1 + entropy2
	
	if (delt == 0.):
		return (coeff1 + coeff2)/2

	return (np.multiply(entropy1/delt, coeff1) + np.multiply(entropy2/delt, coeff2))

