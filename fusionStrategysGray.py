import numpy as np
import math
from skimage import filters
from metrics import entr

def minimGray(coeff1, coeff2):
	"""
	Apply the minimum fusion strategy to the coefficients given in parameters
	
	coeff1 - coefficient of the RGB image
	coeff2 - coefficient of the IR image
	
	
	Returns fused coefficient 
	"""
	w, h = coeff1.shape[0], coeff1.shape[1]
	
	coeff = np.zeros(coeff1.shape)
	
	a = 0.0
	b = 0.0
	
	for i in range(w):
		for j in range(h):
			a = coeff1[i][j]
			b = coeff2[i][j]
			
			delt = abs(a) + abs(b) 
			
			if delt == 0:
				coeff[i][j] = 0.0
			elif math.isclose(a, 0.0, rel_tol=1e-3) or math.isclose(b, 0.0, rel_tol=1e-3):
				coeff[i][j] = min(a, b)
			elif (a < 0.0 and b < 0.0) or (a > 0.0 and b > 0.0):
				coeff[i][j] = (abs(b)/delt) * a + (abs(a)/delt) * b
			elif a < 0. and b > 0.:
				coeff[i][j] = (max(abs(a), abs(b))/delt) * a + (min(abs(a), abs(b))/delt) * b
			else:
				coeff[i][j] = (min(abs(a), abs(b))/delt) * a + (max(abs(a), abs(b))/delt) * b
	
	return coeff


def maximGray(coeff1, coeff2):
	"""
	Apply the maximum fusion strategy to the coefficients given in parameters
	
	coeff1 - coefficient of the RGB image
	coeff2 - coefficient of the IR image
	
	
	Returns fused coefficient 
	"""
	w, h = coeff1.shape[0], coeff1.shape[1]
	
	coeff = np.zeros(coeff1.shape)
	
	a = 0.0
	b = 0.0
	
	for i in range(w):
		for j in range(h):
			a = coeff1[i][j]
			b = coeff2[i][j]
			delt = abs(a) + abs(b)
			
			if delt == 0:
				coeff[i][j] = 0.0
			elif math.isclose(a, 0.0, rel_tol=1e-2) or math.isclose(b, 0.0, rel_tol=1e-2):
				coeff[i][j] = max(a, b)
			elif (a < 0. and b < 0.) or (a > 0. and b > 0.) :
				coeff[i][j] = (abs(a)/delt) * a + (abs(b)/delt) * b
			elif a < 0. and b > 0.:
				coeff[i][j] = (min(abs(a), abs(b))/delt) * a + (max(abs(a), abs(b))/delt) * b
			else:
				coeff[i][j] = (max(abs(a), abs(b))/delt) * a + (min(abs(a), abs(b))/delt) * b

	return coeff


def coeffsEntropyGray(coeff1, coeff2):
	"""
	Apply the entropy fusion strategy to the coefficients given in parameters
	
	coeff1 - coefficient of the RGB image
	coeff2 - coefficient of the IR image
	
	
	Returns fused coefficient 
	"""
	w, h = coeff1.shape
	
	sum1 = coeff1
	sum2 = coeff2
	
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
	
def MACD(coeff1, coeff2):
	"""
	Apply the MACD fusion strategy to the coefficients given in parameters
	
	coeff1 - coefficient of the RGB image
	coeff2 - coefficient of the IR image
	
	
	Returns fused coefficient 
	"""
	A1 = Activity(coeff1)
	A2 = Activity(coeff2)
	
	M = Match(coeff1, coeff2)

	D = Decision(A1, A2, M)
	
	
	result = np.empty(coeff1.shape)
	
	w, h = result.shape[0], result.shape[1]
	
	for i in range(w):
		for j in range(h):
			if (D[i][j] == 0.):
				if (A1[i][j] > A2[i][j]):
					result[i][j] = coeff1[i][j]
				else:
					result[i][j] = coeff2[i][j]
			else:
				result[i][j] = D[i][j] * coeff1[i][j] + (1 - D[i][j]) * coeff2[i][j]
	
	return result
	
def Activity(coeff):
	return np.absolute(coeff)

def Match(coeff1, coeff2):
	"""
	Apply the Match step of the MACD fusion strategy to the coefficients given in parameters
	
	coeff1 - coefficient of the RGB image
	coeff2 - coefficient of the IR image
	
	
	Returns match coefficient
	"""
	result = np.empty(coeff1.shape)
	w, h = result.shape[0], result.shape[1]
	
	mult = np.empty(coeff1.shape)
	mult_res = 0.0
	
	mult = np.where((abs(coeff1)**2 + abs(coeff2)**2 == 0.0), coeff1*coeff2, coeff1*coeff2/(abs(coeff1)**2 + abs(coeff2)**2))

	for i in range(w):
		for j in range(h):
			result[i][j] = window(coeff1, coeff2, i, j, mult)
				
	return result

def Decision(coeff1, coeff2, m, fract = 0.5):
	"""
	Apply the Decision step of the MACD fusion strategy to the coefficients given in parameters
	
	coeff1 - activity array of the RGB image
	coeff2 - activity array of the IR image
	m 	   - match array 
	fract  - threshold between pure maximum and weighted max
	
	
	Returns decision coefficient 
	"""
	
	w, h = coeff1.shape[0], coeff1.shape[1]
	
	res = np.zeros(coeff1.shape)
	mean = np.mean(m)
	
	for i in range(w):
		for j in range(h):
			delta = coeff1[i][j] + coeff2[i][j]
			if (delta == 0. or m[i][j] <= 1e-5):
				res[i][j] = 0
			else:
				if (m[i][j] > fract*mean):
					res[i][j] = 1/2
				else:
					res[i][j] = coeff1[i][j]/delta
	
	return res;
	
def window(coeff1, coeff2, a, b, mult, size = 5):
	"""
	Compute a normalized correlation averaged in a neighbourhood of a pixel
	
	coeff1 - activity array of the RGB image
	coeff2 - activity array of the IR image
	a	   - first index of the pixel
	b 	   - second index of the pixel
	c 	   - third index of the pixel
	mult   - array containing the multiplication of corresponding elements from coeff1 and coeff2
	size   - size of the window
	"""
	w, h = coeff1.shape
	res = 0
	for i in range(a - size, a + size):
		for j in range(b - size, b + size):
			if (i >= 0 and i < w and j >= 0 and j < h):
				res += mult[i][j]
				
	return res
	
def edgeDetection(coeff1, coeff2):
	"""
	Fuse two coefficients by first applying a Sobel filter, then taking a weighted average
	of both using their entropy
	
	coeff1 	- first coefficient
	coeff2 	- second coefficient
	"""
	edges_RGB = filters.sobel(coeff1)
	edges_IR = filters.sobel(coeff2)
	
	entropy_RGB = entr(edges_RGB)
	entropy_IR = entr(edges_IR)
	
	entropy_sum = entropy_RGB + entropy_IR
	
	if (entropy_sum > 0.):
		return entropy_RGB/entropy_sum * coeff1 + entropy_IR/entropy_sum * coeff2
	else:
		return (coeff1 + coeff2)/2.

def deviation(coeff1, coeff2, window_size = 8):
	"""
	Fuse two coefficients by first dividing the coefficients, then using the 
	standard deviation criterion
	
	coeff1 	- first coefficient
	coeff2 	- second coefficient
	"""
	w, h = coeff1.shape
	res = np.zeros(coeff1.shape)
	
	Stdr = 0.
	Stdi = 0.
	
	sum_Std = 0.
	for i in range(0, w, window_size):
		for j in range(0, h, window_size):
			RGB = coeff1[i:min(i + window_size, w), j:min(j + window_size, h)]
			IR = coeff2[i:min(i + window_size, w), j:min(j + window_size, h)]
			
			Stdr = np.std(RGB)
			Stdi = np.std(IR)
			
			sum_Std = Stdr + Stdi
			
			if (sum_Std == 0.):
				res[i:min(i + window_size, w), j:min(j + window_size, h)] = (RGB + IR)/2.
			else:
				res[i:min(i + window_size, w), j:min(j + window_size, h)] = Stdr/sum_Std * RGB + Stdi/sum_Std * IR
	
	return res
