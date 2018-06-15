import numpy as np
import math
from skimage import filters
from metrics import entr
from skimage.measure.entropy import shannon_entropy
from scipy import ndimage

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
	return np.where(D == 0., np.maximum(coeff1, coeff2), D * coeff1 + (1 - D) * coeff2)
	
def Activity(coeff):
	return np.absolute(coeff)

def Match(coeff1, coeff2):
	"""
	Apply the Match step of the MACD fusion strategy to the coefficients given in parameters
	
	coeff1 - coefficient of the RGB image
	coeff2 - coefficient of the IR image
	
	
	Returns match coefficient
	"""
	mult = (coeff1 * coeff2) / (np.abs(coeff1)**2 + np.abs(coeff2)**2 + np.finfo(np.float32).eps)
	return ndimage.convolve(mult, np.ones((5, 5, 3)), mode='constant')

def Decision(coeff1, coeff2, m, fract = 0.5):
	"""
	Apply the Decision step of the MACD fusion strategy to the coefficients given in parameters
	
	coeff1 - activity array of the RGB image
	coeff2 - activity array of the IR image
	m 	   - match array 
	fract  - threshold between pure maximum and weighted max
	
	
	Returns decision coefficient 
	"""
	mean = np.mean(m)
	delta = coeff1 + coeff2
	return np.where((delta == 0) | (m > fract * mean), 0.5, coeff1 / (delta + np.finfo(np.float32).eps))

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
