import numpy as np
from skimage import filters
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage.measure.entropy import shannon_entropy
from scipy import ndimage

@adapt_rgb(each_channel)
def sobel_each(image):
	return filters.sobel(image)

def MACD(coeff1, coeff2):
	"""
	Apply the MACD fusion strategy to the coefficients given in parameters
	
	coeff1 - coefficient of the RGB image
	coeff2 - coefficient of the IR image
	
	
	Return fused coefficient 
	"""
	A1 = Activity(coeff1)
	A2 = Activity(coeff2)
	M = Match(coeff1, coeff2)
	D = Decision(A1, A2, M)	
	return np.where(D == 0., np.maximum(coeff1, coeff2), D * coeff1 + (1 - D) * coeff2)
	
def Activity(coeff):
	"""
	Apply the Activity block of the MACD fusion strategy to the coefficient given in parameter
	
	coeff 	- coefficient
	
	
	Return Activity coefficient
	"""
	return np.absolute(coeff)

def Match(coeff1, coeff2):
	"""
	Apply the Match step of the MACD fusion strategy to the coefficients given in parameters
	
	coeff1 - coefficient of the RGB image
	coeff2 - coefficient of the IR image
	
	
	Return match coefficient
	"""
	mult = (coeff1 * coeff2) / (np.abs(coeff1)**2 + np.abs(coeff2)**2 + np.finfo(np.float32).eps)
	if mult.ndim == 3:
		kernel = np.ones((5, 5, 3))
	else:
		kernel = np.ones((5, 5))
	return ndimage.convolve(mult, kernel, mode='constant')

def Decision(coeff1, coeff2, m, fract = 0.5):
	"""
	Apply the Decision step of the MACD fusion strategy to the coefficients given in parameters
	
	coeff1 - activity array of the RGB image
	coeff2 - activity array of the IR image
	m 	   - match array 
	fract  - threshold between pure maximum and weighted max
	
	
	Return decision coefficient 
	"""
	mean = np.mean(m)
	delta = coeff1 + coeff2
	return np.where((delta == 0) | (m > fract * mean), 0.5, coeff1 / (delta + np.finfo(np.float32).eps))
	
def coeffsEntropy(coeff1, coeff2):
	"""
	Apply the entropy fusion strategy to the coefficients given in parameters
	
	coeff1 - coefficient of the RGB image
	coeff2 - coefficient of the IR image
	
	
	Return fused coefficient 
	"""
	entropy1 = shannon_entropy(coeff1 - coeff1.min())
	entropy2 = shannon_entropy(coeff2 - coeff2.min())
	delt = entropy1 + entropy2
	return (entropy1 * coeff1 + entropy2 * coeff2) / delt
	
def edgeDetection(coeff1, coeff2):
	"""
	Fuse two coefficients by first applying a Sobel filter, then taking a weighted average
	of both using their entropy
	
	coeff1 	- first coefficient
	coeff2 	- second coefficient
	
	
	Return fused coefficient
	"""
	edges_RGB = sobel_each(coeff1)
	edges_IR = sobel_each(coeff2)
	
	entropy_RGB = shannon_entropy(edges_RGB - edges_RGB.min())
	entropy_IR = shannon_entropy(edges_IR - edges_IR.min())
	
	entropy_sum = entropy_RGB + entropy_IR + np.finfo(np.float32).eps
	
	return (entropy_RGB * coeff1 + entropy_IR * coeff2) / entropy_sum
	
# def deviation(coeff1, coeff2, window_size = 8):
	# """
	# Fuse two coefficients by first dividing the coefficients, then using the 
	# standard deviation criterion
	
	# coeff1 	- first coefficient
	# coeff2 	- second coefficient
	# """
	# import time
	# time_start = time.time()
	# stdr = std_each(coeff1)
	# print (time.time() - time_start)
	# stdi = std_each(coeff2)
	# sum_std = stdr + stdi + np.finfo(np.float32).eps
	# return (stdr * coeff1 + stdi * coeff2) / sum_std
	
def deviation(coeff1, coeff2, window_size = 4):
	"""
	Fuse two coefficients by first dividing the coefficients, then using the 
	standard deviation criterion
	
	coeff1 	- first coefficient
	coeff2 	- second coefficient
	
	
	Return fused coefficient
	"""
	w, h = coeff1.shape[:2]
	result = np.zeros(coeff1.shape)
	
	for i in range(0, w, window_size):
		for j in range(0, h, window_size):
			RGB = coeff1[i:min(i + window_size, w), j:min(j + window_size, h)]
			IR = coeff2[i:min(i + window_size, w), j:min(j + window_size, h)]

			stdr = np.std(RGB, axis=(0, 1))
			stdi = np.std(IR, axis=(0, 1))
			result[i:min(i+window_size, w), j:min(j+window_size, h)] = (stdr * RGB + stdi * IR) / (stdr + stdi + np.finfo(np.float32).eps)
	
	return result
