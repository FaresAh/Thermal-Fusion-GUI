from skimage.measure import compare_ssim 
import numpy as np
import cv2
from math import sqrt

def IQI(X, Y):
	"""
	Calculate the Image Quality Index of an image compared to a reference image
	More information about this metric can be found here :
	
	X - the image
	Y - the reference image
	
	
	Returns the Image Quality Index of X - IQI(X) [-1, 1]
	"""
	
	x_ = np.mean(X)
	y_ = np.mean(Y)
	
	
	w, h = X.shape 
	
	x_std = 0
	y_std = 0
	xy_std = 0
	for i in range(w):
		for j in range(h):
			x_std = x_std + ((X[i, j] - x_) **2)
			y_std = y_std + ((Y[i, j] - y_) **2)
			xy_std = xy_std + ((X[i, j] - x_) * (Y[i, j] - y_))

	x_std = x_std / (w + h - 1)
	y_std = y_std / (w + h - 1)
	xy_std = xy_std / (w + h - 1)
	
	x_std_sqrt = x_std ** (1/2)
	y_std_sqrt = y_std ** (1/2)
	
	# Correlation coefficient
	Q = (0.0 if ((x_std_sqrt * y_std_sqrt) == 0.) else xy_std/(x_std_sqrt * y_std_sqrt))
	
	# Mean luminance
	Q = Q * (2 * x_ * y_)/(x_ * x_ + y_ * y_)
	
	# Contrasts
	Q = Q * (2 * x_std_sqrt * y_std_sqrt) / (x_std + y_std)
	
	return Q
	
def SSIM(X, Y):
	"""
	Calculate the Structural Similarity index difference of two images
	
	X - the first image
	Y - the second image
	
	
	Returns the Structural Similarity - SSIM(X, Y) [-1, 1]
	"""
	
	return compare_ssim(X, Y)
	
def entr(coeffs):	
	"""
	Calculate the Shannon Entropy of an image
	
	coeffs - the image
	
	
	Returns entropy - H(coeffs)
	"""
	res = coeffs[coeffs > 0]
	res = res/res.sum()
	res = (res * np.log2(res))
	
	return -res.sum()
	

def spatial(I):
	"""
	Calculate the Spatial Frequency of an Image 
	More information about this metric can be found in the following paper :
	
	I - the image
	
	
	Returns the Spatial Frequency - SF(I) > 0
	"""
	w, h = I.shape[0], I.shape[1]
	
	# Row frequency
	row = 0
	for i in range(w):
		for j in range(1, h):
			row = row + ((int(I[i, j]) - int(I[i, j-1])) ** 2)
	row = row / (w*h)
		
	# Column frequency
	column = 0
	for j in range(h):
		for i in range(1, w):
			column = column + ((int(I[i, j]) - int(I[i - 1, j])) ** 2)
	column = column / (w*h)
	
	# Main Diagonal frequency
	diagonal_m = 0
	for i in range(1, w):
		for j in range(1, h):
			diagonal_m = diagonal_m + ((int(I[i, j]) - int(I[i - 1, j - 1])) ** 2)
	diagonal_m = diagonal_m / (w*h)
	diagonal_m = diagonal_m / sqrt(2)
	
	# Secondary Diagonal frequency
	diagonal_s = 0
	for j in range(h - 1):
		for i in range(1, w):
			diagonal_s = diagonal_s + ((int(I[i, j]) - int(I[i - 1, j + 1])) ** 2)
	diagonal_s = diagonal_s / (w*h)
	diagonal_s = diagonal_s / sqrt(2)
	
	# SF = sqrt(RF^2 + CF^2 + MDF^2 + SDF^2)
	
	return sqrt(row * column * diagonal_m * diagonal_s)

def spatial_reference(X, Y):
	"""
	Calculate the Spatial Frequency of two reference images used for the fusion
	
	X - the first reference image
	Y - the second reference image
	
	
	Returns the Spatial Frequency - SF(X, Y) > 0
	"""
	
	w, h = X.shape
	
	row = 0
	first, second = 0, 0
	for i in range(w):
		for j in range(1, h):
			first = (int(X[i, j]) - int(X[i, j-1])) ** 2
			second = (int(Y[i, j]) - int(Y[i, j-1])) ** 2
			row = row + max(first, second)
	row = row / (w*h)
			
	column = 0
	for j in range(h):
		for i in range(1, w):
			first = (int(X[i, j]) - int(X[i - 1, j])) ** 2
			second = (int(Y[i, j]) - int(Y[i - 1, j])) ** 2
			column = column + max(first ,second)
			
	column = column / (w*h)

	diagonal_m = 0
	for i in range(1, w):
		for j in range(1, h):
			first = (int(X[i, j]) - int(X[i - 1, j - 1])) ** 2
			second = (int(Y[i, j]) - int(Y[i - 1, j - 1])) ** 2
			diagonal_m = diagonal_m + max(first, second)
	diagonal_m = diagonal_m / (w*h)
	diagonal_m = diagonal_m / sqrt(2)
	
	diagonal_s = 0
	for j in range(h - 1):
		for i in range(1, w):
			first = (int(X[i, j]) - int(X[i - 1, j + 1])) ** 2
			second = (int(Y[i, j]) - int(Y[i - 1, j + 1])) ** 2
			diagonal_s = diagonal_s + max(first, second)
	diagonal_s = diagonal_s / (w*h)
	diagonal_s = diagonal_s / sqrt(2)

	
	return sqrt(row * column * diagonal_s * diagonal_m)
	

def rSFe(SF_input, SF_ref):
	"""
	Calculate the ratio of Spatial Frequency Error of the result of a fused image
	
	SF_input 	- the spatial frequency of the fused image
	SF_ref 		- the spatial frequency of the reference images
	
	
	Returns the ratio of Spatial Frequency Error. A positive result means that an over-fused image, 
	with some distortion or noise introduced, has resulted. A negative result denotes that an under-fused image, 
	with some meaningful information lost, has been produced.
	"""
	return (SF_input - SF_ref)/SF_ref
