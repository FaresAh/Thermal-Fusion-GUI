import numpy as np
import cv2
from skimage.measure.entropy import shannon_entropy
from skimage.measure import compare_ssim 
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import convolve
from scipy.ndimage.filters import correlate
from scipy.fftpack import fftshift

def IQI(X, Y):
	"""
	Calculate the Image Quality Index of an image compared to a reference image
	
	X - the image
	Y - the reference image
	
	
	Return the Image Quality Index of X - IQI(X) ∈ [-1, 1]
	"""
	def __conv(x):
		window = np.ones((BLOCK_SIZE, BLOCK_SIZE))
		if len(x.shape) < 3:
			return convolve(x, window)
		else:
			channels = x.shape[2]
			f = [convolve(x[:, :, c], window) for c in range(channels)]
			return np.array(f)

	def __get_filtered(im1, im2, BLOCK_SIZE):
		(im1im1, im2im2, im1im2) = (im1 * im1, im2 * im2, im1 * im2)
		(b1, b2, b3, b4, b5) = map(__conv, (im1, im2, im1im1, im2im2, im1im2))
		(b6, b7) = (b1 * b2, b1 * b1 + b2 * b2)
		return (b1, b2, b3, b4, b5, b6, b7)

	def __get_quality_map(b1, b2, b3, b4, b5, b6, b7, BLOCK_SIZE):
		N = BLOCK_SIZE * BLOCK_SIZE
		numerator = 4.0 * (N * b5 - b6) * b6
		denominator1 = N * (b3 + b4) - b7
		denominator = denominator1 * b7
		index = np.bitwise_and(denominator1 == 0, b7 != 0)
		quality_map = np.ones(denominator.shape)
		quality_map[index] = 2.0 * b6[index] / b7[index]
		index = (denominator != 0)
		quality_map[index] = numerator[index] / denominator[index]
		return quality_map[index]

	BLOCK_SIZE = 8
	(img1, img2) = (X.astype('double'), Y.astype('double'))
	(b1, b2, b3, b4, b5, b6, b7) = __get_filtered(img1, img2, BLOCK_SIZE)
	quality_map = __get_quality_map(b1, b2, b3, b4, b5, b6, b7, BLOCK_SIZE)
	value = quality_map.mean()
	return value
	
def SSIM(X, Y):
	"""
	Calculate the Structural Similarity index difference of two images
	
	X - the first image
	Y - the second image
	
	
	Return the Structural Similarity - SSIM(X, Y) ∈ [-1, 1]
	"""
	
	return compare_ssim(X, Y, multichannel=True)
	
def entr(coeffs):
	"""
	Calculate the Shannon Entropy of an image
	
	coeffs - the image
	
	
	Return entropy - H(coeffs) > 0
	"""
	return shannon_entropy(coeffs)
	

def spatial(I):
	"""
	Calculate the Spatial Frequency of an Image 
	
	I - the image
	
	Return the Spatial Frequency - SF(I) > 0
	"""
	
	Input = I.astype(int)
	
	row = (np.diff(Input, axis=1)**2).mean(axis=(0, 1))
	column = (np.diff(Input, axis=0)**2).mean(axis=(0, 1))
	
	# Main Diagonal frequency
	diagonal_m = (Input[1:, 1:] - Input[:-1, :-1])**2
	diagonal_m = diagonal_m.mean(axis=(0, 1)) / np.sqrt(2)
	
	# Secondary Diagonal frequency
	diagonal_s = (Input[1:, :-1] - Input[:-1, 1:]) ** 2
	diagonal_s = diagonal_s.mean(axis=(0, 1)) / np.sqrt(2)
	
	# SF = sqrt(RF^2 + CF^2 + MDF^2 + SDF^2)
	
	return np.sqrt(row * column * diagonal_m * diagonal_s)

def spatial_reference(X, Y):
	"""
	Calculate the Spatial Frequency of two reference images used for the fusion
	
	X - the first reference image
	Y - the second reference image
	
	
	Return the Spatial Frequency - SF(X, Y) > 0
	"""
	RefFirst = X.astype(int)
	RefSecond = Y.astype(int)
	
	first = np.diff(RefFirst, axis=1)**2
	second = np.diff(RefSecond, axis=1)**2
	row = np.maximum(first, second).mean(axis=(0, 1))
	
	first = np.diff(RefFirst, axis=0)**2
	second = np.diff(RefSecond, axis=0)**2
	column = np.maximum(first, second).mean(axis=(0, 1))
	
	first = (RefFirst[1:, 1:] - RefFirst[:-1, :-1])**2
	second = (RefSecond[1:, 1:] - RefSecond[:-1, :-1])**2
	diagonal_m = np.maximum(first, second).mean(axis=(0, 1)) / np.sqrt(2)

	first = (RefFirst[1:, :-1] - RefFirst[:-1, 1:]) ** 2
	second = (RefSecond[1:, :-1] - RefSecond[:-1, 1:]) ** 2
	diagonal_s = np.maximum(first, second).mean(axis=(0, 1)) / np.sqrt(2)
	
	return np.sqrt(row * column * diagonal_s * diagonal_m)
	

def rSFe(SF_input, SF_ref):
	"""
	Calculate the ratio of Spatial Frequency Error of the result of a fused image
	
	SF_input 	- the spatial frequency of the fused image
	SF_ref 		- the spatial frequency of the reference images
	
	
	Returns the ratio of Spatial Frequency Error ∈ [-1, 1]. A positive result means that an over-fused image, 
	with some distortion or noise introduced, has resulted. A negative result denotes that an under-fused image, 
	with some meaningful information lost, has been produced.
	"""
	return (SF_input - SF_ref) / SF_ref
