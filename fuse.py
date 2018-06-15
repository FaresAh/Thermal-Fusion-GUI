import numpy as np
from pywt import wavedec2, waverec2, wavelist
from fusionStrategies import MACD, edgeDetection, deviation, coeffsEntropy

def fusedImage(I1, I2, FUSION_METHOD, wavelet = 'db'):
	"""
	Fusion algorithm using wavelets
	
	I1				- the first image
	I2				- the second image
	FUSION_METOD	- the fusion strategy to apply to the coefficients
	wavelet 		- the wavelet to use
	
	Returns the fused image I
	"""
	wave = wavelist(wavelet)[0]
	
	# Apply wavelets on both image
	coeff1 = wavedec2(I1, wave, level=4, axes=(0, 1))
	coeff2 = wavedec2(I2, wave, level=4, axes=(0, 1))
	
	# For each level of decomposition, apply the fusion scheme wanted
	fusedCoeff = []
	for i in range(len(coeff1)):
		# coeffs = (cA, (cH_n, cV_n, cD_n), ...., ..(cH_1, cV_1, cD_1))
		if (i == 0):
			cA = fuseCoeff(coeff1[0], coeff2[0], FUSION_METHOD)
			fusedCoeff.append(cA)

		else:
			# For the rest of the levels we have tupels with 3 coefficents
			cH = fuseCoeff(coeff1[i][0], coeff2[i][0], FUSION_METHOD)
			cV = fuseCoeff(coeff1[i][1], coeff2[i][1], FUSION_METHOD)
			cD = fuseCoeff(coeff1[i][2], coeff2[i][2], FUSION_METHOD)
			fusedCoeff.append((cH, cV, cD))
			
	# Recompose the result image
	fusedImage = waverec2(fusedCoeff, wave, axes=(0, 1))
	# Normalize the values
	fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)
	fusedImage = fusedImage.astype(np.uint8)
	
	return fusedImage

def fuseCoeff(coeff1, coeff2, method):
	"""
	Apply the fusion strategy given in parameter to the coefficient from both images
	
	coeff1 - coefficient of the RGB image
	coeff2 - coefficient of the IR image
	method - the fusion strategy to apply
	
	
	Returns fused coefficient 
	"""
	if (method == 'Mean'):
		return (coeff1 + coeff2)/2
	elif (method == 'Min'):
		return np.minimum(coeff1, coeff2)
	elif (method == 'Max'):
		return np.maximum(coeff1, coeff2)
	elif (method == "Entropy"):
		return coeffsEntropy(coeff1, coeff2)
	elif method == 'MACD':
		return MACD(coeff1, coeff2)
	elif method == 'Edge':
		return edgeDetection(coeff1, coeff2)
	elif (method == "Deviation"):
		return deviation(coeff1, coeff2)
