import numpy as np
from pywt import wavedec2, waverec2, wavelist
from fusionStrategys import minim, maxim, coeffsEntropy
from fusionStrategysGray import minimGray, maximGray, coeffsEntropyGray, MACD, edgeDetection, deviation

def fusedImage(I1, I2, FUSION_METHOD, wavelet = 'db', gray = False):
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
	coeff1 = wavedec2(I1, wave, level= 1)
	coeff2 = wavedec2(I2, wave, level= 1)

	# For each level of decomposition, apply the fusion scheme wanted
	fusedCoeff = []
	for i in range(len(coeff1)):
		# coeffs = (cA, (cH_n, cV_n, cD_n), ...., ..(cH_1, cV_1, cD_1))
		
		if (i == 0):
			cA = fuseCoeff(coeff1[0], coeff2[0], FUSION_METHOD, gray)
			
			fusedCoeff.append(cA)

		else:
			# For the rest of the levels we have tupels with 3 coefficents

			cH = fuseCoeff(coeff1[i][0], coeff2[i][0], FUSION_METHOD, gray)
			cV = fuseCoeff(coeff1[i][1], coeff2[i][1], FUSION_METHOD, gray)
			cD = fuseCoeff(coeff1[i][2], coeff2[i][2], FUSION_METHOD, gray)
			
			fusedCoeff.append((cH, cV, cD))
			
			

	# Recompose the result image
	fusedImage = waverec2(fusedCoeff, wave)

	# Normalize the values
	fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)
	fusedImage = fusedImage.astype(np.uint8)
	
	return fusedImage

def fusedEdge(I1, I2, wavelet = 'db'):
	"""
	Fusion algorithm using convolutions and intensity
	
	I1				- the first image
	I2				- the second image
	wavelet 		- the wavelet to use
	
	
	Returns the fused image I
	"""
	wave = wavelist(wavelet)[0]
	
	coeff1 = wavedec2(I1, wave, level=4)
	coeff2 = wavedec2(I2, wave, level=4)
	
	# coeffs = (cA, (cH_4, cV_4, cD_4), (cH_3, cV_3, cD_3), (cH_2, cV_2, cD_2), (cH_1, cV_1, cD_1))

	fusedCoeff = []
	
	fusedCoeff.append(edgeDetection(coeff1[0], coeff2[0]))
	
	cH_4 = edgeDetection(coeff1[1][0], coeff2[1][0])
	cV_4 = edgeDetection(coeff1[1][1], coeff2[1][1])
	cD_4 = edgeDetection(coeff1[1][2], coeff2[1][2])
	
	fusedCoeff.append((cH_4, cV_4, cD_4))
	
	cH_3 = edgeDetection(coeff1[2][0], coeff2[2][0])
	cV_3 = edgeDetection(coeff1[2][1], coeff2[2][1])
	cD_3 = edgeDetection(coeff1[2][2], coeff2[2][2])
	
	fusedCoeff.append((cH_3, cV_3, cD_3))
	
	cH_2 = edgeDetection(coeff1[3][0], coeff2[3][0])
	cV_2 = edgeDetection(coeff1[3][1], coeff2[3][1])
	cD_2 = edgeDetection(coeff1[3][2], coeff2[3][2])
	
	fusedCoeff.append((cH_2, cV_2, cD_2))
	
	cH_1 = minimGray(coeff1[4][0], coeff2[4][0])
	cV_1 = minimGray(coeff1[4][1], coeff2[4][1])
	cD_1 = minimGray(coeff1[4][2], coeff2[4][2])
	
	fusedCoeff.append((cH_1, cV_1, cD_1))
			
	#problem of size here, forgot one tuple of coeffs
	
	# Recompose the result image
	fusedImage = waverec2(fusedCoeff, wave)
	
	# Normalize the values
	fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)
	fusedImage = fusedImage.astype(np.uint8)

	return fusedImage

def fuseCoeff(coeff1, coeff2, method, gray = False):
	"""
	Apply the fusion strategy given in parameter to the coefficient from both images
	
	coeff1 - coefficient of the RGB image
	coeff2 - coefficient of the IR image
	method - the fusion strategy to apply
	
	
	Returns fused coefficient 
	"""
	
	if (gray):
		return fuseCoeffGray(coeff1, coeff2, method)
		
	coeff = []
	
	if (method == 'Mean'):
		coeff = (coeff1 + coeff2)/2
	elif (method == 'Min'):
		coeff = minim(coeff1, coeff2)
	elif (method == 'Max'):
		coeff = maxim(coeff1, coeff2)
	elif (method == "Entropy"):
		coeff = coeffsEntropy(coeff1, coeff2)

	return coeff

def fuseCoeffGray(coeff1, coeff2, method):
	"""
	Apply the fusion strategy given in parameter to the coefficient from both grayscale images
	
	coeff1 - coefficient of the RGB image
	coeff2 - coefficient of the IR image
	method - the fusion strategy to apply
	
	
	Returns fused coefficient 
	"""
	coeff = []
	
	if (method == 'Mean'):
		coeff = (coeff1 + coeff2)/2
	elif (method == 'Min'):
		coeff = minimGray(coeff1, coeff2)
	elif (method == 'Max'):
		coeff = maximGray(coeff1, coeff2)
	elif (method == "Entropy"):
		coeff = coeffsEntropyGray(coeff1, coeff2)
	elif (method == "MACD"):
		coeff = MACD(coeff1, coeff2)
	elif (method == "Deviation"):
		coeff = deviation(coeff1, coeff2)

	return coeff
