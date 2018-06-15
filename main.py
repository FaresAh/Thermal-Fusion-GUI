import cv2
import numpy as np
from matplotlib import pyplot as plt
from fuse import fusedImage
from metrics import *
import time
import math

def show_images(images, lines = 1, titles = None, blocking = False):
	"""
	Displays a figure of images with titles
	
	images 	- the images to display
	lines 	- the number of lines the figure should have
	titles 	- the titles of each images
	"""
	
	n_images = len(images)
	
	fig = plt.figure()
	
	for n, (image, title) in enumerate(zip(images, titles)):
		a = fig.add_subplot(lines, np.ceil(n_images/float(lines)), n + 1)
		plt.imshow(image)
		plt.axis('off')
		a.set_title(title)
	
	fig.tight_layout()
	plt.show(block=blocking)

def main(rgb_path, ir_path, strategy = "All", gray=False, wavelet='db'):
	"""
	Main Fusion procedure, applies the fusion algorithm on the image
	
	index 	  - the index of the image
	strategy  - the fuison strategy to apply to the image
	gray 	  - indicate if the fusion process is to be applied on grayscale images
	wavelet   - the wavelet to use
	-----------
	
	Returns a tuple (array, Results, Titles). 
	
	array 	- The metrics results of the fused image(s) (array)
	Results - The fused image(s) (array)
	Titles 	- The name of the image(s) for the display (array)
	"""
	I1 = cv2.imread(rgb_path, 1)[:,:382]
	I2 = cv2.imread(ir_path, 1)
	
	if gray:
		I1 = cv2.cvtColor(I1, cv2.COLOR_RGB2GRAY)
		I2 = cv2.cvtColor(I2, cv2.COLOR_RGB2GRAY)
	
	if (strategy == "All"):
		R_min = fuseSelection(I1, I2, "Min", wavelet)
		R_max = fuseSelection(I1, I2, "Max", wavelet)
		R_mean = fuseSelection(I1, I2, "Mean", wavelet)
		R_entropy = fuseSelection(I1, I2, "Entropy", wavelet)
		R_MACD = fuseSelection(I1, I2, "MACD", wavelet)
		R_Edge = fuseSelection(I1, I2, "Edge", wavelet)
		R_Deviation = fuseSelection(I1, I2, "Deviation", wavelet)
		
		array = R_min[0] + ["------"] + R_max[0] + ["------"] + R_mean[0] + ["------"] + R_entropy[0] \
						 + ["------"] + R_MACD[0] + ["------"] + R_Edge[0] + ["------"] + R_Deviation[0]
						 
		Results = R_min[1] + R_max[1] + R_mean[1] + R_entropy[1] \
				  + R_MACD[1] + R_Edge[1] + R_Deviation[1]
		
		
		Titles = R_min[2] + R_max[2] + R_mean[2] + R_entropy[2] \
				 + R_MACD[2] + R_Edge[2] + R_Deviation[2]
		
		return (array, Results, Titles)
	else:
		return fuseSelection(I1, I2, I1_gray, I2_gray, strategy, wavelet)
	
def fuseSelection(I1, I2, strategy, wavelet):
	"""
	Fuse the images with the fusion strategy given as parameters 
	
	I1 			- the first image
	I2 			- the second image
	I1_gray 	- a grayscale version of the first image
	I2_gray 	- a grayscale version of the second image
	strategy	- the strategy to apply
	sp_input 	- the spatial frequency of the reference images
	is_gray 	- indicates if the fusion should be apply with grayscale images
	wavelet 	- the wavelet to use
	-------------
	
	Returns a tuple (array, Results, Titles)
	
	array 	- The metrics results of the fused image (array)
	Results - The fused image (array)
	Titles 	- The name of the image for the display (array)
	"""	
	array = []
	
	time_start = time.time()
	
	fusion_result = fusedImage(I1, I2, strategy, wavelet)
	if fusion_result.ndim == 3:
		result = cv2.cvtColor(fusion_result, cv2.COLOR_BGR2RGB)
		gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
	else:
		gray = fusion_result
		result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
		
	timing = (time.time() - time_start)
	
	# sp_input = spatial_reference(I1, I2)
	# sp_m = spatial(gray)

	# Entropy_m = "%.3f" % entr(gray)
	# IQI_m = "%.3f" % IQI(I1_gray, gray)
	# Spatial_m = "%.3f" % sp_m
	# rSFe_m = "%.3f" % ((sp_m - sp_input)/sp_input)
	# SSIM_m = "%.3f" % SSIM(I1_gray, gray)
		
	# array.append("Spatial Frequency of " + strategy + " : " + Spatial_m + ", rsFe : " + rSFe_m)
	# array.append("SSIM of " + strategy + " : " + SSIM_m)
	# array.append("Entropy of " + strategy + " : " + Entropy_m)
	# array.append("IQI of " + strategy + " : " + IQI_m)
	array.append("Time elapsed for " + strategy + " : " + str(timing)+ "s")

	Results = [result]
	Titles = [strategy]
	
	cv2.imwrite(strategy + ".png", result)
	
	return (array, Results, Titles)
	
	
if __name__ == '__main__':
	# rgb_path = str(input("Enter the rgb image path : "))
	# ir_path = str(input("Enter the thermal image path : "))
	
	# strategy = input("Strategy to use : (All - Min - Max - Mean - Entropy - MACD - Edge - Deviation) : ")
	
	# gray = int(input("Grayscale ? (0 for no) : "))
	
	rgb_path = 'examples/rgb.jpg'
	ir_path = 'examples/ir.png'
	strategy = 'All'
	gray = 1
	
	
	array, Results, Titles = main(rgb_path, ir_path, strategy, gray != 0)

	for s in array:
		print(s)
	
	show_images(Results, 3, Titles, True)
