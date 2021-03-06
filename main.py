import cv2
import numpy as np
from matplotlib import pyplot as plt
from fuse import fusedImage
from metrics import *
import time
import math
from tkinter import filedialog
import tkinter as tk
import os

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

def main(rgb_path, ir_path, strategy = "All", wavelet='db'):
	"""
	Main Fusion procedure, applies the fusion algorithm on the image
	
	rgb_path  - the path to the RGB image
	ir_path	  - the path to the infrared image
	strategy  - the fuison strategy to apply to the image
	wavelet   - the wavelet to use
	-----------
	
	Returns a tuple (array, Results, Titles). 
	
	array 	- The metrics results of the fused image(s) (array)
	Results - The fused image(s) (array)
	Titles 	- The name of the image(s) for the display (array)
	"""
	I1 = cv2.imread(rgb_path, 1)
	I2 = cv2.imread(ir_path, 1)
	
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
		return fuseSelection(I1, I2, strategy, wavelet)
	
def fuseSelection(I1, I2, strategy, wavelet):
	"""
	Fuse the images with the fusion strategy given as parameters 
	
	I1 			- the first image
	I2 			- the second image
	strategy	- the strategy to apply
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
		
	timing = "%.2f" % (time.time() - time_start)
	
	I1_gray = cv2.cvtColor(I1, cv2.COLOR_RGB2GRAY)
	I2_gray = cv2.cvtColor(I2, cv2.COLOR_RGB2GRAY)
	
	sp_input = spatial_reference(I1_gray, I2_gray)
	sp_m = spatial(gray)
	

	Entropy_m = "%.3f" % shannon_entropy(result)
	IQI_m = "%.3f" % IQI(I1, result)
	Spatial_m = "%.3f" % sp_m.sum()
	rSFe_m = "%.3f" % rSFe(sp_m, sp_input).sum()
	SSIM_m = "%.3f" % SSIM(I1, result)
		
	array.append("Spatial Frequency of " + strategy + " : " + Spatial_m + ", rsFe : " + rSFe_m)
	array.append("SSIM of " + strategy + " : " + SSIM_m)
	array.append("Entropy of " + strategy + " : " + Entropy_m)
	array.append("IQI of " + strategy + " : " + IQI_m)
	array.append("Time elapsed for " + strategy + " : " + timing+ "s")

	Results = [result]
	Titles = [strategy]
	
	return (array, Results, Titles)
	
	
if __name__ == '__main__':
	root = tk.Tk()
	root.withdraw()
	
	rgb_path = filedialog.askopenfilename(initialdir=os.getcwd(),title='Choose the RGB Image', filetypes = [("Image File (.png, .jpg)", "*.jpg *.png")])
	
	if not rgb_path:
		print("No RGB Image selected, ending program...")
		exit()
	
	
	ir_path = filedialog.askopenfilename(initialdir=os.getcwd(),title='Choose the Thermal Image', filetypes = [("Image File (.png, .jpg)", "*.jpg *.png")])
	
	if not ir_path:
		print("No Thermal Image selected, ending program..")
		exit()
		
	strategy = 'All'
	
	# strategy = input("Strategy to use : (All - Min - Max - Mean - Entropy - MACD - Edge - Deviation) : ")
	
	array, Results, Titles = main(rgb_path, ir_path, strategy)

	for s in array:
		print(s)
	
	show_images(Results, math.ceil(len(Results) / 3.0), Titles, True)
