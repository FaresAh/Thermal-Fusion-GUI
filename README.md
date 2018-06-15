### Multi-spectral image and video dataset and its applications - Image Fusion
## Fares Ahmed (246443)

# Content

This zip contains the following files :
 * *main* is the main script of the image fusion process. By default, it applys all the fusion strategys available to the example images and display the metrics results in a command prompt.
 * *fuse* is the wavelet-based image fusion pipeline. It contains the functions used to decompose and recompose the inputs components.
 * *fusionStrategys* contains all the fusion strategys available.
 * *GUI* contains the simple GUI that can be used in place of the main program. With the GUI, you can choose the images, the strategy to apply and the wavelet to use.
 * *metrics* contains all the implemented metrics.
 * *ImageRegistration* contains the code used for the registration of the visual images.

# Fusion Strategys

The following fusion strategys are currently available :
 * Minimum
 * Maximum
 * Mean
 * Entropy
 * MACD
 * Edge
 * Deviation

# Metrics

The following metrics are currently available :
 * Entropy
 * Structural Similarity Index
 * Image Quality Index
 * Spatial Frequency
 * ratio of Spatial Frequency Error
 
# References

Here are some references for the implemented algorithms available. Note that it uses those resources mostly as inspiration, and should not be considered as complete implementation of those papers :
 * MACD : G. Piella, "A region-based multiresolution image fusion algorithm," 2002.
 * Deviation : S. Kanisetty and B. Hima, "Modified approach of multimodal medical image fusion using daubechies wavelet transform," International Journal of Advanced Research in Computer and Communication Engineering, vol. 2, 2013.
 * Image Quality Index : Z. Wang and A. Bovik, "A universal image quality index," IEEE Signal Processing Letters, vol. 9, 2002.
 * Spatial Frequency and ratio of Spatial Frequency Error : Y. Zheng, E. Essock, B. Hansen, and A. Haun, "A new metric based on extended spatial frequency and its application to dwt based fusion algorithm," Science Direct, 2004.
 * Structural Similarity Index :  Z. Wang, A. Bovik, H. Sheikh, and E. Simoncelli, "Image quality assessment: From error visibility to structural similarity," IEEE Transactions On Image Processing, vol. 13, 2004.
