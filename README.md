### Multi-spectral image and video dataset and its applications - Image Fusion
## Fares Ahmed (246443)

# Content

This zip contains the following files :
 * *main* is the main script of the image fusion process. When launched, the user is prompted to indicate the image's index to use as inputs, the strategy to apply, and if the fusion should apply on greyscale or rgb images.
 * *fuse* is the wavelet-based image fusion pipeline. It contains the functions used to decompose and recompose the inputs components.
 * *fusionStrategys* contains all the fusion strategys available for rgb images.
 * *fusionStrategysGray* contains all the fusion strategys available for greyscale images.
 * *GUI* contains the simple GUI that can be used in place of the main program. With the GUI, you can choose the images, the strategy to apply, the wavelet to use and if the fusion should apply on greyscale or rgb images.
 * *metrics* contains all the implemented metrics.
 * *ImageRegistration* contains the code used for the registration of the visual images.

# Fusion Strategys

The following fusion strategys are currently available :
 * Minimum - both greyscale and rgb images
 * Maximum - both greyscale and rgb images
 * Mean - both greyscale and rgb images
 * Entropy - both greyscale and rgb images
 * MACD - only greyscale images
 * Edge - only greyscale images
 * Deviation - only greyscale images

# Metrics

The following metrics are currently available :
 * Entropy
 * Structural Similarity Index
 * Image Quality Index
 * Spatial Frequency
 * ratio of Spatial Frequency Error
