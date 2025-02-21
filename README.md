# Basic Image Restoration and Compression Application
## Abstract
The image restoration and compression application implements and analyzes various image processing techniques, without using existing image processing libraries. The application provides an easy to use GUI built on PyQt5 to perform these operations on grayscale and colour images, and is completely implemented in Python. It was tested against a set of degraded images with varying levels of noise (AWGN) and known/unknown blur kernels.   

The application implements the following techniques:

### Image Restoration
* Inverse filtering
* Truncated inverse filtering
* Minimum mean square error (Weiner) filtering
* Constrained least squares filtering

### Image Compression
* DCT-based compression (similar to JPEG)
* Wavelet compression
* SVD-based compression
* Pre-compression filtering options:
  - Gaussian blur
  - Median filter
  - Bilateral filter

## Project Structure
- `main.py` - Main application entry point and GUI controller
- `gui.py` - GUI layout and widget definitions using PyQt5
- `imageRestorationFns.py` - Implementation of image restoration algorithms
- `imageCompressionFns.py` - Implementation of image compression algorithms
- `kernels/` - Directory containing blur kernel definitions
- `images/` - Sample input images for testing

## Dependencies
- Python v3
- PyQt5
- OpenCV (cv2) - For image I/O operations
- NumPy - For numerical computations
- SciPy - For DCT/IDCT operations
- PyWavelets - For wavelet compression
- Matplotlib - For visualization

## Instructions to Run
~~~~
python main.py
~~~~
Sample input images are available in the images folder.  

## Features
- Interactive GUI with real-time parameter adjustment
- Support for multiple image formats
- Quality/compression ratio control
- Pre-compression filtering options
- Compression statistics (ratio, file size)
- PSNR and SSIM quality metrics
- Before/after image comparison
- Save compressed/restored images

## Results
A screenshot of the application is given below.

![Basic Image Restoration Application Screenshot](https://github.com/Akshat190/Cv/blob/main/images/Screenshot%202025-02-21%20232904.png)

A demo video of the application is available [here](https://drive.google.com/open?id=1mvm7J7mfmm7ShP9_k_yBArl_OcwzjpvZ).  

## References
[1] Gonzalez, Rafael C., and Woods, Richard E. "Digital image processing. 3E" (2008).  
[2] http://webdav.is.mpg.de/pixel/benchmark4camerashake/  
[3] https://elementztechblog.wordpress.com/2015/04/14/getting-started-with-pycharm-and-qt-4-designer/  
[4] https://docs.scipy.org/doc/numpy/reference/  
[5] http://noise.imageonline.co/  
[6] Hore, Alain, and Djemel Ziou. "Image quality metrics: PSNR vs. SSIM." Pattern recognition (icpr), 2010 20th international conference on. IEEE, 2010.  
[7] Wallace, Gregory K. "The JPEG still picture compression standard." IEEE transactions on consumer electronics 38.1 (1992): xviii-xxxiv.
[8] Mallat, Stéphane G. "A theory for multiresolution signal decomposition: the wavelet representation." IEEE transactions on pattern analysis and machine intelligence 11.7 (1989): 674-693.
