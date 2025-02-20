import numpy as np
import cv2
from scipy.fftpack import dct, idct

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')    

def compress_image(image, compression_ratio):
    """
    Compress image using DCT and quantization
    """
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Work on 8x8 blocks
    height, width = image.shape[:2]
    block_size = 8
    
    # Quantization matrices
    Y_quantization = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ]) * compression_ratio

    CbCr_quantization = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ]) * compression_ratio

    compressed = np.zeros_like(ycrcb, dtype=float)

    # Process each channel
    for channel in range(3):
        quantization = Y_quantization if channel == 0 else CbCr_quantization
        
        for i in range(0, height-block_size+1, block_size):
            for j in range(0, width-block_size+1, block_size):
                block = ycrcb[i:i+block_size, j:j+block_size, channel].astype(float) - 128
                dct_block = dct2(block)
                quantized = np.round(dct_block / quantization)
                dequantized = quantized * quantization
                compressed[i:i+block_size, j:j+block_size, channel] = idct2(dequantized) + 128

    # Convert back to BGR
    result = cv2.cvtColor(compressed.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return result

def apply_filter(image, filter_type, params):
    """
    Apply different types of filters for compression
    """
    if filter_type == "gaussian":
        return cv2.GaussianBlur(image, (params['kernel_size'], params['kernel_size']), params['sigma'])
    elif filter_type == "median":
        return cv2.medianBlur(image, params['kernel_size'])
    elif filter_type == "bilateral":
        return cv2.bilateralFilter(image, params['d'], params['sigma_color'], params['sigma_space'])
    return image

def calculate_compression_ratio(original_image, compressed_image):
    """
    Calculate compression ratio and size reduction
    """
    original_size = original_image.nbytes
    compressed_size = compressed_image.nbytes
    ratio = original_size / compressed_size
    reduction = (original_size - compressed_size) / original_size * 100
    return ratio, reduction 