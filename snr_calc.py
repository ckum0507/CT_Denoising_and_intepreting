import pywt
import numpy as np
import cv2

def estimate_snr_wavelet(image):
    if image is None:
        raise ValueError("Image not loaded. Check file path and format.")
    
    image = image.astype(np.float32)
    if image.max() > 1:
        image /= 255.0  # Normalize if in 8-bit range
    
    coeffs = pywt.wavedec2(image, 'haar', level=1)
    cA, (cH, cV, cD) = coeffs  # Approximation, Horizontal, Vertical, Diagonal details
    
    sigma_n = np.median(np.abs(np.concatenate([cH.ravel(), cV.ravel(), cD.ravel()]))) / 0.6745  # MAD estimator
    
    epsilon = 1e-10
    sigma_n = max(sigma_n, epsilon)
    
    mean_signal = np.abs(np.mean(cA))
    
    snr = mean_signal / sigma_n
    
    snr_db = 10 * np.log10(snr)
    
    return snr_db

def snr_loader(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load grayscale CT image
    snr_value = estimate_snr_wavelet(image)
    return f"{snr_value:.2f}"

def snr_printer(image):
    snr_value = estimate_snr_wavelet(image)
    return f"{snr_value:.2f}"