import numpy as np
from scipy.fftpack import dct, idct

def dct2(block: np.ndarray) -> np.ndarray:
    """Computes the 2D Discrete Cosine Transform of an 8x8 block."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block: np.ndarray) -> np.ndarray:
    """Computes the 2D Inverse Discrete Cosine Transform of an 8x8 block."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def apply_dct_blocks(img: np.ndarray) -> np.ndarray:
    """
    Splits the image into 8x8 blocks, subtracts 128 (level shifting),
    and applies 2D DCT on each block.
    """
    H, W = img.shape
    coeffs = np.zeros_like(img)
    for i in range(0, H, 8):
        for j in range(0, W, 8):
            block = img[i:i+8, j:j+8] - 128.0
            coeffs[i:i+8, j:j+8] = dct2(block)
    return coeffs


def reconstruct_from_coeffs(coeffs: np.ndarray) -> np.ndarray:
    """
    Reconstructs the image from 8x8 DCT coefficients by applying 2D IDCT,
    adding 128 (level shifting back), rounding, and clamping values to [0, 255].
    """
    H, W = coeffs.shape
    recon = np.zeros_like(coeffs)
    for i in range(0, H, 8):
        for j in range(0, W, 8):
            block = idct2(coeffs[i:i+8, j:j+8])
            recon[i:i+8, j:j+8] = block + 128.0
    return np.clip(np.round(recon), 0, 255).astype(np.uint8)
