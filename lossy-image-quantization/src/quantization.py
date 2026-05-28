import numpy as np

# ──────────────────────────────────────────
# 1. Quantification scalaire uniforme
# ──────────────────────────────────────────
def uniform_quantize(coeffs: np.ndarray, delta: float) -> np.ndarray:
    return np.round(coeffs / delta)


def uniform_dequantize(q: np.ndarray, delta: float) -> np.ndarray:
    return q * delta


# ──────────────────────────────────────────
# 2. Quantification à zone morte (Deadzone)
# ──────────────────────────────────────────
def deadzone_quantize(coeffs: np.ndarray, delta: float, alpha: float = 1.5) -> np.ndarray:
    """
    Quantification à zone morte symétrique.
    La zone morte autour de 0 est élargie à ±(alpha * delta / 2).
    alpha=1.5 est la valeur standard.
    """
    threshold = alpha * delta / 2.0
    q = np.zeros_like(coeffs)
    mask_pos = coeffs > threshold
    mask_neg = coeffs < -threshold
    q[mask_pos] = np.floor((coeffs[mask_pos] - threshold) / delta) + 1
    q[mask_neg] = np.ceil((coeffs[mask_neg] + threshold) / delta) - 1
    return q


def deadzone_dequantize(q: np.ndarray, delta: float, alpha: float = 1.5) -> np.ndarray:
    """
    Déquantification à zone morte : reconstruction au centre de chaque intervalle.
    """
    threshold = alpha * delta / 2.0
    recon = np.zeros_like(q)
    mask_pos = q > 0
    mask_neg = q < 0
    recon[mask_pos] = q[mask_pos] * delta + threshold
    recon[mask_neg] = q[mask_neg] * delta - threshold
    return recon


# ──────────────────────────────────────────
# 3. Quantification matricielle (JPEG)
# ──────────────────────────────────────────
def matrix_quantize(coeffs: np.ndarray, Q: np.ndarray) -> np.ndarray:
    H, W = coeffs.shape
    q = np.zeros_like(coeffs)
    for i in range(0, H, 8):
        for j in range(0, W, 8):
            q[i:i+8, j:j+8] = np.round(coeffs[i:i+8, j:j+8] / Q)
    return q


def matrix_dequantize(q: np.ndarray, Q: np.ndarray) -> np.ndarray:
    H, W = q.shape
    recon = np.zeros_like(q)
    for i in range(0, H, 8):
        for j in range(0, W, 8):
            recon[i:i+8, j:j+8] = q[i:i+8, j:j+8] * Q
    return recon
