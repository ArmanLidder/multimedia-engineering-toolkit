# DCT Image Quantization Comparison

This repository contains Python code to implement and compare different image quantization strategies on 2D Discrete Cosine Transform (DCT) blocks. 

Specifically, it implements:
1. **Uniform Scalar Quantization**: Constant step size $\Delta$ across all frequency components.
2. **Deadzone Scalar Quantization**: Broadens the reconstruction interval around zero by a factor of $\alpha=1.5$ to eliminate low-amplitude high-frequency noise.
3. **JPEG Perceptual Matrix Quantization**: Quantizes frequency components using the standard JPEG luminance matrix, scaled by $1\times$, $2\times$, or $4\times$.

We use this suite to evaluate the rate-distortion performance of these methods using Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Sparsity (% of zero coefficients).

---

## Code Example

To run the block-DCT transforms and quantization steps directly in Python:

```python
import numpy as np
from src.dct import apply_dct_blocks, reconstruct_from_coeffs
from src.quantization import uniform_quantize, uniform_dequantize

# Load your image as a numpy array `img_np`
# 1. Apply DCT on 8x8 blocks
coeffs = apply_dct_blocks(img_np)

# 2. Quantize (e.g., Uniform with delta=16)
q = uniform_quantize(coeffs, delta=16)

# 3. Dequantize and Reconstruct
coeffs_rec = uniform_dequantize(q, delta=16)
recon = reconstruct_from_coeffs(coeffs_rec)
```

---

## Setup & Running

Install dependencies:
```bash
pip install numpy pandas pillow scipy scikit-image matplotlib tabulate
```

Run the comparison benchmark pipeline:
```bash
python main.py
```

This script runs all three quantization methods (for various steps and scales) on the reference image `data/image.png`, prints a comparison table, and generates diagnostic plots in the `results/` folder:
- `reconstructions_comparison.png`: Rendered outputs of all runs side-by-side.
- `coefficient_distributions.png`: Log-scale histograms showing the coefficient zero-mass.
- `sparsity_vs_quality.png`: Rate-distortion plots mapping SSIM vs. Sparsity.
- `residuals_comparison.png`: Color-mapped spatial residuals showing where errors occurred.

---

## Experimental Results

Benchmarks run on the reference grayscale image ($512 \times 512$):

| Quantizer | Parameter | PSNR (dB) | SSIM | Sparsity (% zeros) |
| :--- | :---: | :---: | :---: | :---: |
| **Uniform** | $\Delta=8$ | 41.46 | 0.984 | 44.3% |
| **Uniform** | $\Delta=16$ | 36.03 | 0.952 | 62.5% |
| **Uniform** | $\Delta=32$ | 31.06 | 0.878 | 80.7% |
| **Deadzone** | $\Delta=8$ | 36.39 | 0.962 | 54.9% |
| **Deadzone** | $\Delta=16$ | 31.52 | 0.905 | 73.6% |
| **Deadzone** | $\Delta=32$ | 27.69 | 0.802 | 88.5% |
| **JPEG** | standard ($1\times$) | 30.21 | 0.879 | 80.6% |
| **JPEG** | scale $2\times$ | 28.28 | 0.817 | 88.1% |
| **JPEG** | scale $4\times$ | 26.58 | 0.733 | 93.3% |

### Key Observations
* **PSNR vs. SSIM (Perceptual Trade-off)**: At a similar sparsity level of $\approx 80.6\%$, **Uniform $\Delta=32$** has a slightly higher PSNR than **JPEG Standard** ($31.06\text{ dB}$ vs. $30.21\text{ dB}$). However, their SSIM values are identical ($0.878$ vs. $0.879$). Uniform quantization introduces blocky artifacts in flat areas because it quantizes important low frequencies heavily. JPEG preserves low frequencies and concentrates distortion in high frequencies where human vision is less sensitive.
* **Deadzone Sparsity**: Deadzone scalar quantization generates high sparsity very quickly ($88.5\%$ at $\Delta=32$), but causes a large drop in PSNR due to aggressive zeroing of mid-range transition details.

---

## Academic Context
This project was originally written as part of the **INF8770 (Technologies Multimédias)** course at **Polytechnique Montréal**. The complete academic report detailing the experimental methodology and detailed analysis is available in [docs/report.pdf](docs/report.pdf).
