# Lossless Image Compression Benchmarks

This repository contains custom Python implementations of three classic lossless compression algorithms (**LZ77**, **LZW (12-bit)**, and **RLE (16-bit)**) and tools to analyze the spatial complexity of images using **Shannon Entropy** (both global entropy and local sliding-window maps).

We wrote this project to benchmark how different compression algorithms perform on different image characteristics (natural noise vs. synthetic patterns vs. binary flat zones) in relation to their theoretical entropy limits.

---

## Code Example

You can use the compression engines directly in Python:

```python
from src.compressor import LZW12, LZ77, RLE16

# LZW (12-bit)
lzw = LZW12()
compressed = lzw.compress(b"banana banana")
original = lzw.decompress(compressed)

# LZ77 (sliding window)
lz77 = LZ77(window_size=4096, max_match=32)
compressed = lz77.compress(b"banana banana")
original = lz77.decompress(compressed)
```

---

## Setup & Running

Install dependencies:
```bash
pip install numpy pandas pillow matplotlib tabulate
```

Run the benchmark suite:
```bash
python main.py
```

This script will:
1. Compute the global Shannon entropy and generate local sliding-window entropy heatmaps in the `results/` folder.
2. Run LZW on the natural image, LZ77 on the synthetic image, and RLE on the binary image.
3. Verify that the compression is mathematically lossless (checking original vs. decompressed bytes).
4. Save the performance metrics to `results/compression_results.csv`.

---

## Experimental Results

The suite tests each algorithm against a specific image type representing different spatial characteristics:
- **Natural Image (`image1_natural.png`)**: High-entropy, noisy RGB image ($24\text{ bpp}$).
- **Synthetic Image (`image2_synthetic.png`)**: Palette-based image ($8\text{ bpp}$) with clean geometric paths.
- **Binary/Line-Art Image (`image3_binary.png`)**: High-resolution 16-bit grayscale image ($16\text{ bpp}$) with large solid areas.

### Image Complexity Metrics
| Image | Mode | Dimensions | Unique Values | Global Entropy (bits) | Mean Local Entropy (bits) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `image1_natural.png` | RGB | $512 \times 768$ | 255 | 7.48 | 3.37 |
| `image2_synthetic.png` | P | $477 \times 599$ | 162 | 4.00 | 2.92 |
| `image3_binary.png` | I;16 | $1024 \times 1024$ | 254 | 4.94 | 0.61 |

### Compression Performance
| Image | Method | Dimensions | Original Size | Compressed Size | Reduction (%) | Ratio | Original BPP | Compressed BPP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `image1_natural.png` | **LZW** | $512 \times 768$ | $1,179,648\text{ B}$ | $1,507,380\text{ B}$ | -27.78% | 0.78:1 | 24.00 | 30.67 |
| `image2_synthetic.png` | **LZ77** | $477 \times 599$ | $285,723\text{ B}$ | $254,348\text{ B}$ | 10.98% | 1.12:1 | 8.00 | 7.12 |
| `image3_binary.png` | **RLE** | $1024 \times 1024$ | $2,097,152\text{ B}$ | $867,948\text{ B}$ | 58.61% | 2.42:1 | 16.00 | 6.62 |

### Key Observations
- **LZW on Natural Images**: Natural images contain noise and high color variation (global entropy $\approx 7.48$ bits/pixel). Attempting LZW on the raw RGB byte stream causes dictionary overflow and code expansion (negative compression), showing why predictive (PNG) or lossy (JPEG) pipelines are required for natural photos.
- **LZ77 on Synthetic Images**: Synthetic drawings have repetitive patterns. The sliding window (4096 bytes) matches duplicate structures, yielding a positive compression ratio.
- **RLE on Binary Images**: RLE is highly effective here because the image has a very low mean local entropy ($0.61\text{ bits}$), meaning it has long consecutive runs of identical pixels. The size was reduced by **58.6%** without any quality loss.

---

## Academic Context
This project was originally written as part of the **INF8770 (Technologies Multimédias)** course at **Polytechnique Montréal**. The complete academic report detailing our experimental methodology and mathematical proofs is included in [docs/report.pdf](docs/report.pdf).
