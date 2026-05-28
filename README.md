# Multimedia Engineering Toolkit

This repository is a consolidated suite of Python tools demonstrating core principles of multimedia processing, data compression, and temporal video analysis. It includes custom implementations of lossless codecs, lossy block-coding algorithms, and computer vision shot boundary detection.

---

## Suite Components

The toolkit is organized into three independent sub-projects:

### 1. [Lossless Image Compression Suite](./lossless-image-compression)
A benchmarking workspace for lossless data encoding.
* **Algorithms**: Custom implementations of **LZ77**, **LZW (12-bit)**, and **RLE (16-bit)**.
* **Analysis**: Tool to calculate global and local sliding-window **Shannon Entropy** to map spatial complexity and determine the theoretical lower bounds of compression.
* **Results**: Benchmarked on natural (RGB), synthetic (Palette), and binary (16-bit) images to demonstrate algorithm strengths.

### 2. [Lossy DCT-Based Quantization](./lossy-image-quantization)
An analysis tool for lossy frequency-domain image compression.
* **Transform**: $8 \times 8$ block level-shifting and 2D Discrete Cosine Transform (DCT).
* **Quantization**: Compares **Uniform Scalar**, **Deadzone Scalar** (with parameter $\alpha$), and **JPEG Perceptual Matrix** quantization.
* **Evaluation**: Rates reconstruction fidelity using PSNR, SSIM (Human Visual System proxy), and Sparsity curves.

### 3. [Video Shot Boundary Detection & Summarization](./video-shot-detection)
A computer vision pipeline for temporal video indexing.
* **Detection**: Segments videos by computing inter-frame **HSV block-histogram Chi-Square distances** combined with an **adaptive rolling threshold** to detect hard camera cuts.
* **Summarization**: Clusters frames within each shot using **KMeans clustering** on 3D HSV descriptors to select representative keyframes and generate visual mosaic grids.
* **Evaluation**: Benchmarked against manual ground truths using Precision, Recall, and F-Score metrics.

---

## Directory Navigation

```
portfolio/
├── lossless-image-compression/   # Lossless LZW, LZ77, RLE, and Shannon Entropy
├── lossy-image-quantization/     # Block 2D DCT, Deadzone, and JPEG Matrix quantization
└── video-shot-detection/         # Video cut detection and KMeans keyframes
```
*Each folder contains its own self-contained source code, local dataset, result files, and detailed technical documentation.*

---

## Academic Context
These projects were originally written as part of the **INF8770 (Technologies Multimédias)** course at **Polytechnique Montréal**. The respective student reports and mathematical proofs can be found inside the `docs/` folders of the sub-projects.
