# Video Shot Boundary Detection & Keyframe Extraction

This repository contains Python code for automated video segmentation (cut detection) and visual summarization. 

Specifically, it implements:
1. **Shot Boundary Detection**: Converts video frames to HSV, divides them into a $4 \times 4$ spatial grid, computes normalized Hue-Saturation block-histograms, and calculates Chi-Square inter-frame differences. Shot boundaries are triggered using an adaptive sliding-window threshold.
2. **Keyframe Extraction**: Extracts 3D color descriptors for all frames within each detected shot, applies KMeans clustering, and selects the frames closest to the cluster centers to assemble a visual mosaic.

---

## Code Example

To run the shot detection steps directly in your own Python script:

```python
import cv2
from src.detector import detect_shots
from src.summarizer import select_keyframes, save_mosaic

# 1. Run cut detection (using adaptive threshold multiplier k=4.0)
cuts_sec, cuts_frames, frames, diffs, fps = detect_shots(
    "path/to/video.mp4",
    video_name="GoT",
    k_adaptive=4.0,
    min_gap_sec=0.8
)

# 2. Extract keyframes from detected shot limits
keyframes = select_keyframes(frames, cuts_frames, fps)

# 3. Save as a visual grid
save_mosaic(keyframes, "GoT Summary", "results/got_summary.png")
```

---

## Setup & Running

Install dependencies:
```bash
pip install numpy pandas opencv-python scikit-learn matplotlib tabulate
```

Run the pipeline:
```bash
python main.py --video_dir path/to/mp4/videos --out_dir results
```

This script will run detection on F1, GoT, and StarWars clips, print the evaluation metrics against manual ground truths, and save difference plots and mosaic grids inside the `results/` folder.

---

## Evaluation Results

The pipeline evaluates performance against ground truth frame indices using a $\pm 2.0$-second tolerance window:

| Video | Detected Cuts | Ground Truth Cuts | TP | FP | FN | Precision | Recall | F-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Formula 1 Highlights** | 25 | 16 | 15 | 10 | 1 | 0.60 | 0.94 | 0.73 |
| **Game of Thrones (Battle)** | 19 | 18 | 16 | 3 | 2 | 0.84 | 0.89 | 0.86 |
| **Star Wars Opening Crawl** | 2 | 1 | 1 | 1 | 0 | 0.50 | 1.00 | 0.67 |

### Key Observations
* **F1 (High Motion)**: The video features extremely high action and fast camera panning. While the system gets almost all cuts (94% recall), high camera motion triggers several false positives (10 FPs) in the Hue-Saturation histograms.
* **GoT (Low Contrast)**: The system handles dark scenes and lightning flashes very well, achieving an 86% F-Score.
* **Star Wars (Fades)**: The system correctly finds the hard cut at the start of the crawl (9s), but also catches a false positive when the title fades out, illustrating the limitation of basic histogram difference thresholds when handling soft transitions.

---

## Academic Context
This project was originally written as part of the **INF8770 (Technologies Multimédias)** course at **Polytechnique Montréal**.
