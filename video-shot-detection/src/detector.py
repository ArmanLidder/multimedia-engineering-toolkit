import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def compute_frame_diff(frame1: np.ndarray, frame2: np.ndarray, grid: tuple[int, int] = (4, 4)) -> float:
    """
    Computes the inter-frame distance by splitting frames into a grid, 
    calculating normalized 2D HSV histograms for each block, 
    and comparing them using the Chi-Square distance.
    """
    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    h, w = hsv1.shape[:2]
    bh = h // grid[0]
    bw = w // grid[1]
    dists = []
    
    for r in range(grid[0]):
        for c in range(grid[1]):
            b1 = hsv1[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
            b2 = hsv2[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
            
            # Compute 2D Hue-Saturation Histogram
            hist1 = cv2.calcHist([b1], [0, 1], None, [16, 16], [0, 180, 0, 256])
            hist2 = cv2.calcHist([b2], [0, 1], None, [16, 16], [0, 180, 0, 256])
            
            cv2.normalize(hist1, hist1)
            cv2.normalize(hist2, hist2)
            
            dists.append(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR))
            
    return float(np.mean(dists))


def merge_close_cuts(cuts: list[int], min_gap_frames: int) -> list[int]:
    """Merges detected boundaries that are closer than a minimum frame gap."""
    if not cuts:
        return cuts
    merged = [cuts[0]]
    for c in cuts[1:]:
        if c - merged[-1] >= min_gap_frames:
            merged.append(c)
    return merged


def detect_shots(video_path: str, video_name: str, k_adaptive: float = None, 
                 min_gap_sec: float = 1.0, window_size: int = 50, gt_end_sec: float = None):
    """
    Reads the video, computes inter-frame differences, and applies adaptive or global thresholding 
    to detect shot boundaries.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = int((gt_end_sec + 5) * fps) if gt_end_sec is not None else total_frames

    diffs = []
    frames = []
    prev = None
    idx = 0

    while idx < max_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame_small = cv2.resize(frame, (320, 180))
        if prev is not None:
            diffs.append(compute_frame_diff(prev, frame_small))
        frames.append(frame)
        prev = frame_small
        idx += 1

    cap.release()
    diffs = np.array(diffs)
    cuts_frames = []

    if k_adaptive is None:
        # Global thresholding (e.g. Star Wars crawl)
        threshold = np.percentile(diffs, 99.5)
        for i in range(len(diffs)):
            if diffs[i] > threshold:
                cuts_frames.append(i + 1)
    else:
        # Adaptive sliding-window thresholding
        for i in range(len(diffs)):
            start = max(0, i - window_size)
            end = i
            if end - start < 5:
                continue
            local_mean = np.mean(diffs[start:end])
            local_std = np.std(diffs[start:end])
            if diffs[i] > local_mean + k_adaptive * local_std:
                cuts_frames.append(i + 1)

    min_gap_frames = int(fps * min_gap_sec)
    cuts_frames = merge_close_cuts(cuts_frames, min_gap_frames)
    cuts_sec = [f / fps for f in cuts_frames]

    return cuts_sec, cuts_frames, frames, diffs, fps


def evaluate(detected_sec: list[float], ground_truth_sec: list[float], tolerance_sec: float = 2.0):
    """Computes TP, FP, FN, Precision, Recall, and F1-Score against ground truths."""
    matched_gt = set()
    matched_det = set()

    for i, d in enumerate(detected_sec):
        for j, g in enumerate(ground_truth_sec):
            if j not in matched_gt and abs(d - g) <= tolerance_sec:
                matched_gt.add(j)
                matched_det.add(i)
                break

    vp = len(matched_gt)
    fp = len(detected_sec) - len(matched_det)
    fn = len(ground_truth_sec) - vp

    precision = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    recall = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    fscore = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return vp, fp, fn, precision, recall, fscore


def plot_diffs(diffs: np.ndarray, cuts_frames: list[int], fps: float, video_name: str, out_path: Path):
    """Plots the inter-frame difference signal with vertical lines showing detected cuts."""
    times = np.arange(len(diffs)) / fps
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(times, diffs, linewidth=0.7, color="steelblue", label="Inter-frame Difference")
    for cf in cuts_frames:
        ax.axvline(x=cf / fps, color="red", linewidth=0.8, alpha=0.7)
    
    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_ylabel("Chi-Square Distance", fontweight="bold")
    ax.set_title(f"Inter-frame Difference Signal — {video_name}", fontsize=11, fontweight="bold")
    red_patch = mpatches.Patch(color="red", label="Detected Shot Boundary")
    ax.legend(handles=[ax.lines[0], red_patch], loc="upper right")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
