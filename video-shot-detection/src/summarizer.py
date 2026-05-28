import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path

def select_keyframes(frames: list[np.ndarray], shot_boundaries_frames: list[int], fps: float,
                     keyframes_per_sec: float = 0.5, min_keyframes: int = 1, max_keyframes: int = 5) -> list[tuple[int, np.ndarray]]:
    """
    Selects representative keyframes for each detected shot using KMeans clustering 
    on 3D HSV color histograms of the frames.
    """
    boundaries = [0] + list(shot_boundaries_frames) + [len(frames)]
    keyframes = []

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = min(boundaries[i + 1], len(frames))
        shot_frames = frames[start:end]
        if not shot_frames:
            continue

        duration_sec = (end - start) / fps
        k = max(min_keyframes, min(max_keyframes, int(duration_sec * keyframes_per_sec)))
        k = min(k, len(shot_frames))

        descriptors = []
        for f in shot_frames:
            # Downsample and convert to HSV to calculate color descriptor
            small = cv2.resize(f, (64, 36))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            descriptors.append(hist.flatten())

        descriptors = np.array(descriptors)

        # Single keyframe required or shot is too short
        if k == 1 or len(shot_frames) <= k:
            mid = len(shot_frames) // 2
            keyframes.append((start + mid, shot_frames[mid]))
            continue

        # Multiple keyframes: apply KMeans clustering
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(descriptors)

        selected = []
        for cluster_id in range(k):
            indices = np.where(km.labels_ == cluster_id)[0]
            if len(indices) == 0:
                continue
            center = km.cluster_centers_[cluster_id]
            dists = [np.linalg.norm(descriptors[j] - center) for j in indices]
            best = indices[np.argmin(dists)]
            
            descriptor_best = descriptors[best]
            too_similar = any(
                np.linalg.norm(descriptor_best - descriptors[s]) < 0.05
                for s in selected
            )
            if not too_similar:
                selected.append(best)
                keyframes.append((start + best, shot_frames[best]))

    keyframes.sort(key=lambda x: x[0])
    return keyframes


def save_mosaic(keyframes: list[tuple[int, np.ndarray]], video_name: str, out_path: Path):
    """Saves keyframes in a clean, visual grid mosaic."""
    if not keyframes:
        return
    
    imgs = [cv2.cvtColor(cv2.resize(f, (213, 120)), cv2.COLOR_BGR2RGB) for _, f in keyframes]
    n = len(imgs)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2))
    axes = np.array(axes).reshape(-1) if n > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < len(imgs):
            ax.imshow(imgs[i])
            ax.set_title(f"Keyframe {i+1}", fontsize=8, fontweight="bold")
        ax.axis("off")

    fig.suptitle(f"Keyframe Summary Mosaic — {video_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
