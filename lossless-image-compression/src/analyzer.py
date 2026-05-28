import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def to_gray_u8(img: Image.Image) -> np.ndarray:
    if img.mode == "P":
        img = img.convert("RGB")
    if img.mode in ("I;16", "I;16B", "I;16L"):
        a16 = np.asarray(img, dtype=np.uint16)
        return (a16 >> 8).astype(np.uint8)
    if img.mode == "L":
        return np.asarray(img, dtype=np.uint8)
    if img.mode != "RGB":
        img = img.convert("RGB")
    a = np.asarray(img, dtype=np.uint8)
    g = (0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]).round()
    return g.astype(np.uint8)


def shannon_entropy_from_hist(hist: np.ndarray) -> float:
    s = hist.sum()
    if s == 0:
        return 0.0
    p = hist[hist > 0].astype(np.float64) / float(s)
    return float(-(p * np.log2(p)).sum())


def save_gray(gray_u8: np.ndarray, out_path: Path):
    Image.fromarray(gray_u8, mode="L").save(out_path)


def save_hist(gray_u8: np.ndarray, out_path: Path, title: str):
    hist = np.bincount(gray_u8.ravel(), minlength=256)
    x = np.arange(256)
    plt.figure()
    plt.plot(x, hist, color="steelblue", linewidth=1.5)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Intensity Value (0..255)")
    plt.ylabel("Pixel Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return hist


def entropy_map(gray_u8: np.ndarray, window: int = 15, bins: int = 32) -> np.ndarray:
    if window % 2 == 0 or window < 3:
        raise ValueError("window size must be odd and >= 3")

    q = ((gray_u8.astype(np.uint16) * bins) // 256).astype(np.uint8)
    r = window // 2
    qp = np.pad(q, ((r, r), (r, r)), mode="reflect")
    H, W = gray_u8.shape
    area = float(window * window)
    log2_area = np.log2(area)

    out = np.zeros((H, W), dtype=np.float32)

    for k in range(bins):
        m = (qp == k).astype(np.uint16)
        ii = np.cumsum(np.cumsum(m, axis=0), axis=1)
        ii = np.pad(ii, ((1, 0), (1, 0)), mode="constant", constant_values=0)

        c = ii[window:, window:] - ii[:-window, window:] - ii[window:, :-window] + ii[:-window, :-window]
        c = c.astype(np.float32)

        mask = c > 0
        if np.any(mask):
            out[mask] += (c[mask] / area) * (np.log2(c[mask]) - log2_area)

    return -out


def save_entropy_heatmap(ent: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(8, 6))
    plt.imshow(ent, cmap="viridis")
    plt.title(title, fontsize=12, fontweight="bold")
    plt.colorbar(label="Local Entropy (bits)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
