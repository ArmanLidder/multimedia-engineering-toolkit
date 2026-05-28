import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# Add src to path just in case
sys.path.append(str(Path(__file__).parent / "src"))

from compressor import LZW12, LZ77, RLE16, RawImage
from analyzer import to_gray_u8, shannon_entropy_from_hist, save_gray, save_hist, entropy_map, save_entropy_heatmap


def load_image1_rgb(path: Path) -> RawImage:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    payload = img.tobytes()
    return RawImage(path.name, path, w, h, "RGB", payload, {})


def load_image2_palette(path: Path) -> RawImage:
    img = Image.open(path)
    w, h = img.size
    extra = {}
    if img.mode == "P":
        idx = np.asarray(img, dtype=np.uint8)
        payload = idx.tobytes()
        extra["palette"] = img.getpalette()
        extra["mode"] = "P"
    else:
        payload = img.tobytes()
        extra["mode"] = img.mode
    return RawImage(path.name, path, w, h, img.mode, payload, extra)


def load_image3_u16(path: Path) -> RawImage:
    img = Image.open(path)
    w, h = img.size
    if img.mode not in ("I;16", "I;16B", "I;16L"):
        img = img.convert("I;16")
    arr = np.asarray(img, dtype=np.uint16)
    payload = arr.tobytes()
    return RawImage(path.name, path, w, h, "I;16", payload, {"shape": (h, w)})


def bpp(total_bytes: int, w: int, h: int) -> float:
    return (total_bytes * 8.0) / float(w * h)


def run_one_compression(raw: RawImage, method_name: str, compressor) -> dict:
    original_bytes = raw.payload
    original_size = len(original_bytes)

    if method_name == "LZW":
        comp = compressor.compress(original_bytes)
        dec = compressor.decompress(comp)
        ok = dec == original_bytes
        comp_size = len(comp)
    elif method_name == "LZ77":
        comp = compressor.compress(original_bytes)
        dec = compressor.decompress(comp)
        ok = dec == original_bytes
        comp_size = len(comp)
    elif method_name == "RLE":
        shape = raw.extra["shape"]
        arr = np.frombuffer(original_bytes, dtype=np.uint16).reshape(shape)
        comp = compressor.compress(arr)
        dec_arr = compressor.decompress(comp, shape)
        ok = dec_arr.tobytes() == original_bytes
        comp_size = len(comp)
    else:
        raise ValueError(f"Unknown compression method: {method_name}")

    ratio = (original_size / comp_size) if comp_size else float("inf")
    pct = (1.0 - (comp_size / original_size)) * 100.0 if original_size else 0.0

    return {
        "image": raw.name,
        "method": method_name,
        "mode": raw.mode,
        "dimensions": f"{raw.height}x{raw.width}",
        "original_bytes": int(original_size),
        "compressed_bytes": int(comp_size),
        "compression_percent": float(pct),
        "compression_ratio": float(ratio),
        "original_bpp": float(bpp(original_size, raw.width, raw.height)),
        "compressed_bpp": float(bpp(comp_size, raw.width, raw.height)),
        "lossless_ok": bool(ok),
    }


def analyze_entropy(img_path: Path, out_dir: Path, window: int, bins: int) -> dict:
    img = Image.open(img_path)
    gray = to_gray_u8(img)

    img_out = out_dir / img_path.stem
    img_out.mkdir(parents=True, exist_ok=True)

    save_gray(gray, img_out / f"{img_path.stem}_gray.png")
    hist = save_hist(gray, img_out / f"{img_path.stem}_hist_gray.png", f"{img_path.stem} Grayscale Histogram")
    ent_global = shannon_entropy_from_hist(hist)
    ent_map = entropy_map(gray, window=window, bins=bins)
    
    save_entropy_heatmap(
        ent_map, 
        img_out / f"{img_path.stem}_entropy_map.png",
        f"{img_path.stem} Local Entropy ({window}x{window}, {bins} bins)"
    )

    return {
        "file": img_path.name,
        "mode": img.mode,
        "height": int(gray.shape[0]),
        "width": int(gray.shape[1]),
        "gray_min": int(gray.min()),
        "gray_max": int(gray.max()),
        "unique_gray_values": int(np.unique(gray).size),
        "global_entropy_bits": float(ent_global),
        "mean_local_entropy_bits": float(ent_map.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Lossless Image Compression Suite")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--window", type=int, default=15, help="Window size for local entropy")
    parser.add_argument("--bins", type=int, default=32, help="Number of bins for local entropy")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image1 = data_dir / "image1_natural.png"
    image2 = data_dir / "image2_synthetic.png"
    image3 = data_dir / "image3_binary.png"

    # Verify data files exist
    for p in [image1, image2, image3]:
        if not p.exists():
            print(f"Error: Required file {p} not found.")
            sys.exit(1)

    print("=" * 60)
    print("1. RUNNING IMAGE ENTROPY & SPATIAL COMPLEXITY ANALYSIS")
    print("=" * 60)
    
    entropy_results = []
    for p in [image1, image2, image3]:
        print(f"Analyzing {p.name}...")
        res = analyze_entropy(p, out_dir, args.window, args.bins)
        entropy_results.append(res)
    
    df_entropy = pd.DataFrame(entropy_results)
    entropy_csv = out_dir / "entropy_analysis.csv"
    df_entropy.to_csv(entropy_csv, index=False)
    print(df_entropy.to_string(index=False))
    print(f"\nSaved entropy analysis to: {entropy_csv.resolve()}\n")

    print("=" * 60)
    print("2. RUNNING LOSSLESS COMPRESSION BENCHMARKS")
    print("=" * 60)

    # Define the benchmarks
    # LZW on Natural, LZ77 on Synthetic, RLE on Binary
    raws = [
        ("LZW", load_image1_rgb(image1), LZW12()),
        ("LZ77", load_image2_palette(image2), LZ77(window_size=4096, max_match=32)),
        ("RLE", load_image3_u16(image3), RLE16()),
    ]

    compression_results = []
    for method_name, raw, comp in raws:
        print(f"Compressing {raw.name} using {method_name}...")
        res = run_one_compression(raw, method_name, comp)
        compression_results.append(res)

    df_comp = pd.DataFrame(compression_results)
    df_comp = df_comp[[
        "image", "method", "mode", "dimensions",
        "original_bytes", "compressed_bytes",
        "compression_percent", "compression_ratio",
        "original_bpp", "compressed_bpp",
        "lossless_ok"
    ]]

    comp_csv = out_dir / "compression_results.csv"
    comp_md = out_dir / "compression_results.md"
    df_comp.to_csv(comp_csv, index=False)
    comp_md.write_text(df_comp.to_markdown(index=False), encoding="utf-8")

    print("\n" + df_comp.to_string(index=False))
    print(f"\nSaved compression results to:\n- CSV: {comp_csv.resolve()}\n- Markdown: {comp_md.resolve()}")
    print("\nDone! All analyses ran successfully.")


if __name__ == "__main__":
    main()
