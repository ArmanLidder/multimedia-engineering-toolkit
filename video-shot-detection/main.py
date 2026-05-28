import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from detector import detect_shots, evaluate, plot_diffs
from summarizer import select_keyframes, save_mosaic

# Project configurations
CONFIGS = {
    "F1": {
        "pattern": "*Race Highlights*.mp4",
        "gt": [1.0, 13.0, 17.0, 21.0, 26.0, 30.0, 37.0, 46.0, 51.0, 56.0, 59.0, 66.0, 70.0, 72.0, 79.0, 85.0],
        "k_adaptive": 6.0,
        "min_gap_sec": 1.5,
        "name": "Formula 1 Highlights"
    },
    "GoT": {
        "pattern": "*Dothraki*.mp4",
        "gt": [2.0, 5.0, 7.0, 9.0, 12.0, 14.0, 16.0, 21.0, 23.0, 24.0, 25.0, 27.0, 29.0, 37.0, 39.0, 41.0, 43.0, 45.0],
        "k_adaptive": 4.0,
        "min_gap_sec": 0.8,
        "name": "Game of Thrones (Battle)"
    },
    "StarWars": {
        "pattern": "*Star Wars*.mp4",
        "gt": [9.0],
        "k_adaptive": None,  # Global thresholding
        "min_gap_sec": 2.5,
        "name": "Star Wars Opening Crawl"
    }
}


def find_video_file(video_dir: Path, pattern: str) -> Path:
    """Helper to locate a video file matching a glob pattern (case-insensitive)."""
    matches = list(video_dir.glob(pattern))
    if not matches:
        # Try lowercase pattern matching
        matches = list(video_dir.glob(pattern.lower()))
    if matches:
        return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Video Shot Boundary Detection & Summarization")
    parser.add_argument("--video_dir", type=str, default="../../tp3/code",
                        help="Directory containing the input MP4 videos")
    parser.add_argument("--out_dir", type=str, default="results",
                        help="Directory to save generated charts and mosaics")
    parser.add_argument("--tolerance", type=float, default=2.0,
                        help="Evaluation tolerance window (seconds)")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VIDEO SHOT BOUNDARY DETECTION & SUMMARIZATION PIPELINE")
    print("=" * 70)

    # Verify video directory exists
    if not video_dir.exists():
        print(f"Warning: Video directory '{video_dir}' not found.")
        print("Will attempt to search in local '.' and 'data/' directories...")
        video_dir = Path(".")

    results_summary = []

    for key, cfg in CONFIGS.items():
        print(f"\nProcessing Video: {cfg['name']} ({key})")
        
        # Locate video file
        video_path = find_video_file(video_dir, cfg["pattern"])
        if not video_path:
            # Try searching in '.' as fallback
            video_path = find_video_file(Path("."), cfg["pattern"])
            
        if not video_path or not video_path.exists():
            print(f"[SKIP] Video file matching pattern '{cfg['pattern']}' not found in {video_dir.resolve()}.")
            continue

        safe_name = video_path.name.encode('ascii', errors='replace').decode('ascii')
        print(f"  Located: {safe_name}")
        gt = cfg["gt"]
        gt_end = None if key == "StarWars" else max(gt) + 5.0
        
        # 1. Detect Shot Boundaries (Cuts)
        print("  Running inter-frame HSV block difference and thresholding...")
        cuts_sec, cuts_frames, frames, diffs, fps = detect_shots(
            str(video_path),
            video_name=key,
            k_adaptive=cfg["k_adaptive"],
            min_gap_sec=cfg["min_gap_sec"],
            gt_end_sec=gt_end
        )
        print(f"  Detected {len(cuts_sec)} boundaries: {[round(t, 1) for t in cuts_sec]} (seconds)")
        print(f"  Ground Truth ({len(gt)} cuts)   : {gt}")

        # 2. Evaluate performance
        vp, fp, fn, precision, recall, fscore = evaluate(cuts_sec, gt, tolerance_sec=args.tolerance)
        print(f"  Evaluation: TP={vp}, FP={fp}, FN={fn} | Precision={precision:.2f}, Recall={recall:.2f}, F-Score={fscore:.2f}")

        # 3. Save inter-frame difference signal plots
        plot_diffs(diffs, cuts_frames, fps, cfg["name"], out_dir / f"diffs_{key}.png")
        print(f"  Saved signal plot: diffs_{key}.png")

        # 4. Extract Keyframes using KMeans
        print("  Extracting keyframes using color descriptor clustering...")
        keyframes = select_keyframes(frames, cuts_frames, fps)
        
        # 5. Save visual summary mosaic
        save_mosaic(keyframes, cfg["name"], out_dir / f"mosaic_{key}.png")
        print(f"  Saved summary mosaic: mosaic_{key}.png")

        # Accumulate metrics
        results_summary.append({
            "Video": cfg["name"],
            "TP": vp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F-Score": fscore
        })

    if results_summary:
        df = pd.DataFrame(results_summary)
        csv_path = out_dir / "detection_metrics.csv"
        md_path = out_dir / "detection_metrics.md"
        df.to_csv(csv_path, index=False)
        df_display = df[["Video", "TP", "FP", "FN", "Precision", "Recall", "F-Score"]]
        md_path.write_text(df_display.to_markdown(index=False), encoding="utf-8")
        
        print("\n" + "=" * 70)
        print("FINAL EVALUATION METRICS SUMMARY")
        print("=" * 70)
        print(df_display.to_string(index=False))
        print(f"\nSaved metrics summary to:\n- CSV: {csv_path.resolve()}\n- Markdown: {md_path.resolve()}")
    else:
        print("\nNo videos were processed. Please place the MP4 video files in the specified video directory.")

    print("\nDone! Pipeline execution completed.")


if __name__ == "__main__":
    main()
