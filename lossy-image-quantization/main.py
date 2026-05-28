import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from skimage import metrics

sys.path.append(str(Path(__file__).parent / "src"))

from dct import apply_dct_blocks, reconstruct_from_coeffs
from quantization import (
    uniform_quantize, uniform_dequantize,
    deadzone_quantize, deadzone_dequantize,
    matrix_quantize, matrix_dequantize
)

# Reference Standard JPEG Luminance Matrix
Q_JPEG = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99]
], dtype=np.float64)


def run_pipeline(img_np, method, param):
    coeffs = apply_dct_blocks(img_np)
    
    if method == "Uniform":
        q = uniform_quantize(coeffs, param)
        coeffs_rec = uniform_dequantize(q, param)
    elif method == "Deadzone":
        q = deadzone_quantize(coeffs, param, alpha=1.5)
        coeffs_rec = deadzone_dequantize(q, param, alpha=1.5)
    elif method == "JPEG":
        q = matrix_quantize(coeffs, param)
        coeffs_rec = matrix_dequantize(q, param)
    else:
        raise ValueError(f"Unknown method {method}")
        
    recon = reconstruct_from_coeffs(coeffs_rec)
    
    psnr = metrics.peak_signal_noise_ratio(img_np.astype(np.uint8), recon, data_range=255)
    ssim = metrics.structural_similarity(img_np.astype(np.uint8), recon, data_range=255)
    sparsity = np.sum(q == 0) / q.size * 100
    
    return recon, q, psnr, ssim, sparsity


def generate_plots(img_np, results, out_dir):
    # 1. Image Reconstructions Comparison
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    fig.patch.set_facecolor("#FFFFFF")
    
    methods_order = ["Uniform", "Deadzone", "JPEG"]
    params_lbl = {
        "Uniform": [8, 16, 32],
        "Deadzone": [8, 16, 32],
        "JPEG": ["standard", "x2", "x4"]
    }
    
    for row_idx, method in enumerate(methods_order):
        # Original Image at index 0 of each row
        axes[row_idx, 0].imshow(img_np, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 0].set_title(f"Original", color="black", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")
        
        for col_idx, key in enumerate(params_lbl[method]):
            dict_key = f"{method} {key}"
            lbl = f"{method} {key}" if isinstance(key, str) else f"{method} $\\Delta$={key}"
            recon, _, psnr, ssim, sparsity = results[dict_key]
            
            axes[row_idx, col_idx + 1].imshow(recon, cmap="gray", vmin=0, vmax=255)
            axes[row_idx, col_idx + 1].set_title(
                f"{lbl}\nPSNR={psnr:.1f} dB | SSIM={ssim:.3f}\nZeros={sparsity:.1f}%",
                color="black", fontsize=10
            )
            axes[row_idx, col_idx + 1].axis("off")
            
    fig.suptitle("Quantization Reconstruction Comparison — DCT 8x8", color="black", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "reconstructions_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Histogram Comparison
    fig2 = plt.figure(figsize=(16, 12))
    fig2.patch.set_facecolor("#FFFFFF")
    gs = gridspec.GridSpec(3, 3, figure=fig2, wspace=0.3, hspace=0.4)
    
    plot_keys = [
        "Uniform 8", "Uniform 16", "Uniform 32",
        "Deadzone 8", "Deadzone 16", "Deadzone 32",
        "JPEG standard", "JPEG x2", "JPEG x4"
    ]
    colors = ["#2196F3", "#2196F3", "#2196F3", "#FF16E8", "#FF16E8", "#FF16E8", "#00BCD4", "#00BCD4", "#00BCD4"]
    
    for idx, (key, col) in enumerate(zip(plot_keys, colors)):
        ax = fig2.add_subplot(gs[idx])
        ax.set_facecolor("#FFFFFF")
        
        _, q, psnr, ssim, sparsity = results[key]
        q_flat = q.flatten()
        non_zero = q_flat[q_flat != 0]
        
        vmax = int(np.percentile(np.abs(non_zero), 99)) + 1 if len(non_zero) > 0 else 10
        bins = np.arange(-vmax, vmax + 2) - 0.5
        
        ax.hist(q_flat, bins=bins, color=col, alpha=0.75, edgecolor='none')
        zero_count = np.sum(q_flat == 0)
        ax.bar(0, zero_count, width=0.9, color="black", alpha=0.9, label=f"Zeros ({sparsity:.1f}%)", zorder=5)
        
        ax.set_yscale("log")
        ax.set_xlim(-vmax, vmax)
        ax.set_xlabel("Quantized Code", color="black", fontsize=9)
        ax.set_ylabel("Frequency (log)", color="black", fontsize=9)
        ax.set_title(f"{key}\nPSNR={psnr:.1f}dB | SSIM={ssim:.3f}", color="black", fontsize=10, fontweight="bold")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)
        
    fig2.suptitle("DCT Quantized Coefficient Distributions", color="black", fontsize=14, fontweight="bold")
    plt.savefig(out_dir / "coefficient_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 3. Sparsity vs. Quality Trade-off Plot
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    fig3.patch.set_facecolor("#FFFFFF")
    ax3.set_facecolor("#FFFFFF")
    
    markers = {"Uniform": "o-", "Deadzone": "s--", "JPEG": "^-."}
    plot_colors = {"Uniform": "#2196F3", "Deadzone": "#FF16E8", "JPEG": "#00BCD4"}
    
    for method in ["Uniform", "Deadzone", "JPEG"]:
        sub_keys = [k for k in results.keys() if k.startswith(method)]
        spars_list = [results[k][4] for k in sub_keys]
        ssim_list = [results[k][3] for k in sub_keys]
        
        # Sort by sparsity for clean line plot
        sorted_indices = np.argsort(spars_list)
        x_plot = np.array(spars_list)[sorted_indices]
        y_plot = np.array(ssim_list)[sorted_indices]
        
        ax3.plot(x_plot, y_plot, markers[method], color=plot_colors[method], label=method, linewidth=2, markersize=8)
        
        # Add labels to points
        for k in sub_keys:
            s = results[k][4]
            q_val = results[k][3]
            param_lbl = k.split()[-1]
            ax3.annotate(param_lbl, (s, q_val), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
            
    ax3.set_xlabel("Sparsity (Percentage of Zero Coefficients)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Quality Index (SSIM)", fontsize=11, fontweight="bold")
    ax3.set_title("Rate-Distortion Profile: Sparsity vs. SSIM", fontsize=13, fontweight="bold")
    ax3.set_xlim(35, 95)
    ax3.set_ylim(0.75, 1.0)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10, loc="lower left")
    
    plt.savefig(out_dir / "sparsity_vs_quality.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 4. Residual Errors Heatmaps (selected comparisons)
    fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
    fig4.patch.set_facecolor("#FFFFFF")
    
    comp_keys = ["Uniform 32", "Deadzone 32", "JPEG standard"]
    for idx, key in enumerate(comp_keys):
        recon, _, _, _, _ = results[key]
        residual = img_np.astype(np.float64) - recon.astype(np.float64)
        im = axes4[idx].imshow(residual, cmap="bwr", vmin=-40, vmax=40)
        axes4[idx].set_title(f"Residual Error — {key}", fontsize=11, fontweight="bold")
        axes4[idx].axis("off")
        plt.colorbar(im, ax=axes4[idx], fraction=0.046, pad=0.04)
        
    plt.suptitle("Spatial Distribution of Quantization Errors (Residuals)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "residuals_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Lossy DCT Quantization Pipeline")
    parser.add_argument("--image", type=str, default="data/image.png")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()
    
    img_path = Path(args.image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not img_path.exists():
        print(f"Error: Reference image {img_path} not found.")
        sys.exit(1)
        
    # Load grayscale reference image
    img = Image.open(img_path).convert("L")
    img = img.resize((512, 512), Image.Resampling.LANCZOS)
    img_np = np.array(img, dtype=np.float64)
    
    results = {}
    
    # 1. Run Uniform Quantification
    for delta in [8, 16, 32]:
        recon, q, psnr, ssim, sparsity = run_pipeline(img_np, "Uniform", delta)
        results[f"Uniform {delta}"] = (recon, q, psnr, ssim, sparsity)
        
    # 2. Run Deadzone Quantification
    for delta in [8, 16, 32]:
        recon, q, psnr, ssim, sparsity = run_pipeline(img_np, "Deadzone", delta)
        results[f"Deadzone {delta}"] = (recon, q, psnr, ssim, sparsity)
        
    # 3. Run JPEG Matrix Quantification
    recon, q, psnr, ssim, sparsity = run_pipeline(img_np, "JPEG", Q_JPEG)
    results["JPEG standard"] = (recon, q, psnr, ssim, sparsity)
    
    recon_x2, q_x2, psnr_x2, ssim_x2, sparsity_x2 = run_pipeline(img_np, "JPEG", Q_JPEG * 2)
    results["JPEG x2"] = (recon_x2, q_x2, psnr_x2, ssim_x2, sparsity_x2)
    
    recon_x4, q_x4, psnr_x4, ssim_x4, sparsity_x4 = run_pipeline(img_np, "JPEG", Q_JPEG * 4)
    results["JPEG x4"] = (recon_x4, q_x4, psnr_x4, ssim_x4, sparsity_x4)
    
    # Generate statistics table
    table_rows = []
    for key, (recon, q, psnr, ssim, sparsity) in results.items():
        parts = key.split()
        table_rows.append({
            "Quantizer": parts[0],
            "Parameter": parts[1],
            "PSNR (dB)": float(psnr),
            "SSIM": float(ssim),
            "Sparsity (% zeros)": float(sparsity)
        })
        
    df = pd.DataFrame(table_rows)
    csv_path = out_dir / "quantization_metrics.csv"
    md_path = out_dir / "quantization_metrics.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    
    print("=" * 60)
    print("LOSSY COMPRESSION QUANTIZATION PERFORMANCE SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\nSaved metrics to:\n- CSV: {csv_path.resolve()}\n- Markdown: {md_path.resolve()}")
    
    print("\nGenerating visual comparison plots...")
    generate_plots(img_np, results, out_dir)
    print(f"Saved visualization plots in: {out_dir.resolve()}")
    
    print("\nDone! Pipeline execution completed.")


if __name__ == "__main__":
    main()
