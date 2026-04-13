# analysis_manifold/03_plot_lle_clean.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def segment_color_map(seg_order):
    """
    Return dict: segment -> RGBA color
    Uses a stable colormap so colors are consistent across plots.
    """
    cmap = plt.get_cmap("tab10") if len(seg_order) <= 10 else plt.get_cmap("tab20")
    seg2color = {seg: cmap(i) for i, seg in enumerate(seg_order)}
    return seg2color


def add_segment_legend(fig_or_ax, seg_order, seg2color, title="segment", loc="lower center"):
    """
    Add a legend showing segment -> color.
    Works for fig (global legend) or ax (axes legend).
    """
    handles = []
    for seg in seg_order:
        handles.append(
            Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=seg2color[seg], markeredgecolor="k",
                   markersize=7, label=seg)
        )

    # If you pass a Figure, use fig.legend; if Axes, use ax.legend
    if hasattr(fig_or_ax, "legend") and not hasattr(fig_or_ax, "get_legend_handles_labels"):
        # it's a Figure
        fig_or_ax.legend(handles=handles, title=title, loc=loc, ncol=len(seg_order), frameon=True)
    else:
        # it's an Axes
        fig_or_ax.legend(handles=handles, title=title, loc="best", frameon=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_csv", required=True, help="Path to embedding_lle.csv (or embedding_umap.csv)")
    ap.add_argument("--out_root", required=True, help="Where to save clean plots")
    ap.add_argument("--xcol", default="lle_1", help="Embedding x column (lle_1 or umap_1)")
    ap.add_argument("--ycol", default="lle_2", help="Embedding y column (lle_2 or umap_2)")
    ap.add_argument("--segments", default="baseline_warmup,meditation_2A,meditation_2B",
                    help="Comma-separated segment order")
    args = ap.parse_args()

    emb_csv = Path(args.emb_csv)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    df = pd.read_csv(emb_csv)

    for c in ["subject", "segment", args.xcol, args.ycol]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {emb_csv}")

    seg_order = [s.strip() for s in args.segments.split(",") if s.strip()]
    df["segment"] = df["segment"].astype(str)
    df["subject"] = df["subject"].astype(str)

    # Keep only segments we care about (and in order)
    df = df[df["segment"].isin(seg_order)].copy()
    df["segment"] = pd.Categorical(df["segment"], categories=seg_order, ordered=True)

    seg2color = segment_color_map(seg_order)

    # ---------- Plot A: small multiples (one subject per panel) ----------
    subjects = sorted(df["subject"].unique().tolist())
    n = len(subjects)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(4 * ncols, 3.5 * nrows))

    for i, subj in enumerate(subjects, start=1):
        ax = plt.subplot(nrows, ncols, i)
        sort_cols = ["segment", "t_rel"] if "t_rel" in df.columns else ["segment"]
        g = df[df["subject"] == subj].sort_values(sort_cols)

        # plot each segment separately (no cross-segment line connections)
        for seg in seg_order:
            gg = g[g["segment"] == seg]
            if gg.empty:
                continue
            color = seg2color[seg]
            ax.plot(gg[args.xcol].to_numpy(), gg[args.ycol].to_numpy(),
                    linewidth=1, color=color, alpha=0.9)
            ax.scatter(gg[args.xcol], gg[args.ycol],
                       s=10, color=color, alpha=0.9)

        ax.set_title(subj)
        ax.set_xlabel(args.xcol)
        ax.set_ylabel(args.ycol)

    fig.suptitle("LLE small multiples (each panel = one subject; segments plotted separately)", y=1.02)

    # Global legend for the whole figure
    add_segment_legend(fig, seg_order, seg2color, title="segment", loc="lower center")

    fig.tight_layout()
    out_a = out_root / "lle_small_multiples.png"
    fig.savefig(out_a, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_a)

    # ---------- Plot B: centroids + arrows (baseline -> 2A -> 2B) ----------
    fig, ax = plt.subplots(figsize=(7, 5))

    for subj in subjects:
        g = df[df["subject"] == subj]

        cents = []
        for seg in seg_order:
            gg = g[g["segment"] == seg]
            if gg.empty:
                cents.append(None)
                continue
            cx = float(np.nanmean(gg[args.xcol].to_numpy()))
            cy = float(np.nanmean(gg[args.ycol].to_numpy()))
            cents.append((cx, cy))

        # plot centroids (colored by segment)
        for seg_i, seg in enumerate(seg_order):
            if seg_i < len(cents) and cents[seg_i] is not None:
                cx, cy = cents[seg_i]
                ax.scatter([cx], [cy], s=35, color=seg2color[seg], alpha=0.9)

        # arrows baseline->2A, 2A->2B
        for a, b in [(0, 1), (1, 2)]:
            if a < len(cents) and b < len(cents) and cents[a] is not None and cents[b] is not None:
                x0, y0 = cents[a]
                x1, y1 = cents[b]
                ax.arrow(x0, y0, x1 - x0, y1 - y0,
                         length_includes_head=True, head_width=0.02,
                         alpha=0.6, color="gray")

    ax.set_title("LLE centroids per subject (baseline → 2A → 2B)")
    ax.set_xlabel(args.xcol)
    ax.set_ylabel(args.ycol)

    # Legend on this plot too
    add_segment_legend(ax, seg_order, seg2color, title="segment")

    fig.tight_layout()
    out_b = out_root / "lle_centroids_arrows.png"
    fig.savefig(out_b, dpi=150)
    plt.close(fig)
    print("Saved:", out_b)

    print("\nDone. Clean plots in:", out_root)


if __name__ == "__main__":
    main()

"""
run:
python3 analysis_manifold/03_plot_lle_clean.py \
  --emb_csv analysis_outputs/manifold_v2/emb_within/embedding_lle.csv \
  --out_root analysis_outputs/manifold_v2/emb_within/plots_clean \
  --xcol lle_1 --ycol lle_2
"""