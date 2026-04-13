# analysis_rawppg_lle/code/04_embed_lle_from_psd.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import LocallyLinearEmbedding, trustworthiness

from matplotlib.lines import Line2D
from matplotlib.colors import Normalize


META_COLS = {
    "subject", "segment", "t_start", "t_end", "t_center", "t_rel",
    "ppg_stream", "ppg_channel_1based", "fs", "window_s", "step_s",
    "bandpass", "low_hz", "high_hz"
}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def pick_feature_columns(df: pd.DataFrame) -> list:
    cols = []
    for c in df.columns:
        if c in META_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def add_segment_legend(ax, segments, seg_to_i, cmap, norm, title="segment"):
    handles = []
    for s in segments:
        i = seg_to_i[s]
        color = cmap(norm(i))
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=color, markersize=8, label=str(s)))
    ax.legend(handles=handles, title=title, loc="best", frameon=True)


def plot_scatter(df, xcol, ycol, out_png, title):
    ensure_dir(out_png.parent)

    segs = df["segment"].astype(str)
    segments = sorted(segs.unique().tolist())
    seg_to_i = {s: i for i, s in enumerate(segments)}
    c = segs.map(seg_to_i).to_numpy()

    cmap = plt.get_cmap("tab10") if len(segments) <= 10 else plt.get_cmap("tab20")
    norm = Normalize(vmin=0, vmax=max(len(segments) - 1, 1))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df[xcol], df[ycol], c=c, s=10, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    add_segment_legend(ax, segments, seg_to_i, cmap, norm)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_trajectories(df, xcol, ycol, out_png, title):
    """
    One line per subject (time-ordered within segment), points colored by segment.
    """
    ensure_dir(out_png.parent)

    segments = sorted(df["segment"].astype(str).unique().tolist())
    seg_to_i = {s: i for i, s in enumerate(segments)}
    df2 = df.copy()
    df2["seg_i"] = df2["segment"].astype(str).map(seg_to_i)

    cmap = plt.get_cmap("tab10") if len(segments) <= 10 else plt.get_cmap("tab20")
    norm = Normalize(vmin=0, vmax=max(len(segments) - 1, 1))

    fig, ax = plt.subplots(figsize=(8, 5))

    for subj, g in df2.sort_values(["subject", "segment", "t_rel"]).groupby("subject"):
        ax.plot(g[xcol].to_numpy(), g[ycol].to_numpy(), linewidth=1, alpha=0.5)

    ax.scatter(df2[xcol], df2[ycol], c=df2["seg_i"].to_numpy(), s=10, cmap=cmap, norm=norm)

    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    add_segment_legend(ax, segments, seg_to_i, cmap, norm)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="analysis_rawppg_lle/outputs/psd_features/rawppg_psd_features.csv",
                    help="CSV created by 03_psd_features_from_windows.py")
    ap.add_argument("--out_root", default="analysis_rawppg_lle/outputs/emb_psd_lle",
                    help="Output folder for embedding + plots")

    ap.add_argument("--scale", action="store_true",
                    help="Standardize features before LLE (often helps). Recommended: ON for PSD features.")
    ap.add_argument("--lle_neighbors", type=int, default=25)
    ap.add_argument("--lle_method", default="modified",
                    choices=["standard", "modified", "hessian", "ltsa"])
    ap.add_argument("--random_state", type=int, default=0)
    ap.add_argument("--tw_k", type=int, default=15)

    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    df = pd.read_csv(in_csv)
    for c in ["subject", "segment"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {in_csv}")

    feat_cols = pick_feature_columns(df)
    if len(feat_cols) < 2:
        raise RuntimeError(f"Not enough numeric feature cols found. Found: {feat_cols}")

    X = df[feat_cols].to_numpy(dtype=float)
    X[~np.isfinite(X)] = np.nan
    X = SimpleImputer(strategy="median").fit_transform(X)

    if args.scale:
        X = StandardScaler().fit_transform(X)

    print(f"Loaded: {in_csv}")
    print(f"Rows: {len(df)}")
    print(f"Features used ({len(feat_cols)}): first 10 -> {feat_cols[:10]}")
    print(f"LLE: neighbors={args.lle_neighbors}, method={args.lle_method}, scale={bool(args.scale)}")

    lle = LocallyLinearEmbedding(
        n_neighbors=int(args.lle_neighbors),
        n_components=2,
        method=str(args.lle_method),
        random_state=int(args.random_state),
    )
    Y = lle.fit_transform(X)
    tw = trustworthiness(X, Y, n_neighbors=int(args.tw_k), metric="euclidean")
    print(f"Trustworthiness(k={args.tw_k}): {tw:.4f}")

    # Save embedding CSV (keep all original columns + lle dims)
    out = df.copy()
    out["lle_1"] = Y[:, 0]
    out["lle_2"] = Y[:, 1]
    out_csv = out_root / "embedding_lle_psd.csv"
    out.to_csv(out_csv, index=False)

    # Plots
    plot_scatter(
        out, "lle_1", "lle_2",
        out_root / "lle_psd_scatter_by_segment.png",
        f"PPG PSD→LLE (neighbors={args.lle_neighbors}, method={args.lle_method}, bins≈64)"
    )
    plot_trajectories(
        out, "lle_1", "lle_2",
        out_root / "lle_psd_trajectories.png",
        "PPG PSD→LLE trajectories (each line = subject; points colored by segment)"
    )

    # Summary
    summary = out_root / "lle_psd_summary.txt"
    summary.write_text("\n".join([
        f"in_csv: {in_csv}",
        f"n_rows: {len(df)}",
        f"n_features: {len(feat_cols)}",
        f"scale: {bool(args.scale)}",
        f"lle_neighbors: {args.lle_neighbors}",
        f"lle_method: {args.lle_method}",
        f"trustworthiness(k={args.tw_k}): {tw:.4f}",
        "segments: " + ", ".join(sorted(df["segment"].astype(str).unique().tolist())),
        f"saved_embedding: {out_csv}",
        "saved_plots: lle_psd_scatter_by_segment.png, lle_psd_trajectories.png",
    ]))

    print(f"\nSaved embedding: {out_csv}")
    print(f"Saved plots in:  {out_root}")
    print(f"Saved summary:   {summary}")
    print("Done.")


if __name__ == "__main__":
    main()

"""
python3 analysis_rawppg_lle/code/04_embed_lle_from_psd.py \
  --in_csv analysis_rawppg_lle/outputs/psd_features/rawppg_psd_features.csv \
  --out_root analysis_rawppg_lle/outputs/emb_psd_lle \
  --lle_neighbors 25 --lle_method modified \
  --tw_k 15 \
  --scale

or for within subjectZ:

python3 analysis_rawppg_lle/code/04_embed_lle_from_psd.py \
  --in_csv analysis_rawppg_lle/outputs/psd_features/rawppg_psd_features_withinSubjectZ.csv \
  --out_root analysis_rawppg_lle/outputs/emb_psd_lle_withinSubjectZ \
  --lle_neighbors 25 --lle_method modified \
  --tw_k 15
"""