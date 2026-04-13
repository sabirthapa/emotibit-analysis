# analysis_rawppg_lle/code/02_embed_lle_raw.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.manifold import LocallyLinearEmbedding, trustworthiness
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


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
    One line per subject, ordered by t_rel within each segment.
    (Simple view of motion through embedding.)
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
    ap.add_argument("--X_npy", default="analysis_rawppg_lle/outputs/rawppg_windows_X.npy",
                    help="Saved windows matrix (n_windows, win_n)")
    ap.add_argument("--meta_csv", default="analysis_rawppg_lle/outputs/rawppg_windows_meta.csv",
                    help="Saved metadata CSV aligned with X rows")
    ap.add_argument("--out_root", default="analysis_rawppg_lle/outputs/emb_raw",
                    help="Output folder for embedding + plots")

    ap.add_argument("--lle_neighbors", type=int, default=25)
    ap.add_argument("--lle_method", default="modified",
                    choices=["standard", "modified", "hessian", "ltsa"])
    ap.add_argument("--n_components", type=int, default=2, choices=[2, 3],
                    help="2D or 3D embedding")
    ap.add_argument("--random_state", type=int, default=0)

    ap.add_argument("--tw_k", type=int, default=15)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ensure_dir(out_root)

    X = np.load(args.X_npy)  # [N, D]
    meta = pd.read_csv(args.meta_csv)

    if len(meta) != X.shape[0]:
        raise RuntimeError(f"Meta rows ({len(meta)}) != X rows ({X.shape[0]})")

    # LLE can be sensitive to NaNs/inf: guard
    X = X.astype(float)
    X[~np.isfinite(X)] = np.nan
    X = SimpleImputer(strategy="median").fit_transform(X)

    print(f"Loaded X: {args.X_npy} shape={X.shape}")
    print(f"Loaded meta: {args.meta_csv} rows={len(meta)}")

    lle = LocallyLinearEmbedding(
        n_neighbors=int(args.lle_neighbors),
        n_components=int(args.n_components),
        method=str(args.lle_method),
        random_state=int(args.random_state),
    )
    Y = lle.fit_transform(X)

    tw = trustworthiness(X, Y, n_neighbors=int(args.tw_k), metric="euclidean")
    print(f"LLE trustworthiness(k={args.tw_k}): {tw:.4f}")

    # Save embedding
    emb = meta.copy()
    if args.n_components == 2:
        emb["lle_1"] = Y[:, 0]
        emb["lle_2"] = Y[:, 1]
        out_csv = out_root / "embedding_lle_raw_2d.csv"
    else:
        emb["lle_1"] = Y[:, 0]
        emb["lle_2"] = Y[:, 1]
        emb["lle_3"] = Y[:, 2]
        out_csv = out_root / "embedding_lle_raw_3d.csv"

    emb.to_csv(out_csv, index=False)
    print(f"Saved embedding: {out_csv}")

    # Save summary
    summary = out_root / "lle_raw_summary.txt"
    summary.write_text("\n".join([
        f"X_npy: {args.X_npy}",
        f"meta_csv: {args.meta_csv}",
        f"n_windows: {X.shape[0]}",
        f"window_dim: {X.shape[1]}",
        f"lle_neighbors: {args.lle_neighbors}",
        f"lle_method: {args.lle_method}",
        f"n_components: {args.n_components}",
        f"trustworthiness(k={args.tw_k}): {tw:.4f}",
        "segments: " + ", ".join(sorted(emb['segment'].astype(str).unique().tolist()))
    ]))
    print(f"Saved: {summary}")

    # Plots (2D only)
    if args.n_components == 2:
        plot_scatter(
            emb, "lle_1", "lle_2",
            out_root / "lle_raw_scatter_by_segment.png",
            f"Raw PPG LLE (neighbors={args.lle_neighbors}, method={args.lle_method})"
        )
        plot_trajectories(
            emb, "lle_1", "lle_2",
            out_root / "lle_raw_trajectories.png",
            "Raw PPG LLE trajectories (each line = subject)"
        )
        print("Saved plots (2D).")
    else:
        print("3D embedding saved (no 3D plot in this script).")

    print("Done.")


if __name__ == "__main__":
    main()

"""
python3 analysis_rawppg_lle/code/02_embed_lle_raw.py \
  --X_npy analysis_rawppg_lle/outputs/rawppg_windows_X.npy \
  --meta_csv analysis_rawppg_lle/outputs/rawppg_windows_meta.csv \
  --out_root analysis_rawppg_lle/outputs/emb_raw \
  --lle_neighbors 25 --lle_method modified \
  --tw_k 15 --n_components 2
"""