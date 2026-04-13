# analysis_manifold/02_embed_umap_lle.py
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


# -----------------------------
# Helpers
# -----------------------------
META_COLS_DEFAULT = {
    "subject", "segment", "t_center", "t_rel",
    "valid_pct", "n_samples", "beats_med",
    "t_start", "t_end"
}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def pick_feature_columns(df: pd.DataFrame, meta_cols: set) -> list:
    """Pick numeric columns excluding meta columns."""
    cols = []
    for c in df.columns:
        if c in meta_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def prep_X(df: pd.DataFrame, feature_cols: list, do_scale: bool = False) -> np.ndarray:
    """
    Prepare feature matrix:
    - convert to float
    - replace non-finite with NaN
    - median impute
    - optional StandardScaler
    """
    X = df[feature_cols].to_numpy(dtype=float)
    X[~np.isfinite(X)] = np.nan
    X = SimpleImputer(strategy="median").fit_transform(X)
    if do_scale:
        X = StandardScaler().fit_transform(X)
    return X


def save_embedding_csv(df_meta: pd.DataFrame, Y: np.ndarray, out_csv: Path, prefix: str):
    out = df_meta.copy()
    out[f"{prefix}_1"] = Y[:, 0]
    out[f"{prefix}_2"] = Y[:, 1]
    out.to_csv(out_csv, index=False)


def _add_segment_legend(ax, uniq, seg_to_i, cmap, norm, title="segment"):
    """Legend that matches the scatter colors."""
    handles = []
    for s in uniq:
        i = seg_to_i[s]
        color = cmap(norm(i))
        handles.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color, markersize=8, label=str(s))
        )
    ax.legend(handles=handles, title=title, loc="best", frameon=True)


def plot_scatter(df_emb: pd.DataFrame, xcol: str, ycol: str, out_png: Path, title: str):
    ensure_dir(out_png.parent)

    segs = df_emb["segment"].astype(str)
    uniq = sorted(segs.unique().tolist())
    seg_to_i = {s: i for i, s in enumerate(uniq)}
    c = segs.map(seg_to_i).to_numpy()

    cmap = plt.get_cmap("tab10") if len(uniq) <= 10 else plt.get_cmap("tab20")
    norm = Normalize(vmin=0, vmax=max(len(uniq) - 1, 1))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df_emb[xcol], df_emb[ycol], c=c, s=10, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)

    _add_segment_legend(ax, uniq, seg_to_i, cmap, norm, title="segment")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_trajectories(df_emb: pd.DataFrame, xcol: str, ycol: str, out_png: Path, title: str):
    """
    Lines show each subject path (in time order);
    points colored by segment.
    """
    ensure_dir(out_png.parent)

    segs = df_emb["segment"].astype(str)
    uniq = sorted(segs.unique().tolist())
    seg_to_i = {s: i for i, s in enumerate(uniq)}

    df_plot = df_emb.copy()
    df_plot["seg_i"] = df_plot["segment"].astype(str).map(seg_to_i)

    cmap = plt.get_cmap("tab10") if len(uniq) <= 10 else plt.get_cmap("tab20")
    norm = Normalize(vmin=0, vmax=max(len(uniq) - 1, 1))

    fig, ax = plt.subplots(figsize=(8, 5))

    # draw subject paths
    if "t_rel" in df_plot.columns:
        order_cols = ["subject", "t_rel"]
    else:
        order_cols = ["subject", "t_center"] if "t_center" in df_plot.columns else ["subject"]

    for subj, g in df_plot.sort_values(order_cols).groupby("subject"):
        ax.plot(g[xcol].to_numpy(), g[ycol].to_numpy(), linewidth=1, alpha=0.6)

    # points colored by segment
    ax.scatter(df_plot[xcol], df_plot[ycol],
               c=df_plot["seg_i"].to_numpy(), s=10, cmap=cmap, norm=norm)

    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)

    _add_segment_legend(ax, uniq, seg_to_i, cmap, norm, title="segment")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="analysis_outputs/manifold_v2/features_windows_within_z.csv",
                    help="Input windows CSV (within_z or global_z)")
    ap.add_argument("--out_root", default="analysis_outputs/manifold_v2/embeddings",
                    help="Output folder for embeddings + plots")
    ap.add_argument("--min_valid_pct", type=float, default=0.0,
                    help="Drop windows below this valid_pct (0 keeps all)")

    ap.add_argument("--feature_cols", default="",
                    help="Comma-separated feature cols. If empty, auto-pick numeric cols (excluding meta).")
    ap.add_argument("--scale", action="store_true",
                    help="Apply StandardScaler (usually NOT needed for *_z CSV).")

    # LLE
    ap.add_argument("--lle_neighbors", type=int, default=15)
    ap.add_argument("--lle_method", default="modified",
                    choices=["standard", "modified", "hessian", "ltsa"])
    ap.add_argument("--random_state", type=int, default=0)

    # UMAP
    ap.add_argument("--umap_neighbors", type=int, default=15)
    ap.add_argument("--umap_min_dist", type=float, default=0.1)
    ap.add_argument("--umap_metric", default="euclidean")

    ap.add_argument("--tw_k", type=int, default=15,
                    help="k for trustworthiness()")

    ap.add_argument(
        "--drop_outlier_eb3_baseline",
        action="store_true",
        help="Drop EmotiBit_3 baseline outlier windows by t_center (sanity check)"
    )

    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    df = pd.read_csv(in_csv)

    # required
    for c in ["subject", "segment", "t_center", "t_rel"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {in_csv}")

    if args.min_valid_pct > 0 and "valid_pct" in df.columns:
        df = df[df["valid_pct"] >= float(args.min_valid_pct)].reset_index(drop=True)

    # ---- OPTIONAL outlier drop (fixed indentation + only runs if flag set) ----
    if args.drop_outlier_eb3_baseline:
        before = len(df)
        bad = (
            (df["subject"] == "EmotiBit_3") &
            (df["segment"] == "baseline_warmup") &
            ((df["t_center"] >= 62931.0) | (df["t_center"] == 62896.0))
        )
        df = df[~bad].reset_index(drop=True)
        print(f"Dropped {before - len(df)} rows (EB3 baseline outlier windows).")

    meta_cols = set(META_COLS_DEFAULT)
    df_meta = df[[c for c in df.columns if c in meta_cols]].copy()

    # pick features
    if args.feature_cols.strip():
        feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
    else:
        feature_cols = pick_feature_columns(df, meta_cols=meta_cols)

    if len(feature_cols) < 2:
        raise ValueError(f"Not enough feature columns: {feature_cols}")

    print(f"Loaded: {in_csv}")
    print(f"Rows: {len(df)}")
    print(f"Using {len(feature_cols)} feature cols:")
    print("  " + ", ".join(feature_cols))

    # build X
    X = prep_X(df, feature_cols, do_scale=bool(args.scale))

    # -----------------------------
    # LLE
    # -----------------------------
    lle = LocallyLinearEmbedding(
        n_neighbors=int(args.lle_neighbors),
        n_components=2,
        method=args.lle_method,
        random_state=int(args.random_state),
    )
    Y_lle = lle.fit_transform(X)
    tw_lle = trustworthiness(X, Y_lle, n_neighbors=int(args.tw_k), metric="euclidean")

    lle_csv = out_root / "embedding_lle.csv"
    save_embedding_csv(df_meta, Y_lle, lle_csv, "lle")
    print(f"\nSaved LLE embedding: {lle_csv}")
    print(f"LLE trustworthiness(k={args.tw_k}): {tw_lle:.4f}")

    df_lle = pd.read_csv(lle_csv)
    plot_scatter(df_lle, "lle_1", "lle_2",
                 out_root / "lle_scatter_by_segment.png",
                 f"LLE scatter (neighbors={args.lle_neighbors}, method={args.lle_method})")
    plot_trajectories(df_lle, "lle_1", "lle_2",
                      out_root / "lle_trajectories.png",
                      "LLE trajectories (each line = subject; points = windows)")

    # -----------------------------
    # UMAP (runs only if installed)
    # -----------------------------
    tw_umap = None
    try:
        import umap  # pip install umap-learn
        um = umap.UMAP(
            n_neighbors=int(args.umap_neighbors),
            min_dist=float(args.umap_min_dist),
            n_components=2,
            metric=str(args.umap_metric),
            random_state=int(args.random_state),
        )
        Y_umap = um.fit_transform(X)
        tw_umap = trustworthiness(X, Y_umap, n_neighbors=int(args.tw_k), metric="euclidean")

        umap_csv = out_root / "embedding_umap.csv"
        save_embedding_csv(df_meta, Y_umap, umap_csv, "umap")

        print(f"\nSaved UMAP embedding: {umap_csv}")
        print(f"UMAP trustworthiness(k={args.tw_k}): {tw_umap:.4f}")

        df_umap = pd.read_csv(umap_csv)
        plot_scatter(df_umap, "umap_1", "umap_2",
                     out_root / "umap_scatter_by_segment.png",
                     f"UMAP scatter (neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist})")
        plot_trajectories(df_umap, "umap_1", "umap_2",
                          out_root / "umap_trajectories.png",
                          "UMAP trajectories (each line = subject; points = windows)")

    except Exception as e:
        print("\nUMAP not run (umap-learn missing or failed).")
        print("Reason:", repr(e))
        print("Install with: pip install umap-learn")

    # summary text
    summary_txt = out_root / "embedding_summary.txt"
    lines = [
        f"input_csv: {in_csv}",
        f"n_rows: {len(df)}",
        f"n_features: {len(feature_cols)}",
        f"features: {', '.join(feature_cols)}",
        f"LLE: neighbors={args.lle_neighbors}, method={args.lle_method}, trustworthiness(k={args.tw_k})={tw_lle:.4f}",
    ]
    if tw_umap is not None:
        lines.append(
            f"UMAP: neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist}, metric={args.umap_metric}, trustworthiness(k={args.tw_k})={tw_umap:.4f}"
        )
    else:
        lines.append("UMAP: not run")

    summary_txt.write_text("\n".join(lines))
    print(f"\nSaved: {summary_txt}")
    print(f"Done. Outputs in: {out_root}")


if __name__ == "__main__":
    main()

"""
Example run (within-z):
python3 analysis_manifold/02_embed_umap_lle.py \
  --in_csv analysis_outputs/manifold_v2/features_windows_within_z.csv \
  --out_root analysis_outputs/manifold_v2/emb_within \
  --lle_neighbors 15 --umap_neighbors 15 --umap_min_dist 0.1 \
  --tw_k 15

Example run with EB3 baseline outlier drop:
python3 analysis_manifold/02_embed_umap_lle.py \
  --in_csv analysis_outputs/manifold_v2/features_windows_within_z.csv \
  --out_root analysis_outputs/manifold_v2/emb_within_noeb3outlier \
  --lle_neighbors 15 --umap_neighbors 15 --umap_min_dist 0.1 \
  --tw_k 15 \
  --drop_outlier_eb3_baseline
"""