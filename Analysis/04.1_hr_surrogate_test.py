import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def pairwise_mean_r(mat: pd.DataFrame) -> float:
    c = mat.corr().to_numpy()
    vals = c[np.triu_indices_from(c, k=1)]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if len(vals) else float("nan")


def circular_shift_cols(mat: np.ndarray, rng: np.random.Generator, min_shift: int = 5) -> np.ndarray:
    """
    Circularly shift each column by a random amount.
    min_shift avoids tiny shifts that keep alignment almost intact.
    """
    n, k = mat.shape
    out = np.empty_like(mat)
    for j in range(k):
        if n <= 2 * min_shift + 1:
            shift = rng.integers(1, max(2, n))  # fallback
        else:
            # sample shift from [min_shift, n-min_shift)
            shift = int(rng.integers(min_shift, n - min_shift))
        out[:, j] = np.roll(mat[:, j], shift)
    return out


def load_aligned_matrix(csv_path: Path) -> pd.DataFrame:
    mat = pd.read_csv(csv_path, index_col=0)
    # Ensure numeric
    mat = mat.apply(pd.to_numeric, errors="coerce")
    # Drop any remaining NaNs (should be none if built with strict overlap)
    mat = mat.dropna(axis=0, how="any")
    return mat


def surrogate_test(mat: pd.DataFrame, n_perm: int, seed: int, min_shift: int) -> dict:
    obs = pairwise_mean_r(mat)

    arr = mat.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    sur = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        shifted = circular_shift_cols(arr, rng, min_shift=min_shift)
        sur[i] = pairwise_mean_r(pd.DataFrame(shifted, columns=mat.columns))

    mu = float(np.mean(sur))
    sd = float(np.std(sur, ddof=1)) if n_perm > 1 else float("nan")
    z = float((obs - mu) / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan")
    p_one_sided = float((np.sum(sur >= obs) + 1) / (n_perm + 1))  # add-one smoothing

    return {
        "observed_mean_r": float(obs),
        "surrogate_mean": mu,
        "surrogate_sd": sd,
        "z": z,
        "p_one_sided": p_one_sided,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync_root", default="analysis_outputs/synchrony",
                    help="Root that contains <segment>/aligned_HR_matrix.csv")
    ap.add_argument("--segments", default="baseline_warmup,meditation_all")
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_shift", type=int, default=10,
                    help="Min circular shift in seconds (avoid near-zero shifts)")
    ap.add_argument("--out_csv", default="analysis_outputs/synchrony/surrogate_results.csv")
    args = ap.parse_args()

    sync_root = Path(args.sync_root)
    segments = [s.strip() for s in args.segments.split(",") if s.strip()]

    rows = []
    for seg in segments:
        path = sync_root / seg / "aligned_HR_matrix.csv"
        if not path.exists():
            print(f"[SKIP] missing: {path}")
            continue

        mat = load_aligned_matrix(path)
        if mat.shape[1] < 2 or len(mat) < 30:
            print(f"[SKIP] not enough data: {seg} shape={mat.shape}")
            continue

        res = surrogate_test(mat, n_perm=args.n_perm, seed=args.seed, min_shift=args.min_shift)
        res.update({
            "segment": seg,
            "n_subjects": int(mat.shape[1]),
            "n_seconds": int(len(mat)),
        })
        rows.append(res)

        print(f"\nSEG: {seg}")
        print("  observed_mean_r     :", res["observed_mean_r"])
        print("  surrogate_mean ± sd :", res["surrogate_mean"], "±", res["surrogate_sd"])
        print("  z                  :", res["z"])
        print("  p (one-sided)      :", res["p_one_sided"])

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\nSaved: {args.out_csv}")
    if len(out):
        print("\nSummary table:")
        print(out[["segment","n_subjects","n_seconds","observed_mean_r","surrogate_mean","surrogate_sd","z","p_one_sided"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()

"""
python3 analysis/04.1_hr_surrogate_test.py \
  --sync_root analysis_outputs/synchrony \
  --segments baseline_warmup,meditation_all \
  --n_perm 1000 --seed 0 --min_shift 10
"""