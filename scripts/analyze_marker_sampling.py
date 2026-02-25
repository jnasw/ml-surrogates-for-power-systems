"""Analyze marker-directed sampling runs and dataset geometry.

Examples:
  python scripts/analyze_marker_sampling.py \
    --run-dir outputs/.../marker_directed/.../qbc \
    --dataset baseline_lhs=outputs/.../lhs_static/.../data/SM4/dataset_v1 \
    --dataset marker=outputs/.../marker_directed/.../data/SM4/dataset_v1
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.active.marker_features import compute_marker_matrix
from src.data.raw_trajectory_loader import load_trajectory_dataset_from_raw
from src.surrogate.baseline import BaselineConfig, evaluate_baseline, train_baseline_surrogate


def _parse_dataset_item(text: str) -> tuple[str, Path]:
    if "=" not in text:
        p = Path(text)
        return p.name, p
    label, raw_path = text.split("=", 1)
    return label.strip(), Path(raw_path.strip())


def _iter_raw_arrays(raw_dir: Path) -> Iterable[np.ndarray]:
    for p in sorted(raw_dir.glob("*.pkl")):
        with p.open("rb") as handle:
            payload = pickle.load(handle)
        for traj in payload:
            yield np.asarray(traj, dtype=np.float32)


def load_dataset_root(dataset_root: Path, init_conditions_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    raw_dir = dataset_root / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    arrays = list(_iter_raw_arrays(raw_dir))
    if not arrays:
        raise ValueError(f"No raw trajectories found in {raw_dir}")

    model_flag = dataset_root.parent.name
    guide_path = init_conditions_dir / "modellings_guide.yaml"
    guide = OmegaConf.load(str(guide_path))
    state_names: list[str] | None = None
    for item in guide:
        if item.get("name") == model_flag:
            state_names = list(item.get("keys"))
            break
    if state_names is None:
        raise ValueError(f"Model '{model_flag}' not found in {guide_path}")

    t = arrays[0][0]
    y = np.stack([arr[1 : 1 + len(state_names), :].T for arr in arrays], axis=0).astype(np.float32)
    return y, t.astype(np.float32), state_names


def _standardize(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std


def _pca_n90(x_std: np.ndarray) -> tuple[np.ndarray, int]:
    xc = x_std - x_std.mean(axis=0, keepdims=True)
    _, svals, vt = np.linalg.svd(xc, full_matrices=False)
    var = (svals**2) / max(1, xc.shape[0] - 1)
    if float(var.sum()) <= 0:
        return xc[:, :1], 1
    cum = np.cumsum(var / var.sum())
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    z = xc @ vt[:n90].T
    return z, n90


def summarize_dataset(dataset_root: Path, init_conditions_dir: Path) -> dict[str, float]:
    trajs, t, state_names = load_dataset_root(dataset_root, init_conditions_dir)
    markers, marker_names = compute_marker_matrix(trajs=trajs, time_grid=t, state_names=state_names)
    x_std = _standardize(markers)
    z, n90 = _pca_n90(x_std)

    if z.shape[0] <= 1:
        mean_nn = 0.0
    else:
        d = cdist(z, z)
        np.fill_diagonal(d, np.inf)
        mean_nn = float(np.min(d, axis=1).mean())

    return {
        "n_traj": float(trajs.shape[0]),
        "n_states": float(trajs.shape[2]),
        "n_markers": float(len(marker_names)),
        "pca_n90": float(n90),
        "mean_nn_dist_pca": mean_nn,
    }


def _dataset_root_to_loader_args(dataset_root: Path) -> tuple[str, str, int]:
    # Expect .../<dataset_dir>/<model_flag>/dataset_vN
    if not dataset_root.name.startswith("dataset_v"):
        raise ValueError(f"Expected dataset_vN folder, got: {dataset_root}")
    model_flag = dataset_root.parent.name
    dataset_dir = str(dataset_root.parent.parent)
    number_str = dataset_root.name.replace("dataset_v", "")
    if not number_str.isdigit():
        raise ValueError(f"Could not parse dataset number from: {dataset_root.name}")
    return dataset_dir, model_flag, int(number_str)


def evaluate_dataset_baseline(
    dataset_root: Path,
    init_conditions_dir: Path,
    split_ratio: float,
    seed: int,
    hidden_dim: int,
    hidden_layers: int,
    dropout: float,
    lr: float,
    batch_size: int,
    epochs: int,
    device: str,
) -> dict[str, float]:
    dataset_dir, model_flag, dataset_number = _dataset_root_to_loader_args(dataset_root)
    ds = load_trajectory_dataset_from_raw(
        dataset_dir=dataset_dir,
        init_conditions_dir=str(init_conditions_dir),
        model_flag=model_flag,
        dataset_number=dataset_number,
        split_ratio=split_ratio,
        shuffle=False,
        seed=seed,
    )

    cfg = BaselineConfig(
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
    )
    model = train_baseline_surrogate(dataset=ds, seed=seed, cfg=cfg)
    metrics = evaluate_baseline(model=model, dataset=ds)
    return {
        "n_train": float(ds.n_train),
        "n_test": float(ds.n_test),
        "mse": float(metrics.get("mse", np.nan)),
        "rmse": float(metrics.get("rmse", np.nan)),
    }


def summarize_run_dir(run_dir: Path) -> None:
    history = run_dir / "history.jsonl"
    rounds_dir = run_dir / "rounds"
    if not history.exists():
        raise FileNotFoundError(f"History file not found: {history}")

    rows: list[dict] = []
    with history.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        print("No round entries in history.jsonl.")
        return

    print("\nRun round summaries:")
    for r in rows:
        rid = int(r["round_idx"])
        msg = (
            f"round={rid:03d} train={int(r.get('train_size', -1))} "
            f"mean_score={float(r.get('mean_score', r.get('mean_disagreement', np.nan))):.4f} "
            f"max_score={float(r.get('max_score', r.get('max_disagreement', np.nan))):.4f}"
        )
        if "marker_pca_components" in r:
            msg += f" pca_comp={int(r['marker_pca_components'])}"
        if "mean_selected_to_train_distance" in r:
            msg += f" mean_sel_to_train_d={float(r['mean_selected_to_train_distance']):.4f}"
        print(msg)

        round_dir = rounds_dir / f"round_{rid:03d}"
        d_path = round_dir / "marker_diversity.npy"
        s_path = round_dir / "marker_sparsity.npy"
        sel_path = round_dir / "selected_indices.npy"
        if d_path.exists() and s_path.exists() and sel_path.exists():
            diversity = np.load(d_path)
            sparsity = np.load(s_path)
            selected = np.load(sel_path).astype(int)
            print(
                "  selected mean diversity="
                f"{float(np.mean(diversity[selected])):.4f} "
                f"| selected mean sparsity={float(np.mean(sparsity[selected])):.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze marker-directed runs and datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset root in form label=/path/to/data/<MODEL>/dataset_vN (can be repeated).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Optional experiment logger run directory containing history.jsonl and rounds/.",
    )
    parser.add_argument(
        "--init-conditions-dir",
        type=str,
        default="src/conf/ic",
        help="Path to init condition config root (contains modellings_guide.yaml).",
    )
    parser.add_argument(
        "--eval-baseline",
        action="store_true",
        help="Also train/evaluate the fixed baseline surrogate on each dataset (raw split).",
    )
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/test split ratio for baseline eval.")
    parser.add_argument("--seed", type=int, default=37, help="Seed for baseline training/eval split loader.")
    parser.add_argument("--baseline-hidden-dim", type=int, default=256)
    parser.add_argument("--baseline-hidden-layers", type=int, default=4)
    parser.add_argument("--baseline-dropout", type=float, default=0.05)
    parser.add_argument("--baseline-lr", type=float, default=0.001)
    parser.add_argument("--baseline-batch-size", type=int, default=64)
    parser.add_argument("--baseline-epochs", type=int, default=300)
    parser.add_argument("--baseline-device", type=str, default="auto")
    args = parser.parse_args()

    init_dir = Path(args.init_conditions_dir).resolve()
    if args.run_dir:
        summarize_run_dir(Path(args.run_dir).resolve())

    if args.dataset:
        print("\nDataset geometry summaries:")
        parsed = []
        for item in args.dataset:
            label, root = _parse_dataset_item(item)
            parsed.append((label, root.resolve()))
            summary = summarize_dataset(root.resolve(), init_dir)
            print(
                f"{label}: n={int(summary['n_traj'])}, states={int(summary['n_states'])}, "
                f"markers={int(summary['n_markers'])}, pca_n90={int(summary['pca_n90'])}, "
                f"mean_nn_dist_pca={summary['mean_nn_dist_pca']:.4f}"
            )

        if args.eval_baseline:
            print("\nBaseline surrogate performance summaries:")
            for label, root in parsed:
                perf = evaluate_dataset_baseline(
                    dataset_root=root,
                    init_conditions_dir=init_dir,
                    split_ratio=float(args.split_ratio),
                    seed=int(args.seed),
                    hidden_dim=int(args.baseline_hidden_dim),
                    hidden_layers=int(args.baseline_hidden_layers),
                    dropout=float(args.baseline_dropout),
                    lr=float(args.baseline_lr),
                    batch_size=int(args.baseline_batch_size),
                    epochs=int(args.baseline_epochs),
                    device=str(args.baseline_device),
                )
                print(
                    f"{label}: n_train={int(perf['n_train'])}, n_test={int(perf['n_test'])}, "
                    f"mse={perf['mse']:.6e}, rmse={perf['rmse']:.6e}"
                )


if __name__ == "__main__":
    main()
