"""Small helpers shared by thesis result notebooks.

The functions in this module intentionally stay close to notebook mechanics:
loading result files, computing compact summary statistics, formatting tables,
and saving figures. Experiment-specific interpretation belongs in the notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def ensure_repo_root_on_path(current_path: str | Path | None = None) -> Path:
    """Add the repository root to `sys.path` and return it.

    This supports notebooks executed either from the repository root or from the
    `results/` directory.
    """
    import sys

    cwd = Path.cwd() if current_path is None else Path(current_path).resolve()
    candidates = [cwd, *cwd.parents]
    for candidate in candidates:
        if (candidate / "src").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise FileNotFoundError(f"Could not find repository root from {cwd}")


def load_json(path: str | Path, default=None):
    """Load a JSON file, returning `default` when the file is missing or invalid."""
    path = Path(path)
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def sem(values: Iterable[float]) -> float:
    """Return the standard error of the mean, ignoring missing values."""
    series = pd.Series(values).dropna().astype(float)
    if len(series) <= 1:
        return np.nan
    return float(series.std(ddof=1) / np.sqrt(len(series)))


def mean_sem_text(
    row: Mapping[str, object] | pd.Series,
    metric: str,
    *,
    decimals: int = 2,
    scientific: bool = True,
    missing: str = "n/a",
    separator: str = " +/- ",
) -> str:
    """Format `<metric>_mean` and `<metric>_sem` from a summary row."""
    get = row.get if hasattr(row, "get") else row.__getitem__
    mean = get(f"{metric}_mean", np.nan)
    err = get(f"{metric}_sem", np.nan)
    if pd.isna(mean):
        return missing
    fmt = f"{{:.{decimals}e}}" if scientific else f"{{:.{decimals}f}}"
    if pd.isna(err):
        return fmt.format(float(mean))
    return f"{fmt.format(float(mean))}{separator}{fmt.format(float(err))}"


def seed_list(values: Iterable[object]) -> str:
    """Return sorted, comma-separated non-missing seed labels."""
    labels = sorted(map(str, pd.Series(values).dropna().unique()))
    return ",".join(labels)


def aggregate_mean_sem(
    frame: pd.DataFrame,
    *,
    group_cols: str | Sequence[str],
    metrics: Sequence[str],
    seed_col: str | None = "seed_label",
    run_col: str | None = "run_name",
    order_col: str | None = None,
    order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate metrics by group with mean, SEM, optional seeds, and run count."""
    if frame.empty:
        return pd.DataFrame()

    group_cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    aggregations = {}
    for metric in metrics:
        if metric in frame.columns:
            aggregations[f"{metric}_mean"] = (metric, "mean")
            aggregations[f"{metric}_sem"] = (metric, sem)

    if seed_col and seed_col in frame.columns:
        aggregations["seeds"] = (seed_col, seed_list)
    if run_col and run_col in frame.columns:
        aggregations["n_runs"] = (run_col, "count")

    out = frame.groupby(group_cols, observed=True).agg(**aggregations).reset_index()
    if order_col and order:
        out[order_col] = pd.Categorical(out[order_col], categories=list(order), ordered=True)
        out = out.sort_values(order_col)
    return out.reset_index(drop=True)


def run_dir(root: str | Path, batch: object, run_name: object) -> Path:
    """Return the standard run artifact directory for batch-style experiments."""
    return Path(root) / str(batch) / "runs" / str(run_name)


def artifact_path(root: str | Path, batch: object, run_name: object, filename: str) -> Path:
    """Return a file path inside a standard run artifact directory."""
    return run_dir(root, batch, run_name) / filename


def epoch_metrics_path(root: str | Path, batch: object, run_name: object) -> Path:
    """Return the standard per-epoch metrics CSV path."""
    return artifact_path(root, batch, run_name, "epoch_metrics.csv")


def metrics_json_path(root: str | Path, batch: object, run_name: object) -> Path:
    """Return the standard metrics JSON path."""
    return artifact_path(root, batch, run_name, "metrics.json")


def timings_json_path(root: str | Path, batch: object, run_name: object) -> Path:
    """Return the standard timings JSON path."""
    return artifact_path(root, batch, run_name, "timings.json")


def read_csv_if_exists(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a CSV file if it exists, otherwise return an empty DataFrame."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def load_posthoc_external_eval(
    root: str | Path,
    *,
    setting: str = "best_train_exclusive_ood",
    completed_only: bool = True,
) -> pd.DataFrame:
    """Load posthoc checkpoint external-evaluation metrics from per-run JSONs.

    The returned columns intentionally mirror notebook metric names such as
    ``best_id_eval_mse`` and ``best_ood_eval_mse`` so callers can overlay the
    new posthoc values onto existing run tables before aggregation.
    """

    rows: list[dict[str, object]] = []
    for path in sorted(Path(root).glob(f"**/posthoc_evaluation/{setting}/runs/*.json")):
        payload = load_json(path, default={}) or {}
        if completed_only and payload.get("status") != "completed":
            continue

        id_section = payload.get("id") if isinstance(payload.get("id"), dict) else {}
        ood_section = payload.get("ood") if isinstance(payload.get("ood"), dict) else {}
        id_metrics = id_section.get("metrics") if isinstance(id_section.get("metrics"), dict) else {}
        ood_metrics = ood_section.get("metrics") if isinstance(ood_section.get("metrics"), dict) else {}

        row = {
            "posthoc_result_path": str(path),
            "posthoc_setting": setting,
            "posthoc_status": payload.get("status"),
            "posthoc_experiment_name": payload.get("experiment_name"),
            "batch": payload.get("batch_id"),
            "run_name": payload.get("run_name"),
            "posthoc_eval_split": payload.get("eval_split"),
            "posthoc_ood_mode": payload.get("ood_mode"),
            "posthoc_id_eval_root": payload.get("id_eval_root"),
            "posthoc_ood_eval_root": payload.get("ood_eval_root"),
            "posthoc_model_flag": payload.get("model_flag"),
            "posthoc_init_condition_bounds": payload.get("init_condition_bounds"),
            "posthoc_model_info_source": payload.get("model_info_source"),
            "best_id_eval_mse": id_metrics.get("mse"),
            "best_id_eval_rmse": id_metrics.get("rmse"),
            "best_id_eval_mae": id_metrics.get("mae"),
            "best_id_eval_max_abs_error": id_metrics.get("max_abs_error"),
            "best_id_eval_trajectory_mse_mean": id_metrics.get("trajectory_mse_mean"),
            "best_id_eval_trajectory_mse_p90": id_metrics.get("trajectory_mse_p90"),
            "best_id_eval_trajectory_mse_p95": id_metrics.get("trajectory_mse_p95"),
            "best_id_eval_trajectory_mse_max": id_metrics.get("trajectory_mse_max"),
            "best_id_eval_trajectory_rmse_mean": id_metrics.get("trajectory_rmse_mean"),
            "best_id_eval_trajectory_rmse_p90": id_metrics.get("trajectory_rmse_p90"),
            "best_id_eval_trajectory_rmse_p95": id_metrics.get("trajectory_rmse_p95"),
            "best_id_eval_trajectory_rmse_max": id_metrics.get("trajectory_rmse_max"),
            "best_id_eval_trajectory_max_abs_error_mean": id_metrics.get("trajectory_max_abs_error_mean"),
            "best_id_eval_trajectory_max_abs_error_p95": id_metrics.get("trajectory_max_abs_error_p95"),
            "best_id_eval_trajectory_max_abs_error_max": id_metrics.get("trajectory_max_abs_error_max"),
            "best_id_eval_n_rows": id_section.get("n_rows"),
            "best_id_eval_n_trajectories": id_section.get("n_trajectories"),
            "best_ood_eval_mse": ood_metrics.get("mse"),
            "best_ood_eval_rmse": ood_metrics.get("rmse"),
            "best_ood_eval_mae": ood_metrics.get("mae"),
            "best_ood_eval_max_abs_error": ood_metrics.get("max_abs_error"),
            "best_ood_eval_trajectory_mse_mean": ood_metrics.get("trajectory_mse_mean"),
            "best_ood_eval_trajectory_mse_p90": ood_metrics.get("trajectory_mse_p90"),
            "best_ood_eval_trajectory_mse_p95": ood_metrics.get("trajectory_mse_p95"),
            "best_ood_eval_trajectory_mse_max": ood_metrics.get("trajectory_mse_max"),
            "best_ood_eval_trajectory_rmse_mean": ood_metrics.get("trajectory_rmse_mean"),
            "best_ood_eval_trajectory_rmse_p90": ood_metrics.get("trajectory_rmse_p90"),
            "best_ood_eval_trajectory_rmse_p95": ood_metrics.get("trajectory_rmse_p95"),
            "best_ood_eval_trajectory_rmse_max": ood_metrics.get("trajectory_rmse_max"),
            "best_ood_eval_trajectory_max_abs_error_mean": ood_metrics.get("trajectory_max_abs_error_mean"),
            "best_ood_eval_trajectory_max_abs_error_p95": ood_metrics.get("trajectory_max_abs_error_p95"),
            "best_ood_eval_trajectory_max_abs_error_max": ood_metrics.get("trajectory_max_abs_error_max"),
            "best_ood_eval_n_rows_before_filter": ood_section.get("n_rows_before_filter"),
            "best_ood_eval_n_trajectories_before_filter": ood_section.get("n_trajectories_before_filter"),
            "best_ood_eval_n_rows": ood_section.get("n_rows"),
            "best_ood_eval_n_trajectories": ood_section.get("n_trajectories"),
        }
        row["best_id_ood_eval_mse_gap"] = (
            None if row["best_id_eval_mse"] is None or row["best_ood_eval_mse"] is None
            else float(row["best_ood_eval_mse"]) - float(row["best_id_eval_mse"])
        )
        row["best_id_ood_eval_rmse_gap"] = (
            None if row["best_id_eval_rmse"] is None or row["best_ood_eval_rmse"] is None
            else float(row["best_ood_eval_rmse"]) - float(row["best_id_eval_rmse"])
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for col in frame.columns:
        if col not in {
            "posthoc_result_path",
            "posthoc_setting",
            "posthoc_status",
            "posthoc_experiment_name",
            "batch",
            "run_name",
            "posthoc_eval_split",
            "posthoc_ood_mode",
            "posthoc_id_eval_root",
            "posthoc_ood_eval_root",
            "posthoc_model_flag",
            "posthoc_model_info_source",
        }:
            converted = pd.to_numeric(frame[col], errors="coerce")
            non_null = frame[col].notna()
            if converted[non_null].notna().all():
                frame[col] = converted
    return frame.reset_index(drop=True)


def overlay_posthoc_external_eval(
    frame: pd.DataFrame,
    posthoc: pd.DataFrame,
    *,
    on: Sequence[str] = ("batch", "run_name"),
) -> pd.DataFrame:
    """Overlay posthoc external-evaluation columns onto an existing run table."""

    if frame.empty or posthoc.empty:
        return frame.copy()
    key_cols = list(on)
    value_cols = [col for col in posthoc.columns if col not in key_cols]
    merged = frame.merge(
        posthoc[key_cols + value_cols],
        on=key_cols,
        how="left",
        suffixes=("", "_posthoc"),
    )
    for col in value_cols:
        posthoc_col = f"{col}_posthoc" if col in frame.columns else col
        if posthoc_col not in merged.columns:
            continue
        if col in frame.columns:
            merged[col] = merged[posthoc_col].combine_first(merged[col])
            merged = merged.drop(columns=[posthoc_col])
    return merged


def read_epoch_metrics(
    path: str | Path,
    *,
    columns: Iterable[str] | None = None,
    numeric_except: Iterable[str] = ("phase_name",),
) -> pd.DataFrame:
    """Read an epoch metrics CSV and coerce numeric columns."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    column_set = set(columns) if columns is not None else None
    df = pd.read_csv(
        path,
        usecols=(lambda col: col in column_set) if column_set is not None else None,
        low_memory=False,
    )
    non_numeric = set(numeric_except)
    for col in df.columns:
        if col not in non_numeric:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def metric_from_entry(entry: Mapping[str, object] | None, metric: str) -> float:
    """Extract one metric from a nested `{'metrics': ...}` entry."""
    value = ((entry or {}).get("metrics") or {}).get(metric)
    return np.nan if value is None else float(value)


def eval_metric(metrics: Mapping[str, object] | None, section: str, eval_name: str, metric: str) -> float:
    """Extract an external evaluation metric from final or best-checkpoint metrics."""
    metrics = metrics or {}
    if section == "best":
        root = ((metrics.get("best_checkpoint_metrics") or {}).get("evaluation_sets") or {})
    elif section == "final":
        root = metrics.get("evaluation_sets") or {}
    else:
        raise ValueError("section must be 'best' or 'final'")
    return metric_from_entry(root.get(eval_name), metric)


def split_metric(metrics: Mapping[str, object] | None, section: str, split: str, metric: str) -> float:
    """Extract train/val/test split metrics from final or best-checkpoint metrics."""
    metrics = metrics or {}
    if section == "best":
        root = metrics.get("best_checkpoint_metrics") or {}
        value = (((root.get(split) or {}).get("metrics") or {}).get(metric))
    elif section == "final":
        value = ((metrics.get(f"final_{split}_metrics") or {}).get(metric))
    else:
        raise ValueError("section must be 'best' or 'final'")
    return np.nan if value is None else float(value)


def save_and_show(fig, export_dir: str | Path, filename: str, *, dpi: int = 300) -> Path:
    """Save a Matplotlib figure under `export_dir`, show it, and return the path.

    Does not call tight_layout — the thesis rcParams (savefig.bbox=tight,
    savefig.pad_inches=0.04) handle output padding, and tight_layout can
    interfere with custom tick locators on log-scale axes.  Callers that need
    explicit layout adjustment should call fig.tight_layout() themselves before
    passing the figure in.
    """
    import matplotlib.pyplot as plt

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / filename
    fig.savefig(path, dpi=dpi)
    plt.show()
    return path
