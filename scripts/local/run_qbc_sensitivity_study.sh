#!/usr/bin/env bash
set -euo pipefail

# End-to-end local QBC sensitivity workflow:
# 1) Build one shared external test dataset
# 2) Run 8 config smoke sweep on seed s01
# 3) Rank configs by rmse + speed penalty
# 4) Run top-3 configs across 5 seeds (stability pass)
#
# Usage:
#   bash scripts/local/run_qbc_sensitivity_study.sh
# Optional env overrides:
#   PYTHON_BIN=python
#   EXPERIMENT_ID=thesis_sm4_qbc_sens_local
#   MODEL_FLAG=SM4
#   BUDGET=b4096
#   BASELINE_EPOCHS=20
#   TOPK=3

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_ID="${EXPERIMENT_ID:-thesis_sm4_qbc_sens_local}"
MODEL_FLAG="${MODEL_FLAG:-SM4}"
PHASE="${PHASE:-qbc_sensitivity}"
BUDGET="${BUDGET:-b4096}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-20}"
TOPK="${TOPK:-3}"

# Stage geometry used in your current b4096 setting.
QBC_N0="${QBC_N0:-512}"
QBC_P="${QBC_P:-1024}"
QBC_K="${QBC_K:-256}"
QBC_T="${QBC_T:-14}"
QBC_N_TEST="${QBC_N_TEST:-256}"

TIME_H="${TIME_H:-0.5}"
NUM_POINTS="${NUM_POINTS:-200}"
SHARED_TEST_MAX="${SHARED_TEST_MAX:-2048}"

ROOT="outputs/qbc_sens"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_BASE="${ROOT}/${EXPERIMENT_ID}_${STAMP}"
SHARED_RUN_ROOT="${RUN_BASE}/shared_test_source"
SMOKE_ROOT="${RUN_BASE}/smoke"
STAB_ROOT="${RUN_BASE}/stability"
RANKING_CSV="${RUN_BASE}/smoke_ranking.csv"
SELECTED_TXT="${RUN_BASE}/selected_top_configs.txt"

mkdir -p "${RUN_BASE}" "${SMOKE_ROOT}" "${STAB_ROOT}"

echo "[1/4] Building shared test source dataset..."
"${PYTHON_BIN}" run_experiment.py \
  --method lhs_static \
  --budget "${BUDGET}" \
  --seed s01 \
  --phase "${PHASE}" \
  --experiment-id "${EXPERIMENT_ID}_shared_test" \
  --model-flag "${MODEL_FLAG}" \
  --run-root "${SHARED_RUN_ROOT}" \
  --skip-preprocess \
  --skip-baseline \
  --stage1-override "time=${TIME_H}" \
  --stage1-override "num_of_points=${NUM_POINTS}"

SHARED_TEST_ROOT="$(${PYTHON_BIN} - <<PY
import json
p = "${SHARED_RUN_ROOT}/run_manifest.json"
with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)
print(d["artifacts"]["dataset_root"])
PY
)"

echo "Shared test dataset root: ${SHARED_TEST_ROOT}"

echo "[2/4] Running smoke sensitivity sweep (8 configs, seed=s01)..."

# Config definitions (name + disagreement/diversity settings)
CFG_NAMES=(
  cfg01_base
  cfg02_varmax
  cfg03_varp90
  cfg04_l2
  cfg05_no_norm
  cfg06_more_diverse
  cfg07_less_diverse
  cfg08_committee5
)
CFG_OVERRIDES=(
  "qbc_M=3 active.disagreement.metric=variance_mean active.diversity.preselect_factor=5 active.diversity.uncertainty_weight=0.7 active.diversity.distance_weight=0.3 active.diversity.normalize_uncertainty=true active.diversity.normalize_distance=true"
  "qbc_M=3 active.disagreement.metric=variance_max  active.diversity.preselect_factor=5 active.diversity.uncertainty_weight=0.7 active.diversity.distance_weight=0.3 active.diversity.normalize_uncertainty=true active.diversity.normalize_distance=true"
  "qbc_M=3 active.disagreement.metric=variance_p90  active.diversity.preselect_factor=5 active.diversity.uncertainty_weight=0.7 active.diversity.distance_weight=0.3 active.diversity.normalize_uncertainty=true active.diversity.normalize_distance=true"
  "qbc_M=3 active.disagreement.metric=member_l2_mean active.diversity.preselect_factor=5 active.diversity.uncertainty_weight=0.7 active.diversity.distance_weight=0.3 active.diversity.normalize_uncertainty=true active.diversity.normalize_distance=true"
  "qbc_M=3 active.disagreement.metric=variance_mean active.diversity.preselect_factor=5 active.diversity.uncertainty_weight=0.7 active.diversity.distance_weight=0.3 active.diversity.normalize_uncertainty=false active.diversity.normalize_distance=false"
  "qbc_M=3 active.disagreement.metric=variance_mean active.diversity.preselect_factor=8 active.diversity.uncertainty_weight=0.55 active.diversity.distance_weight=0.45 active.diversity.normalize_uncertainty=true active.diversity.normalize_distance=true"
  "qbc_M=3 active.disagreement.metric=variance_mean active.diversity.preselect_factor=3 active.diversity.uncertainty_weight=0.85 active.diversity.distance_weight=0.15 active.diversity.normalize_uncertainty=true active.diversity.normalize_distance=true"
  "qbc_M=5 active.disagreement.metric=variance_mean active.diversity.preselect_factor=5 active.diversity.uncertainty_weight=0.7 active.diversity.distance_weight=0.3 active.diversity.normalize_uncertainty=true active.diversity.normalize_distance=true"
)

for i in "${!CFG_NAMES[@]}"; do
  cfg="${CFG_NAMES[$i]}"
  cfg_overrides="${CFG_OVERRIDES[$i]}"
  run_root="${SMOKE_ROOT}/${cfg}_s01"
  echo "[SMOKE] ${cfg}"

  cmd=(
    "${PYTHON_BIN}" run_experiment.py
    --method qbc_deep_ensemble
    --budget "${BUDGET}"
    --seed s01
    --phase "${PHASE}"
    --experiment-id "${EXPERIMENT_ID}_smoke"
    --model-flag "${MODEL_FLAG}"
    --run-root "${run_root}"
    --baseline-epochs "${BASELINE_EPOCHS}"
    --stage1-override "qbc_n0=${QBC_N0}"
    --stage1-override "qbc_n_test=${QBC_N_TEST}"
    --stage1-override "qbc_P=${QBC_P}"
    --stage1-override "qbc_K=${QBC_K}"
    --stage1-override "qbc_T=${QBC_T}"
    --stage1-override "surrogate.deterministic=true"
    --stage1-override "surrogate.device=cpu"
    --stage1-override "time=${TIME_H}"
    --stage1-override "num_of_points=${NUM_POINTS}"
    --stage2-override "dataset.test_split_mode=shared_dataset"
    --stage2-override "dataset.shared_test_dataset_root=${SHARED_TEST_ROOT}"
    --stage2-override "dataset.shared_test_max_trajectories=${SHARED_TEST_MAX}"
    --stage2-override "dataset.validation_flag=false"
    --stage2-override "dataset.new_coll_points_flag=false"
    --stage2-override "time=${TIME_H}"
    --stage2-override "num_of_points=${NUM_POINTS}"
    --stage3-override "baseline.batch_size=128"
  )

  for ov in ${cfg_overrides}; do
    cmd+=(--stage1-override "${ov}")
  done

  "${cmd[@]}"
done

echo "[3/4] Ranking smoke configs..."

"${PYTHON_BIN}" - <<PY
import csv
import json
import math
import os

smoke_root = "${SMOKE_ROOT}"
out_csv = "${RANKING_CSV}"
selected_txt = "${SELECTED_TXT}"
topk = int("${TOPK}")

rows = []
for d in sorted(os.listdir(smoke_root)):
    run_root = os.path.join(smoke_root, d)
    manifest = os.path.join(run_root, "run_manifest.json")
    telemetry = os.path.join(run_root, "telemetry", "round_telemetry.csv")
    if not os.path.exists(manifest):
        continue
    with open(manifest, "r", encoding="utf-8") as f:
        m = json.load(f)
    rmse = float(m["artifacts"]["baseline_metrics_payload"]["rmse"])

    total_round_seconds = 0.0
    if os.path.exists(telemetry):
        with open(telemetry, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                v = r.get("round_seconds")
                if v not in (None, ""):
                    total_round_seconds += float(v)

    # quality + compute tradeoff (same objective as discussed)
    objective = rmse + 0.03 * math.log1p(total_round_seconds)
    cfg_name = d.replace("_s01", "")
    rows.append({
        "cfg": cfg_name,
        "run_root": run_root,
        "rmse": rmse,
        "total_round_seconds": total_round_seconds,
        "objective": objective,
    })

rows.sort(key=lambda x: x["objective"])

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["rank", "cfg", "rmse", "total_round_seconds", "objective", "run_root"])
    w.writeheader()
    for i, r in enumerate(rows, start=1):
        row = dict(r)
        row["rank"] = i
        w.writerow(row)

picked = rows[:max(1, min(topk, len(rows)))]
with open(selected_txt, "w", encoding="utf-8") as f:
    for r in picked:
        f.write(r["cfg"] + "\n")

print("Top configs:")
for r in picked:
    print(f"  {r['cfg']} objective={r['objective']:.6f} rmse={r['rmse']:.6f} sec={r['total_round_seconds']:.2f}")
print(f"Ranking CSV: {out_csv}")
print(f"Selected configs: {selected_txt}")
PY

echo "[4/4] Running 5-seed stability pass for top configs..."
SEEDS=(s01 s02 s03 s04 s05)

TOP_CFGS=()
while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  TOP_CFGS+=("${line}")
done < "${SELECTED_TXT}"

cfg_overrides_for() {
  local target="$1"
  local i
  for i in "${!CFG_NAMES[@]}"; do
    if [[ "${CFG_NAMES[$i]}" == "${target}" ]]; then
      echo "${CFG_OVERRIDES[$i]}"
      return 0
    fi
  done
  return 1
}

for cfg in "${TOP_CFGS[@]}"; do
  if ! cfg_overrides="$(cfg_overrides_for "${cfg}")"; then
    echo "[WARN] Missing override mapping for ${cfg}, skipping"
    continue
  fi

  for seed in "${SEEDS[@]}"; do
    run_root="${STAB_ROOT}/${cfg}_${seed}"
    echo "[STABILITY] cfg=${cfg} seed=${seed}"

    cmd=(
      "${PYTHON_BIN}" run_experiment.py
      --method qbc_deep_ensemble
      --budget "${BUDGET}"
      --seed "${seed}"
      --phase "${PHASE}"
      --experiment-id "${EXPERIMENT_ID}_stability"
      --model-flag "${MODEL_FLAG}"
      --run-root "${run_root}"
      --baseline-epochs "${BASELINE_EPOCHS}"
      --stage1-override "qbc_n0=${QBC_N0}"
      --stage1-override "qbc_n_test=${QBC_N_TEST}"
      --stage1-override "qbc_P=${QBC_P}"
      --stage1-override "qbc_K=${QBC_K}"
      --stage1-override "qbc_T=${QBC_T}"
      --stage1-override "surrogate.deterministic=true"
      --stage1-override "surrogate.device=cpu"
      --stage1-override "time=${TIME_H}"
      --stage1-override "num_of_points=${NUM_POINTS}"
      --stage2-override "dataset.test_split_mode=shared_dataset"
      --stage2-override "dataset.shared_test_dataset_root=${SHARED_TEST_ROOT}"
      --stage2-override "dataset.shared_test_max_trajectories=${SHARED_TEST_MAX}"
      --stage2-override "dataset.validation_flag=false"
      --stage2-override "dataset.new_coll_points_flag=false"
      --stage2-override "time=${TIME_H}"
      --stage2-override "num_of_points=${NUM_POINTS}"
      --stage3-override "baseline.batch_size=128"
    )

    cfg_override_arr=()
    # Parse space-delimited key=value tokens safely even if IFS was modified.
    IFS=' ' read -r -a cfg_override_arr <<< "${cfg_overrides}"
    for ov in "${cfg_override_arr[@]}"; do
      cmd+=(--stage1-override "${ov}")
    done

    "${cmd[@]}"
  done
done

echo "Completed."
echo "Run base: ${RUN_BASE}"
echo "Smoke ranking: ${RANKING_CSV}"
echo "Top configs: ${SELECTED_TXT}"
