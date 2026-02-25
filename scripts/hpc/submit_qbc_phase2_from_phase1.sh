#!/usr/bin/env bash
set -euo pipefail

# Build and submit a Phase-2 HPC matrix using Phase-1 results.
# Phase 2A (scale transfer): selected policy configs x scale budgets x seeds
# Phase 2B (geometry sensitivity): geometry cfgs x geometry variants x geometry budgets x seeds
#
# Required:
#   PHASE1_RUN_BASE=outputs/qbc_sens/<phase1_run_name>
# Optional examples:
#   SCALE_BUDGETS=b4096,b8192,b16384
#   GEOMETRY_CFGS=cfg04_l2
#   GEOMETRY_VARIANTS=geo_base,geo_fine,geo_coarse,geo_low_n0,geo_high_n0,geo_high_p
#   GEOMETRY_BUDGETS=b4096
#   SEEDS=s01,s02,s03,s04,s05,s06,s07,s08,s09,s10
#   PHASE2_EXPERIMENT_ID=thesis_sm4_qbc_phase2

PHASE1_RUN_BASE="${PHASE1_RUN_BASE:-}"
if [[ -z "${PHASE1_RUN_BASE}" ]]; then
  echo "[ERROR] PHASE1_RUN_BASE is required."
  echo "Example: PHASE1_RUN_BASE=outputs/qbc_sens/thesis_sm4_qbc_sens_local_evalrmse_20260225_093141 bash scripts/hpc/submit_qbc_phase2_from_phase1.sh"
  exit 1
fi
if [[ ! -d "${PHASE1_RUN_BASE}" ]]; then
  echo "[ERROR] PHASE1_RUN_BASE does not exist: ${PHASE1_RUN_BASE}"
  exit 1
fi

SELECTED_CFGS_FILE="${PHASE1_RUN_BASE}/selected_top_configs.txt"
SHARED_MANIFEST="${PHASE1_RUN_BASE}/shared_test_source/run_manifest.json"

if [[ ! -f "${SELECTED_CFGS_FILE}" ]]; then
  echo "[ERROR] Missing selected configs file: ${SELECTED_CFGS_FILE}"
  exit 1
fi
if [[ ! -f "${SHARED_MANIFEST}" ]]; then
  echo "[ERROR] Missing shared-test manifest: ${SHARED_MANIFEST}"
  exit 1
fi

SCALE_BUDGETS="${SCALE_BUDGETS:-b4096,b8192,b16384}"
SEEDS="${SEEDS:-s01,s02,s03,s04,s05,s06,s07,s08,s09,s10}"
# If empty, default to first selected cfg from Phase-1.
GEOMETRY_CFGS="${GEOMETRY_CFGS:-}"
GEOMETRY_VARIANTS="${GEOMETRY_VARIANTS:-geo_base,geo_fine,geo_coarse,geo_low_n0,geo_high_n0,geo_high_p}"
GEOMETRY_BUDGETS="${GEOMETRY_BUDGETS:-b4096}"

PHASE2_PHASE="${PHASE2_PHASE:-qbc_sensitivity}"
PHASE2_EXPERIMENT_ID="${PHASE2_EXPERIMENT_ID:-thesis_sm4_qbc_phase2}"
MODEL_FLAG="${MODEL_FLAG:-SM4}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-20}"

SURROGATE_DEVICE="${SURROGATE_DEVICE:-cuda}"
QBC_N_TEST="${QBC_N_TEST:-256}"
TIME_H="${TIME_H:-0.5}"
NUM_POINTS="${NUM_POINTS:-200}"
SHARED_TEST_MAX="${SHARED_TEST_MAX:-2048}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-128}"
DRY_RUN="${DRY_RUN:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
PHASE2_RUN_BASE="${PHASE2_RUN_BASE:-outputs/qbc_phase2/${PHASE2_EXPERIMENT_ID}_${STAMP}}"
MATRIX_PATH="${PHASE2_RUN_BASE}/phase2_matrix.tsv"
META_PATH="${PHASE2_RUN_BASE}/phase2_meta.json"

mkdir -p "${PHASE2_RUN_BASE}" outputs/slurm_logs

SHARED_TEST_ROOT="$(python - <<PY
import json
p = r'''${SHARED_MANIFEST}'''
with open(p, 'r', encoding='utf-8') as f:
    d = json.load(f)
print(d['artifacts']['dataset_root'])
PY
)"

export SELECTED_CFGS_FILE
export GEOMETRY_CFGS
export SCALE_BUDGETS
export SEEDS
export GEOMETRY_VARIANTS
export GEOMETRY_BUDGETS
export MATRIX_PATH

python - <<'PY'
import csv
import os

policy_map = {
    'cfg01_base': 'qbc_M=3 active.disagreement.metric=variance_mean active.diversity.preselect_factor=5 active.diversity.uncertainty_weight=0.7 active.diversity.distance_weight=0.3 active.diversity.normalize_uncertainty=true active.diversity.normalize_distance=true',
    'cfg02_varmax': 'qbc_M=3 active.disagreement.metric=variance_max active.diversity.preselect_factor=5 active.diversity.uncertainty_weight=0.7 active.diversity.distance_weight=0.3 active.diversity.normalize_uncertainty=true active.diversity.normalize_distance=true',
    'cfg03_varp90': 'qbc_M=3 active.disagreement.metric=variance_p90 active.diversity.preselect_factor=5 active.diversity.uncertainty_weight=0.7 active.diversity.distance_weight=0.3 active.diversity.normalize_uncertainty=true active.diversity.normalize_distance=true',
    'cfg04_l2': 'qbc_M=3 active.disagreement.metric=member_l2_mean active.diversity.preselect_factor=5 active.diversity.uncertainty_weight=0.7 active.diversity.distance_weight=0.3 active.diversity.normalize_uncertainty=true active.diversity.normalize_distance=true',
    'cfg05_no_norm': 'qbc_M=3 active.disagreement.metric=variance_mean active.diversity.preselect_factor=5 active.diversity.uncertainty_weight=0.7 active.diversity.distance_weight=0.3 active.diversity.normalize_uncertainty=false active.diversity.normalize_distance=false',
    'cfg06_more_diverse': 'qbc_M=3 active.disagreement.metric=variance_mean active.diversity.preselect_factor=8 active.diversity.uncertainty_weight=0.55 active.diversity.distance_weight=0.45 active.diversity.normalize_uncertainty=true active.diversity.normalize_distance=true',
    'cfg07_less_diverse': 'qbc_M=3 active.disagreement.metric=variance_mean active.diversity.preselect_factor=3 active.diversity.uncertainty_weight=0.85 active.diversity.distance_weight=0.15 active.diversity.normalize_uncertainty=true active.diversity.normalize_distance=true',
    'cfg08_committee5': 'qbc_M=5 active.disagreement.metric=variance_mean active.diversity.preselect_factor=5 active.diversity.uncertainty_weight=0.7 active.diversity.distance_weight=0.3 active.diversity.normalize_uncertainty=true active.diversity.normalize_distance=true',
}

# Geometry variants keep final size 4096 and focus on schedule/pool effects.
geometry_map = {
    'geo_base': 'qbc_n0=512 qbc_K=256 qbc_T=14 qbc_P=1024',
    'geo_fine': 'qbc_n0=512 qbc_K=128 qbc_T=28 qbc_P=1024',
    'geo_coarse': 'qbc_n0=512 qbc_K=512 qbc_T=7 qbc_P=1024',
    'geo_low_n0': 'qbc_n0=256 qbc_K=256 qbc_T=15 qbc_P=1024',
    'geo_high_n0': 'qbc_n0=1024 qbc_K=256 qbc_T=12 qbc_P=1024',
    'geo_high_p': 'qbc_n0=512 qbc_K=256 qbc_T=14 qbc_P=2048',
}

sel_file = os.environ['SELECTED_CFGS_FILE']
with open(sel_file, 'r', encoding='utf-8') as f:
    selected_cfgs = [ln.strip() for ln in f if ln.strip()]
if not selected_cfgs:
    raise SystemExit('No selected configs found in selected_top_configs.txt')

policy_cfgs = [c for c in selected_cfgs if c in policy_map]
if not policy_cfgs:
    raise SystemExit('None of selected configs are known policy ids')

scale_budgets = [x.strip() for x in os.environ['SCALE_BUDGETS'].split(',') if x.strip()]
seeds = [x.strip() for x in os.environ['SEEDS'].split(',') if x.strip()]
geometry_variants = [x.strip() for x in os.environ['GEOMETRY_VARIANTS'].split(',') if x.strip()]
geometry_budgets = [x.strip() for x in os.environ['GEOMETRY_BUDGETS'].split(',') if x.strip()]

raw_geometry_cfgs = [x.strip() for x in os.environ.get('GEOMETRY_CFGS', '').split(',') if x.strip()]
if not raw_geometry_cfgs:
    geometry_cfgs = [policy_cfgs[0]]
else:
    geometry_cfgs = [c for c in raw_geometry_cfgs if c in policy_map]
    if not geometry_cfgs:
        raise SystemExit('GEOMETRY_CFGS provided, but none match known config ids')

for gv in geometry_variants:
    if gv not in geometry_map:
        raise SystemExit(f'Unknown geometry variant: {gv}')

rows = []
# Phase 2A: scale transfer
for cfg in policy_cfgs:
    for budget in scale_budgets:
        for seed in seeds:
            rows.append({
                'track': 'scale',
                'cfg': cfg,
                'budget': budget,
                'seed': seed,
                'geometry_variant': 'none',
                'policy_overrides': policy_map[cfg],
                'geometry_overrides': '',
            })

# Phase 2B: geometry sensitivity
for cfg in geometry_cfgs:
    for budget in geometry_budgets:
        for gv in geometry_variants:
            for seed in seeds:
                rows.append({
                    'track': 'geometry',
                    'cfg': cfg,
                    'budget': budget,
                    'seed': seed,
                    'geometry_variant': gv,
                    'policy_overrides': policy_map[cfg],
                    'geometry_overrides': geometry_map[gv],
                })

out_path = os.environ['MATRIX_PATH']
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(
        f,
        fieldnames=['track','cfg','budget','seed','geometry_variant','policy_overrides','geometry_overrides'],
        delimiter='\t',
    )
    w.writeheader()
    w.writerows(rows)

print(f'wrote matrix: {out_path}')
print(f'jobs: {len(rows)}')
print('policy_cfgs:', ','.join(policy_cfgs))
print('geometry_cfgs:', ','.join(geometry_cfgs))
print('scale_budgets:', ','.join(scale_budgets))
print('geometry_budgets:', ','.join(geometry_budgets))
print('seeds:', ','.join(seeds))
PY

TOTAL_JOBS="$(( $(wc -l < "${MATRIX_PATH}") - 1 ))"
if (( TOTAL_JOBS <= 0 )); then
  echo "[ERROR] Empty matrix: ${MATRIX_PATH}"
  exit 1
fi

cat > "${META_PATH}" <<JSON
{
  "phase1_run_base": "${PHASE1_RUN_BASE}",
  "phase2_run_base": "${PHASE2_RUN_BASE}",
  "matrix_path": "${MATRIX_PATH}",
  "shared_test_root": "${SHARED_TEST_ROOT}",
  "phase2_experiment_id": "${PHASE2_EXPERIMENT_ID}",
  "phase2_phase": "${PHASE2_PHASE}",
  "model_flag": "${MODEL_FLAG}",
  "total_jobs": ${TOTAL_JOBS}
}
JSON

echo "[INFO] Phase-2 run base: ${PHASE2_RUN_BASE}"
echo "[INFO] Matrix: ${MATRIX_PATH}"
echo "[INFO] Meta: ${META_PATH}"
echo "[INFO] Shared test root: ${SHARED_TEST_ROOT}"
echo "[INFO] Total jobs: ${TOTAL_JOBS}"

SBATCH_CMD=(
  sbatch
  --array="0-$((TOTAL_JOBS - 1))"
  --export=ALL,PHASE2_RUN_BASE="${PHASE2_RUN_BASE}",MATRIX_PATH="${MATRIX_PATH}",SHARED_TEST_ROOT="${SHARED_TEST_ROOT}",PHASE2_EXPERIMENT_ID="${PHASE2_EXPERIMENT_ID}",PHASE2_PHASE="${PHASE2_PHASE}",MODEL_FLAG="${MODEL_FLAG}",BASELINE_EPOCHS="${BASELINE_EPOCHS}",SURROGATE_DEVICE="${SURROGATE_DEVICE}",QBC_N_TEST="${QBC_N_TEST}",TIME_H="${TIME_H}",NUM_POINTS="${NUM_POINTS}",SHARED_TEST_MAX="${SHARED_TEST_MAX}",BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE}"
  scripts/hpc/slurm_array_qbc_phase2.sh
)

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY_RUN] Not submitting to Slurm."
  echo "[DRY_RUN] Command:"
  printf ' %q' "${SBATCH_CMD[@]}"
  echo
  exit 0
fi

SBATCH_OUT="$(${SBATCH_CMD[@]})"
echo "[INFO] ${SBATCH_OUT}"
