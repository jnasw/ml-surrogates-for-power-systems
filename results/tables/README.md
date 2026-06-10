# Thesis Table Exports

This directory contains compact CSV exports used by the thesis tables. The
analysis notebooks may display plots inline, but generated figures and
intermediate long-form analysis outputs should not be committed here.

Folder names omit the redundant `thesis_` prefix because everything under this
directory is thesis-facing.

## Layout

```text
results/tables/
  01_dataset_generation/
    main.csv
    main_numeric.csv
    mechanism_summary.csv
  02_optimizer_comparison/
    main.csv
    main_numeric.csv
  03_loss_balancing/
    main.csv
    main_numeric.csv
    components.csv
    components_numeric.csv
  04_collocation_comparison/
    main.csv
    main_numeric.csv
  05_multistage/
    main.csv
    main_numeric.csv
    design.csv
  06_data_augmentation/
    main.csv
    main_numeric.csv
  07_final_experiment/
    compact_sm4.csv
    compact_sm6.csv
    compact_sm_avr_gov.csv
    compact_numeric.csv
    full_external_sm4.csv
    full_external_sm6.csv
    full_external_sm_avr_gov.csv
    full_external_numeric.csv
    completion.csv
    completion_numeric.csv
  08_across_experiments/
    main.csv
```

## Conventions

- `main.csv` is the formatted table intended for direct thesis use.
- `main_numeric.csv` is the corresponding numeric table, when useful.
- Additional CSVs are allowed only when they are final thesis table data, not
  plot-ready traces or notebook scratch outputs.
- Table exports may need small path, tag, or filtering adjustments when cluster
  jobs are submitted with different campaign tags, shards, model subsets, seed
  subsets, or output roots. The notebooks are the canonical table-generation
  layer; update their input selectors to match the actual `outputs/` structure
  before re-exporting CSVs.
- Do not store `png`, `pdf`, `svg`, `npz`, notebooks, logs, or long intermediate
  analysis tables in this directory.
