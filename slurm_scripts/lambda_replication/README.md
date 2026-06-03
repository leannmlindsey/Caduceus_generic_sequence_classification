# Caduceus LAMBDA_v1 replication

Orchestration layer that fans out the **existing** Caduceus entry points
(`python -m train` for finetuning, `python -m src.inference`, `python -m
src.embedding_analysis`) across the LAMBDA_v1 windows and seeds, then picks the
best seed per window by test-set MCC and runs all diagnostic + genome-wide
inference. The model/experiment code is unchanged — these scripts only invoke it
with the right Hydra overrides and explicit output paths.

## Two-step workflow

```bash
# 0. (one time) confirm the pretrained Caduceus checkpoint + model_config.json
#    exist on Biowulf (CONFIG_PATH / PRETRAINED_PATH in lambda_replication.conf).

# 1. Edit lambda_replication.conf — confirm LAMBDA_BASE, OUTPUT_DIR,
#    CONFIG_PATH, PRETRAINED_PATH.
bash slurm_scripts/lambda_replication/run_lambda_training.sh   # finetune × seeds × windows
# 2. wait — squeue -u $USER
bash slurm_scripts/lambda_replication/check_training.sh        # confirm all seeds healthy
bash slurm_scripts/lambda_replication/run_lambda_inference.sh  # pick winner + all inference
# 3. wait — squeue -u $USER
bash slurm_scripts/lambda_replication/check_inference.sh       # confirm all outputs landed
```

## Files

| File | Role |
|------|------|
| `lambda_replication.conf` | the only file you normally edit — all paths + hyperparameters |
| `run_lambda_training.sh` | submit one finetune job per (window × variant × seed) |
| `lambda_finetune_job.sh` | sbatch body: one finetune run (calls `python -m train`) |
| `select_best_model.py` | pick best-of-N seed per (window × variant) by **test-set MCC** → `winners.json` |
| `run_lambda_inference.sh` | run winner selection + embedding + all diagnostic/genome-wide inference |
| `lambda_inference_job.sh` | sbatch body: one inference run (calls `python -m src.inference`) |
| `lambda_embedding_job.sh` | sbatch body: pretrained-embedding analysis (calls `python -m src.embedding_analysis`) |
| `print_winner_exports.py` | emit shell exports for the winning seed dir |
| `check_training.sh` / `check_inference.sh` | post-hoc verification helpers |

## Model-selection logic (two levels)

1. **Per-seed checkpoint** (inside one finetune run): the `csv_binary`
   experiment runs with `cross_validation=true`, so PyTorch Lightning saves the
   best-validation checkpoint (`monitor=val/accuracy`, `mode=max`) at
   `checkpoints/val/accuracy.ckpt` and loads it for the test phase. Left alone.
2. **Cross-seed winner** (across the N seeds): `select_best_model.py` reads each
   `seed-<N>/test_results.json` and picks the max **test-set MCC** (key
   `eval_mcc`), writing `winners.json`.

`python -m train` (with `train.test=true`, `cross_validation=true`) runs
`trainer.test()`, and the repo's **`TestResultsCallback`**
(`src/callbacks/test_results.py`, wired in by the `csv_binary` pipeline) computes
metrics with scikit-learn on the **full** test set and writes them to
`<hydra.run.dir>/test_results.json` — the test MCC is under **`eval_mcc`**.
The finetune job sets `hydra.run.dir=<OUTPUT_DIR>/<LEN>/finetune/<variant>/seed-<N>`,
so that JSON lands exactly where `select_best_model.py` reads it. **No surfacing
/ copy step is needed** — the metric source is the callback's own
`test_results.json[eval_mcc]`.

## Caduceus-specific notes

- **Single variant.** Only `caduceus` (`VARIANTS="caduceus"`,
  `MODEL_NAME=dna_embedding_caduceus`).
- **Local pretrained model (not an HF id).** `CONFIG_PATH` (model_config.json)
  and `PRETRAINED_PATH` (the seqlen-8k `last.ckpt`) live in the same pretrain
  output dir; both are set in the conf and passed to training, inference, and
  embedding analysis.
- **Char tokenizer → 1 token = 1 bp.** Per-window `MAX_LENGTH` equals the window
  in bp: `2k → 2048`, `4k → 4096`, `8k → 8192`. The pretrained checkpoint is
  seqlen-8k, so 8k is supported.
- **Winning checkpoint.** The inference job resolves the winning seed's Lightning
  checkpoint as `checkpoints/val/accuracy.ckpt` (the best-val checkpoint the test
  phase used), falling back to `checkpoints/last.ckpt`. Override with
  `WINNER_CKPT_RELPATH` in the conf.
- **`conjoin` / decoder overrides.** `CONJOIN_TEST=false` (matches the committed
  `wrapper_run_csv_binary.sh`) and `CONJOIN_TRAIN_DECODER=true`, `d_output=2`. The
  full Hydra override block is copied verbatim from `run_csv_binary.sh`.
- **`val.csv` → `dev.csv`.** LAMBDA_v1 ships `val.csv`. The csv_dataset loader
  (`src/dataloaders/genomics.py`) reads `dev.csv`, so the finetune/embedding job
  scripts stage a per-run input dir with a `dev.csv` symlink to `val.csv` (plus
  `train.csv`/`test.csv` symlinks) — no model code is modified. (Note:
  `src.embedding_analysis` already accepts either `dev.csv` or `val.csv`; the
  symlink is kept for uniformity and is harmless.)
- **Canonical output names** (required by the central harvest aggregator):
  diagnostics `test_predictions.csv` / `fpr_predictions.csv` /
  `gc_control_predictions.csv` / `fnr_predictions.csv` (each with a sibling
  `*_predictions_metrics.json`); genome-wide
  `genome_wide_<stem>_predictions.csv` (+ `_metrics.json`), where `<stem>` is the
  input CSV basename. `src.inference` derives the metrics filename by replacing
  `.csv` → `_metrics.json` on `--output_csv`.
- **Single OUTPUT_DIR tree.** Training, inference, embedding, and `winners.json`
  all live under `OUTPUT_DIR/<LEN>/{finetune,inference,embedding}/<variant>/...`.
  The finetune job redirects `hydra.run.dir` here (it does NOT use the default
  `./outputs/downstream/...` of `run_csv_binary.sh`).
- **No genome-wide aggregate analysis.** Caduceus has no genome-wide
  clustering/threshold entry point, so genome-wide inference produces per-CSV
  prediction files under `inference/<variant>/`; the harvest step does the
  clustering/genome analysis centrally (no per-repo aggregation job).
- **Bare job scripts.** No `set -e` and no `2>/dev/null` masking in the job
  bodies — `source activate` under `set -e` silently kills SLURM jobs.
- Outputs go under `/data/lindseylm/...`, never `/gpfs/gsfs12/...`.
