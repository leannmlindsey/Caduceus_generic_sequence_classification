<p align="center">
    <img src="assets/Caduceus_image.png" alt="Caduceus" width="200"/>
</p>

# Caduceus Generic Sequence Classification

> **Note:** This is a fork of the [original Caduceus repository](https://github.com/kuleshov-group/caduceus) with added support for **generic CSV-based binary/multiclass classification** tasks. This allows you to fine-tune Caduceus (or Mamba/Hyena) on your own DNA sequence classification datasets.

---

## Fine-tuning on Custom CSV Datasets

This fork adds the ability to fine-tune on any classification task using simple CSV files.

### 1. Prepare Your Data

Create a directory containing three CSV files with `sequence` and `label` columns:

```
my_dataset/
├── train.csv
├── dev.csv
└── test.csv
```

Each CSV should have this format:
```csv
sequence,label
ACGTACGTACGT...,0
TGCATGCATGCA...,1
GGCCAATTGGCC...,0
```

- `sequence`: DNA sequence (A, C, G, T, N characters)
- `label`: Integer class label (0, 1 for binary; 0, 1, 2, ... for multiclass)

### 2. Pretrain a Model (or use existing checkpoint)

> **Note:** The HuggingFace checkpoints are not directly compatible with this fine-tuning pipeline (see [this issue](https://github.com/kuleshov-group/caduceus/issues/72) for details). You need to pretrain your own model using this codebase, or use an existing checkpoint that was trained with this repository.

To pretrain a Caduceus model, follow the [Pretraining on Human Reference Genome](#pretraining) instructions in the original README below.

**After pretraining, locate these two files in your pretraining output directory:**

```
outputs/pretrain/hg38/<your_pretraining_run_name>/
├── model_config.json          # <-- CONFIG_PATH: Model architecture configuration
└── checkpoints/
    └── last.ckpt              # <-- PRETRAINED_PATH: Model weights
```

You will need the full paths to both `model_config.json` and `last.ckpt` for fine-tuning.

### 3. Configure the SLURM Script

> **Important:** You only need to edit **one file**: `slurm_scripts/wrapper_run_csv_binary.sh`. The config files in `configs/` do not need to be modified.

Edit `slurm_scripts/wrapper_run_csv_binary.sh` with your paths:

```bash
# === REQUIRED: Dataset Configuration ===
export DATA_DIR="/path/to/my_dataset"        # Directory containing train.csv, dev.csv, test.csv
export DATASET_NAME="my_dataset"              # Name for output directory (no spaces)

# === REQUIRED: Model Configuration (from your pretraining output) ===
export CONFIG_PATH="/path/to/outputs/pretrain/hg38/<run_name>/model_config.json"
export PRETRAINED_PATH="/path/to/outputs/pretrain/hg38/<run_name>/checkpoints/last.ckpt"

# === Model Type (must match what you pretrained) ===
export MODEL="caduceus"                       # Options: hyena, mamba, caduceus
export MODEL_NAME="dna_embedding_caduceus"    # Options: dna_embedding, dna_embedding_mamba, dna_embedding_caduceus

# === Hyperparameters (adjust as needed) ===
export LR="6e-4"                              # Learning rate
export BATCH_SIZE="32"                        # Batch size (reduce if OOM)
export MAX_LENGTH="1024"                      # Max sequence length (truncates longer sequences)
export MAX_EPOCHS="100"                       # Training epochs
export D_OUTPUT="2"                           # Number of classes (2 for binary)
```

### 4. Understanding Reverse Complement Parameters

> **Important:** The RC parameters you use for fine-tuning **must match** the model variant you pretrained. Using mismatched settings will result in poor performance or errors.

Caduceus has special handling for reverse complement (RC) sequences. The parameters depend on which model variant you pretrained:

**Caduceus Variants:**
- **Caduceus-Ph (Post-hoc)**: Runs both forward and reverse complement sequences through the model at test time, then averages predictions.
- **Caduceus-PS (Parameter Sharing)**: The model architecture is inherently RC equivariant, handling both directions internally.

**Parameter Definitions:**

| Parameter | Description |
|-----------|-------------|
| `CONJOIN_TEST` | If `true`, run both forward + RC sequences at test time and average predictions |
| `CONJOIN_TRAIN_DECODER` | If `true`, decoder expects input with shape `(..., 2)` and combines both channels |
| `RC_AUG` | If `true`, randomly apply RC augmentation during training |

**Required Settings by Pretrained Model Type:**

| If you pretrained... | `CONJOIN_TEST` | `CONJOIN_TRAIN_DECODER` | `RC_AUG` |
|----------------------|----------------|-------------------------|----------|
| **Caduceus-Ph** | `true` | `false` | `false` |
| **Caduceus-PS** | `false` | `true` | `false` |
| **Mamba** | `false` | `false` | `true` |
| **Hyena** | `false` | `false` | `true` |

The default settings in the SLURM scripts are configured for **Caduceus-Ph**. If you pretrained a different model variant, update these parameters in `wrapper_run_csv_binary.sh`:

```bash
# === Reverse Complement Settings (must match your pretrained model) ===
export CONJOIN_TEST="true"              # true for Caduceus-Ph, false for others
export CONJOIN_TRAIN_DECODER="false"    # true for Caduceus-PS, false for others
export RC_AUG="false"                   # true for Mamba/Hyena, false for Caduceus
```

### 5. Submit the Job

```bash
cd slurm_scripts
bash wrapper_run_csv_binary.sh
```

Or submit directly:
```bash
sbatch --export=ALL,DATA_DIR=/path/to/data,CONFIG_PATH=/path/to/config.json,PRETRAINED_PATH=/path/to/ckpt.ckpt,DATASET_NAME=mydata run_csv_binary.sh
```

### 6. Output and Test Metrics

**Output Directory Structure:**
```
outputs/downstream/csv_binary/{DATASET_NAME}/{MODEL}_lr-{LR}_batch_size-{BS}_rc_aug-{RC}/seed-{SEED}/
├── test_results.json       # Comprehensive metrics computed on full test set
├── test_predictions.npz    # Raw predictions for further analysis
└── checkpoints/            # Model checkpoints
```

**How Test Metrics Are Computed:**

This fork includes a custom test evaluation callback (`src/callbacks/test_results.py`) that addresses a common issue with batch-level metric averaging. Metrics like MCC (Matthews Correlation Coefficient) can be incorrectly calculated when averaged across batches, especially if individual batches have imbalanced class distributions.

Our solution:
1. **Collects all predictions** from the entire test set during evaluation
2. **Computes metrics using scikit-learn** on the complete predictions (not batch averages)
3. **Saves results** to `test_results.json` with full traceability

**Output Files:**

| File | Description |
|------|-------------|
| `test_results.json` | All metrics computed on the full test set, plus paths to data and checkpoint |
| `test_predictions.npz` | NumPy archive containing `logits`, `probabilities`, `predictions`, and `labels` arrays for custom analysis |

**The `test_results.json` contains:**
```json
{
  "eval_loss": 0.258,
  "eval_accuracy": 0.931,
  "eval_precision": 0.957,
  "eval_recall": 0.904,
  "eval_f1": 0.930,
  "eval_mcc": 0.864,
  "eval_sensitivity": 0.904,
  "eval_specificity": 0.959,
  "eval_auc": 0.983,
  "eval_runtime": 30.15,
  "eval_samples_per_second": 226.4,
  "eval_steps_per_second": 3.55,
  "epoch": 100.0,
  "checkpoint_path": "/path/to/checkpoint.ckpt",
  "train_data_path": "/path/to/train.csv",
  "dev_data_path": "/path/to/dev.csv",
  "test_data_path": "/path/to/test.csv"
}
```

**Loading predictions for custom analysis:**
```python
import numpy as np

data = np.load('test_predictions.npz')
logits = data['logits']           # Raw model outputs
probs = data['probabilities']     # Softmax probabilities
preds = data['predictions']       # Predicted class labels
labels = data['labels']           # True labels
```

### 7. SLURM Scripts

Two SLURM scripts are provided in `slurm_scripts/` for running on HPC clusters (configured for NIH Biowulf):

**File Locations:**
```
slurm_scripts/
├── wrapper_run_csv_binary.sh   # <-- EDIT THIS FILE with your paths
└── run_csv_binary.sh           # Main job script (no edits needed)
```

**`wrapper_run_csv_binary.sh`** - Configuration wrapper (EDIT THIS)
- Set your dataset path, model checkpoint, and hyperparameters here
- Validates paths before submitting
- Calls `sbatch` to submit the job

**`run_csv_binary.sh`** - Main SLURM job script (no edits needed)
- Contains SBATCH directives (partition, GPU, memory, time limit)
- Loads conda environment (`caduceus_env`)
- Runs the training with parameters from the wrapper

**What to Edit in `wrapper_run_csv_binary.sh`:**

| Variable | Description | Example |
|----------|-------------|---------|
| `DATA_DIR` | Path to folder with train.csv, dev.csv, test.csv | `/data/user/my_dataset` |
| `DATASET_NAME` | Name for output folder (no spaces) | `phage_detection` |
| `CONFIG_PATH` | Path to `model_config.json` from pretraining | `/data/user/pretrain/model_config.json` |
| `PRETRAINED_PATH` | Path to checkpoint from pretraining | `/data/user/pretrain/checkpoints/last.ckpt` |
| `MODEL` | Model type | `caduceus`, `mamba`, or `hyena` |
| `MODEL_NAME` | Model registry name | `dna_embedding_caduceus` |
| `LR` | Learning rate | `6e-4` |
| `BATCH_SIZE` | Batch size (reduce if OOM) | `32` |
| `MAX_LENGTH` | Max sequence length | `1024` |
| `MAX_EPOCHS` | Training epochs | `100` |
| `D_OUTPUT` | Number of classes | `2` |

**Modifying SLURM Resources (if needed):**

If you need to change GPU type, memory, or time limit, edit `run_csv_binary.sh`:

```bash
#SBATCH --partition=gpu          # Partition name
#SBATCH --gres=gpu:a100:1        # GPU type and count
#SBATCH --mem=64g                # Memory
#SBATCH --cpus-per-task=8        # CPU cores
#SBATCH --time=24:00:00          # Time limit
```

**Changing the Conda Environment:**

If your conda environment has a different name, edit this line in `run_csv_binary.sh`:

```bash
source activate caduceus_env     # Change 'caduceus_env' to your env name
```

### Key Files Added in This Fork

| File | Description |
|------|-------------|
| `src/dataloaders/datasets/csv_dataset.py` | PyTorch Dataset for loading CSV files |
| `src/dataloaders/genomics.py` | Added `CSVDatasetLoader` class |
| `src/callbacks/test_results.py` | Callback for computing metrics with scikit-learn |
| `configs/experiment/csv_binary.yaml` | Experiment config for CSV classification |
| `configs/pipeline/csv_binary.yaml` | Pipeline config |
| `configs/dataset/csv_dataset.yaml` | Dataset config |
| `slurm_scripts/run_csv_binary.sh` | SLURM job script (Biowulf compatible) |
| `slurm_scripts/wrapper_run_csv_binary.sh` | Helper script for job submission |

---

## Original Caduceus README

The remainder of this README is from the [original Caduceus repository](https://github.com/kuleshov-group/caduceus).

---

# Caduceus &#9764;: Bi-Directional Equivariant Long-Range DNA Sequence Modeling
[[Blog]](https://caduceus-dna.github.io/) &nbsp; | &nbsp; [[arXiv]](https://arxiv.org/abs/2403.03234) &nbsp; | &nbsp; [[HuggingFace 🤗]](https://huggingface.co/collections/kuleshov-group/caducues-65dcb89b4f54e416ef61c350)

This repository contains code for reproducing the results in the paper "Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling," [Schiff et al. (2024)](https://arxiv.org/abs/2403.03234).

## Using Caduceus with 🤗
<a name="HF"></a>
We have uploaded a pre-trained Caduceus model to the Huggingface hub.
The available models are:
- Caduceus-Ph: [kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16](https://huggingface.co/kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16)
  - Trained on sequences of length 131k, with a model size of 256 and 16 layers.
  - Trained for 50k steps and batch size of 8.
  - Trained with reverse-complement (RC) data augmentation.
- Caduceus-PS: [kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16](https://huggingface.co/kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16)
  - Trained on sequences of length 131k, with a model size of 256 and 16 layers.
  - Trained for 50k steps and batch size of 8.
  - Model is RC equivariant, hence no RC data augmentation is required.

You can either use the pre-trained model directly within your trainer scripts or modify the config that initializes the model.

To use the pre-trained model for masked language modeling, use the following snippet:
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

# See the `Caduceus` collection page on the hub for list of available models.
model_name = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
```

Alternatively, you can instantiate a model from scratch to train on your own data as follows:
```python
from transformers import AutoConfig, AutoModelForMaskedLM

# Add any config overrides here, see the `config.json` file on the hub for details.
config_overrides = {}
# See the `Caduceus` collection page on the hub for list of available models.
config = AutoConfig.from_pretrained(
 "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
 **config_overrides,
)
model = AutoModelForMaskedLM.from_config(config)
```

## Getting started in this repository
<a name="getting_started"></a>

To get started, create a conda environment containing the required dependencies.

```bash
conda env create -f caduceus_env.yml
```

Activate the environment.

```bash
conda activate caduceus_env
```

Create the following directories to store saved models and slurm logs:
```bash
mkdir outputs
mkdir watch_folder
```

## Reproducing Experiments

Below, we describe the steps required for reproducing the experiments in the paper.
Throughout, the main entry point for running experiments is the [`train.py`](./train.py) script.
We also provide sample `slurm` scripts for launching pre-training and downstream fine-tuning experiments in the [`slurm_scripts/`](./slurm_scripts) directory.

### Pretraining on Human Reference Genome
<a name="pretraining"></a>
(Data downloading instructions are copied from [HyenaDNA repo](https://github.com/HazyResearch/hyena-dna?tab=readme-ov-file#pretraining-on-human-reference-genome))

First, download the Human Reference Genome data.
It's comprised of 2 files, 1 with all the sequences (the `.fasta` file), and with the intervals we use (`.bed` file).

The file structure should look like

```
data
|-- hg38/
    |-- hg38.ml.fa
    |-- human-sequences.bed
```

Download fasta (.fa format) file (of the entire human genome) into `./data/hg38`.
~24 chromosomes in the whole genome (merged into 1 file), each chromosome is a continuous sequence, basically.
Then download the .bed file with sequence intervals (contains chromosome name, start, end, split, which then allow you to retrieve from the fasta file).
```bash
mkdir -p data/hg38/
curl https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz > data/hg38/hg38.ml.fa.gz
gunzip data/hg38/hg38.ml.fa.gz  # unzip the fasta file
curl https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed > data/hg38/human-sequences.bed
```

Launch pretraining run using the command line

```bash
python -m train \
  experiment=hg38/hg38 \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
  dataset.max_length=1024 \
  dataset.batch_size=1024 \
  dataset.mlm=true \
  dataset.mlm_probability=0.15 \
  dataset.rc_aug=false \
  model=caduceus \
  model.config.d_model=128 \
  model.config.n_layer=4 \
  model.config.bidirectional=true \
  model.config.bidirectional_strategy=add \
  model.config.bidirectional_weight_tie=true \
  model.config.rcps=true \
  optimizer.lr="8e-3" \
  train.global_batch_size=1024 \
  trainer.max_steps=10000 \
  +trainer.val_check_interval=10000 \
  wandb=null
```

or alternatively, if using a cluster that has `slurm` installed, adapt the scripts below:
```
slurm_scripts
|-- run_pretrain_caduceus.sh
|-- run_pretrain_hyena.sh
|-- run_pretrain_mamba.sh
```

and run the training as a batch job:
```bash
cd slurm_scripts
sbatch run_pretrain_caduceus.sh
```

### GenomicBenchmarks
<a name="genomicbenchmarks"></a>

The [GenomicBenchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks) presented in [Grešová et al. (2023)](https://bmcgenomdata.biomedcentral.com/articles/10.1186/s12863-023-01123-8) is comprised of 8 classification tasks.

We can launch a downstream fine-tuning run on one of the tasks using the sample command below:
```bash
python -m train \
    experiment=hg38/genomic_benchmark \
    callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000 \
    dataset.dataset_name="dummy_mouse_enhancers_ensembl" \
    dataset.train_val_split_seed=1 \
    dataset.batch_size=256 \
    dataset.rc_aug=false \
    +dataset.conjoin_train=false \
    +dataset.conjoin_test=false \
    loader.num_workers=2 \
    model=caduceus \
    model._name_=dna_embedding_caduceus \
    +model.config_path="<path to model_config.json>" \
    +model.conjoin_test=false \
    +decoder.conjoin_train=true \
    +decoder.conjoin_test=false \
    optimizer.lr="1e-3" \
    trainer.max_epochs=10 \
    train.pretrained_model_path="<path to .ckpt file>" \
    wandb=null
```

This sample run will fine-tune a pre-trained Caduceus-PS model on the `dummy_mouse_enhancers_ensembl` task.
Note some of the additional arguments present here, relative to the pre-training command from [above](#pretraining):
- `model.config_path` contains the path model config that was saved during pre-training.
This will be saved to the run directory of the pre-training experiment.
- `train.pretrained_model_path` contains the path to the pre-trained model checkpoint.
- `dataset.conjoin_train` determines whether the dataset will return a single sequence (`dataset.conjoin_train=false`) or the concatenation of a sequence and its reverse complement along `dim=-1`, during downstream fine-tuning training.
- `dataset.conjoin_test` is the same as above, but for inference (e.g., validation / test).
- `decoder.conjoin_train` determines whether the prediction head (a mean pooling and linear projection in the case of the Genomics Benchmark) is expecting an input tensor of shape `(batch_size, seq_len, d_model)` or `(batch_size, seq_len, d_model, 2)` during downstream fine-tuning training.
When set to `true` the decoder is run on `input[..., 0]` and `input[..., 1]` and the results are averaged to produce the final prediction.
- `decoder.conjoin_test` is the same as above, but for inference (e.g., validation / test).

Note this benchmark only contains a training and test split for each task.
Therefore, to have a more principled evaluation, we randomly split the training data into training and validation sets (90/10) using the `dataset.train_val_split_seed` argument.
We perform early stopping on validation metric (accuracy) and repeat this for 5 random seeds.

As with [pre-training](#pretraining), we can also launch the fine-tuning run as a batch job using the provided [`run_genomic_benchmark.sh`](./slurm_scripts/run_genomics_benchmark.sh) script.
We also provide a helper shell script [`wrapper_run_genomics.sh`](./slurm_scripts/wrapper_run_genomics.sh) that can be used to launch multiple fine-tuning runs in parallel.

Finally, the [`run_genomics_benchmark_cnn.sh`](./slurm_scripts/run_genomics_benchmark_cnn.sh) script can be used to train the CNN baseline for this experiment from scratch on the downstream tasks.

### Nucleotide Transformer datasets
<a name="nucleotidetransformer"></a>

The Nucleotide Transformer suite of tasks was proposed in [Dalla-Torre et al. (2023)](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v1).
The data is available on HuggingFace: [InstaDeepAI/nucleotide_transformer_downstream_tasks](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks).

We can launch a downstream fine-tuning run on one of the tasks using the sample command below:
```bash
python -m train \
    experiment=hg38/nucleotide_transformer \
    callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000 \
    dataset.dataset_name="${task}" \
    dataset.train_val_split_seed=${seed} \
    dataset.batch_size=${batch_size} \
    dataset.rc_aug="${rc_aug}" \
    +dataset.conjoin_test="${CONJOIN_TEST}" \
    loader.num_workers=2 \
    model._name_=dna_embedding_caduceus \
    +model.config_path="<path to model_config.json>" \
    +model.conjoin_test=false \
    +decoder.conjoin_train=true \
    +decoder.conjoin_test=false \
    optimizer.lr="1e-3" \
    trainer.max_epochs=10 \
    train.pretrained_model_path="<path to .ckpt file>" \
    trainer.max_epochs=20 \
    wandb=null
```

We can also launch as batch jobs (see [`run_nucleotide_transformer.sh`](./slurm_scripts/run_nucleotide_transformer.sh) and [`wrapper_run_nucleotide_transformer.sh`](./slurm_scripts/wrapper_run_nucleotide_transformer.sh) for details).

### eQTL SNP Variant Effect Prediction
<a name="vep"></a>
This task comes from the recently proposed Long Range Benchmark (LRB) in [Kao et al., 2023](https://llms4science-community.github.io/papers/LLMs4Bio24_paper_12.pdf).
The data is available on HuggingFace: [InstaDeepAI/genomics-long-range-benchmark](https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark).
For this task we fit a model to the pre-trained and frozen embeddings of the DNA language models.
Therefore, to perform the evaluation, we proceed in 2 steps:
- **Step 1: Extract the embeddings** from the pre-trained model:
Run the [`vep_embeddings.py`](./vep_embeddings.py) script to extract the embeddings from the pre-trained model.
See the example below:
```bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=8 \
    vep_embeddings.py \
      --num_workers=2 \
      --seq_len=131072  \
      --bp_per_token=1  \
      --embed_dump_batch_size=1 \
      --name="caduceus-ps_downstream-seqlen=131k"  \
      --model_name_or_path="kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16" \
      --rcps
```

The `--rcps` flag is used to indicate that the model is reverse-complement equivariant.
When using other models, set this flag to false with `--no-rcps`.
To speed this step up, this script utilizes torch distributed data parallelism.

Please refer to the slurm script provided in [`slurm_scripts/dump_vep_embeddings.sh`](./slurm_scripts/dump_vep_embeddings.sh)
to launch this step as a batch job.

- **Step 2: Fit an SVM model to the embeddings** using this notebook: [`vep_svm.ipynb`](./vep_svm.ipynb).

## Citation
<a name="citation"></a>

If you find our work useful, please cite our paper using the following:
```
@article{schiff2024caduceus,
  title={Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling},
  author={Schiff, Yair and Kao, Chia-Hsiang and Gokaslan, Aaron and Dao, Tri and Gu, Albert and Kuleshov, Volodymyr},
  journal={arXiv preprint arXiv:2403.03234},
  year={2024}
}
```

## Acknowledgements
<a name="acknowledgements"></a>
This repository is adapted from the [HyenaDNA repo](https://github.com/HazyResearch/hyena-dna) and leverages much of the training, data loading, and logging infrastructure defined there.
HyenaDNA was originally derived from the [S4](https://github.com/state-spaces/s4) and [Safari](https://github.com/HazyResearch/safari) repositories.

We would like to thank Evan Trop and the [InstaDeep](https://www.instadeep.com/) team for useful discussions about the [Nucleotide Transformer leaderboard](https://huggingface.co/spaces/InstaDeepAI/nucleotide_transformer_benchmark)
and the Long Range Benchmark task.

Finally, we would like to thank [MosaicML](https://www.mosaicml.com/) for providing compute resources for some of the pre-training experiments.
