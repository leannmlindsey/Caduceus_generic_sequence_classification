"""
Inference Script for Caduceus

This script performs inference on a CSV file using a fine-tuned Caduceus model.
It outputs predictions with probability scores for threshold analysis.

Input CSV format:
    - sequence: DNA sequence
    - label: Ground truth label (optional, used for comparison)

Output CSV format:
    - sequence: Original sequence
    - label: Original label (if present)
    - prob_0: Probability of class 0
    - prob_1: Probability of class 1
    - pred_label: Predicted label (argmax or threshold-based)

Usage:
    python -m src.inference \
        --input_csv /path/to/test.csv \
        --checkpoint_path /path/to/checkpoint.ckpt \
        --config_path /path/to/config.json \
        --output_csv /path/to/predictions.csv
"""

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)

# Caduceus-specific imports
from caduceus.configuration_caduceus import CaduceusConfig
from caduceus.modeling_caduceus import Caduceus, CaduceusForSequenceClassification
from transformers import AutoTokenizer


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on CSV file with fine-tuned Caduceus model"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV file with 'sequence' column (and optionally 'label')",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to fine-tuned Caduceus checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to Caduceus config JSON file",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file (default: input_csv with _predictions suffix)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--d_output",
        type=int,
        default=2,
        help="Number of output classes",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for prob_1 (default: 0.5)",
    )
    parser.add_argument(
        "--save_metrics",
        action="store_true",
        help="If labels are present, calculate and save metrics to JSON",
    )
    parser.add_argument(
        "--conjoin_test",
        action="store_true",
        help="Use post-hoc reverse complement conjoining (recommended for Caduceus)",
    )
    return parser.parse_args()


class CaduceusClassifier(nn.Module):
    """
    Caduceus model with classification head for inference.
    """

    def __init__(self, config: CaduceusConfig, d_output: int = 2, conjoin_test: bool = False):
        super().__init__()
        self.config = config
        self.d_output = d_output
        self.conjoin_test = conjoin_test

        # Backbone
        self.backbone = Caduceus(config)

        # Classification head - simple linear layer to match training decoder
        d_model = config.d_model
        if config.rcps:
            d_model = d_model * 2  # RCPS doubles the hidden dimension

        self.classifier = nn.Linear(d_model, d_output)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass with mean pooling.
        """
        if self.conjoin_test:
            # Process forward and reverse complement separately
            # input_ids shape: (batch, seq_len, 2) - forward and RC
            if input_ids.dim() == 3:
                fwd_ids = input_ids[:, :, 0]
                rc_ids = input_ids[:, :, 1]

                fwd_hidden = self.backbone(fwd_ids, output_hidden_states=False, return_dict=False)
                rc_hidden = self.backbone(rc_ids, output_hidden_states=False, return_dict=False)

                # Mean of forward and reverse complement
                hidden_states = (fwd_hidden + rc_hidden) / 2
            else:
                hidden_states = self.backbone(input_ids, output_hidden_states=False, return_dict=False)
        else:
            hidden_states = self.backbone(input_ids, output_hidden_states=False, return_dict=False)

        # Mean pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = hidden_states.mean(dim=1)

        logits = self.classifier(pooled)
        return logits


def get_complement_map():
    """
    Get the complement map for DNA tokenization.
    Maps token IDs to their reverse complement token IDs.
    """
    # Standard DNA tokenizer vocabulary:
    # 0: [PAD], 1: [UNK], 2: [CLS], 3: [SEP], 4: [MASK], 5: ., 6: <reserved>,
    # 7: A, 8: C, 9: G, 10: T, 11: N
    complement_map = {
        0: 0,   # [PAD] -> [PAD]
        1: 1,   # [UNK] -> [UNK]
        2: 2,   # [CLS] -> [CLS]
        3: 3,   # [SEP] -> [SEP]
        4: 4,   # [MASK] -> [MASK]
        5: 5,   # . -> .
        6: 6,   # <reserved> -> <reserved>
        7: 10,  # A -> T
        8: 9,   # C -> G
        9: 8,   # G -> C
        10: 7,  # T -> A
        11: 11, # N -> N
    }
    return complement_map


def load_model(
    config_path: str,
    checkpoint_path: str,
    d_output: int,
    conjoin_test: bool,
    device: torch.device,
) -> CaduceusClassifier:
    """Load Caduceus model with classification head from checkpoint."""
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # Handle nested Hydra config format
    # The config file may have a nested 'config' key containing the actual model config
    if 'config' in config_dict and isinstance(config_dict['config'], dict):
        print("  Detected nested Hydra config format, extracting inner config...")
        inner_config = config_dict['config']
        # Remove Hydra-specific keys that start with '_'
        inner_config = {k: v for k, v in inner_config.items() if not k.startswith('_')}
        config_dict = inner_config

    # Handle complement_map being null or missing (required for RCPS)
    if not config_dict.get('complement_map'):
        print("  Setting complement_map for DNA tokenization...")
        config_dict['complement_map'] = get_complement_map()

    print(f"  d_model: {config_dict.get('d_model', 'not specified')}")
    print(f"  n_layer: {config_dict.get('n_layer', 'not specified')}")
    print(f"  rcps: {config_dict.get('rcps', 'not specified')}")

    config = CaduceusConfig(**config_dict)
    model = CaduceusClassifier(config, d_output=d_output, conjoin_test=conjoin_test)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Map state dict keys to model
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key

        # Handle 'model.caduceus.backbone.' prefix from fine-tuned checkpoints
        if key.startswith('model.caduceus.backbone.'):
            new_key = 'backbone.backbone.' + key[24:]  # len('model.caduceus.backbone.') = 24
            new_state_dict[new_key] = value
            continue

        # Handle 'model.' prefix from PyTorch Lightning
        if key.startswith('model.'):
            new_key = key[6:]  # Remove 'model.'

        # Map decoder to classifier
        if key.startswith('decoder.0.output_transform.'):
            new_key = 'classifier.' + key[27:]  # len('decoder.0.output_transform.') = 27
            new_state_dict[new_key] = value
            continue

        # Map caduceus backbone keys
        if new_key.startswith('caduceus.'):
            new_key = 'backbone.' + new_key[9:]
        elif not new_key.startswith('backbone.') and not new_key.startswith('classifier.'):
            if 'decoder' in new_key:
                continue  # Skip other decoder keys
            else:
                new_key = 'backbone.' + new_key

        new_state_dict[new_key] = value

    # Load with strict=False to handle missing classifier keys
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"  Missing keys (will use random init): {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys (ignored): {len(unexpected)}")

    model = model.to(device)
    model.eval()

    return model


def get_tokenizer():
    """Get the character tokenizer for DNA sequences."""
    tokenizer = AutoTokenizer.from_pretrained(
        "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
        trust_remote_code=True,
    )
    return tokenizer


def string_reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N',
                  'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n'}
    return ''.join(complement.get(base, base) for base in reversed(seq))


def run_inference(
    model: CaduceusClassifier,
    tokenizer,
    sequences: List[str],
    batch_size: int,
    max_length: int,
    conjoin_test: bool,
    device: torch.device,
) -> tuple:
    """
    Run inference on sequences.

    Returns:
        Tuple of (probabilities array shape (n, num_classes), predictions array)
    """
    model.eval()
    all_probs = []
    all_preds = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Running inference"):
        batch_seqs = sequences[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_seqs,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        if conjoin_test:
            # Create reverse complement sequences
            batch_seqs_rc = [string_reverse_complement(s) for s in batch_seqs]
            inputs_rc = tokenizer(
                batch_seqs_rc,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            # Stack forward and RC: (batch, seq_len, 2)
            input_ids = torch.stack([input_ids, inputs_rc["input_ids"]], dim=-1)

        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_probs.append(probs)
            all_preds.extend(preds)

    probs_array = np.vstack(all_probs)
    preds_array = np.array(all_preds)

    return probs_array, preds_array


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    """Calculate comprehensive metrics."""
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
    }

    try:
        metrics["auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
    except ValueError:
        metrics["auc"] = 0.0

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)

    return metrics


def main():
    """Main function to run inference."""
    args = parse_arguments()

    print("\n" + "=" * 60)
    print("Caduceus Inference")
    print("=" * 60)

    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load input CSV
    print(f"\nLoading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    if "sequence" not in df.columns:
        raise ValueError("Input CSV must have a 'sequence' column")

    has_labels = "label" in df.columns
    print(f"  Samples: {len(df)}")
    print(f"  Has labels: {has_labels}")

    # Load model and tokenizer
    model = load_model(
        args.config_path,
        args.checkpoint_path,
        args.d_output,
        args.conjoin_test,
        device,
    )
    tokenizer = get_tokenizer()

    # Run inference
    sequences = df["sequence"].tolist()
    probs, preds = run_inference(
        model, tokenizer, sequences,
        args.batch_size, args.max_length, args.conjoin_test, device,
    )

    # Apply custom threshold if specified
    if args.threshold != 0.5:
        print(f"\nApplying custom threshold: {args.threshold}")
        preds_thresholded = (probs[:, 1] >= args.threshold).astype(int)
    else:
        preds_thresholded = preds

    # Create output dataframe
    output_df = df.copy()
    output_df["prob_0"] = probs[:, 0]
    output_df["prob_1"] = probs[:, 1]
    output_df["pred_label"] = preds_thresholded

    # Set output path
    if args.output_csv is None:
        base, ext = os.path.splitext(args.input_csv)
        args.output_csv = f"{base}_predictions{ext}"

    # Save predictions
    output_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved predictions to: {args.output_csv}")

    # Calculate and save metrics if labels present
    if has_labels and args.save_metrics:
        labels = df["label"].values
        metrics = calculate_metrics(labels, preds_thresholded, probs)

        metrics["checkpoint_path"] = args.checkpoint_path
        metrics["config_path"] = args.config_path
        metrics["input_csv"] = args.input_csv
        metrics["threshold"] = args.threshold
        metrics["conjoin_test"] = args.conjoin_test
        metrics["num_samples"] = len(df)

        metrics_path = args.output_csv.replace(".csv", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_path}")

        print("\n" + "=" * 60)
        print("METRICS (threshold = {:.2f})".format(args.threshold))
        print("=" * 60)
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        print(f"  MCC:         {metrics['mcc']:.4f}")
        print(f"  AUC:         {metrics['auc']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print("=" * 60)

    elif has_labels:
        labels = df["label"].values
        acc = accuracy_score(labels, preds_thresholded)
        print(f"\nAccuracy: {acc:.4f}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Throughput: {len(df) / elapsed:.1f} sequences/second")


if __name__ == "__main__":
    main()
