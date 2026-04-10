"""Common metrics and probability utilities for binary Yes/No tasks."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def compute_yes_probs(logits: torch.Tensor, yes_token_id: int, no_token_id: int) -> torch.Tensor:
    """Compute the probability of the 'Yes' token relative to 'No'.

    Extracts the logits for the Yes and No tokens, applies softmax across just
    those two options, and returns the probability of Yes.

    Args:
        logits: Tensor of shape (batch, vocab_size).
        yes_token_id: Token ID for Yes.
        no_token_id: Token ID for No.

    Returns:
        Tensor of shape (batch,) with values in [0, 1].
    """
    # Extract logits for Yes (idx 0 in selection) and No (idx 1 in selection)
    target_logits = logits[:, [yes_token_id, no_token_id]]

    # Softmax over just these two options
    probs = F.softmax(target_logits, dim=-1)

    # Return probability of Yes (index 0)
    return probs[:, 0]


def compute_binary_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Compute standard binary classification metrics.

    Args:
        labels: Binary ground truth labels (0/1).
        probs: Predicted probabilities for the positive class (1).

    Returns:
        Dictionary containing:
            - auroc: Area Under the ROC Curve (↑ better)
            - auprc: Area Under the Precision-Recall Curve / Average Precision
              (↑ better)
            - brier_score: Mean squared error between predicted probabilities and binary
              labels (↓ better). Computed as mean((p_yes - label)²) via sklearn,
              equivalent to the Brier loss used during training.
    """
    metrics = {}

    # Check for single class in labels
    if len(np.unique(labels)) < 2:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    else:
        metrics["auroc"] = roc_auc_score(labels, probs)
        metrics["auprc"] = average_precision_score(labels, probs)

    metrics["brier_score"] = brier_score_loss(labels, probs)

    return metrics
