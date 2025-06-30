"""
Metrics Mixin for AstroLab Models
================================

Pure PyTorch metrics computation for astronomical models.
"""

from typing import Dict, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData


class MetricsMixin:
    """Pure PyTorch metrics computation for astronomical models."""

    def calculate_metrics(
        self, logits: Tensor, batch: Union[Data, HeteroData, Batch], stage: str
    ) -> Dict[str, float]:
        """Calculate task-specific metrics using pure PyTorch."""
        metrics = {}
        task = getattr(self, "task", "classification")
        num_classes = getattr(self, "num_classes", 2)

        if task in ["node_classification", "graph_classification"]:
            metrics.update(
                self._classification_metrics(logits, batch, stage, num_classes)
            )
        elif task in ["node_regression", "graph_regression", "edge_regression"]:
            metrics.update(self._regression_metrics(logits, batch, stage))
        elif task == "link_prediction":
            metrics.update(self._link_prediction_metrics(logits, batch))

        return metrics

    def _classification_metrics(
        self,
        logits: Tensor,
        batch: Union[Data, HeteroData, Batch],
        stage: str,
        num_classes: int,
    ) -> Dict[str, float]:
        """Classification metrics using pure PyTorch."""
        # Extract predictions and labels
        if (
            hasattr(batch, f"{stage}_mask")
            and getattr(batch, f"{stage}_mask") is not None
        ):
            mask = getattr(batch, f"{stage}_mask")
            pred_logits = logits[mask]
            true_labels = batch.y[mask] if hasattr(batch, "y") and batch.y is not None else None
        else:
            pred_logits = logits
            true_labels = batch.y if hasattr(batch, "y") and batch.y is not None else None

        if true_labels is None:
            # Return empty metrics for unsupervised/demo scenarios
            return {"accuracy": 0.0, "note": "no_labels_available"}

        metrics = {}

        if num_classes == 2:
            # Binary classification
            probs = torch.sigmoid(pred_logits.squeeze())
            preds = (probs > 0.5).long()

            # Basic metrics
            correct = (preds == true_labels).float()
            metrics["accuracy"] = correct.mean().item()

            # Precision, Recall, F1
            tp = ((preds == 1) & (true_labels == 1)).float().sum()
            fp = ((preds == 1) & (true_labels == 0)).float().sum()
            fn = ((preds == 0) & (true_labels == 1)).float().sum()
            tn = ((preds == 0) & (true_labels == 0)).float().sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            metrics.update(
                {
                    "precision": precision.item(),
                    "recall": recall.item(),
                    "specificity": specificity.item(),
                    "f1": f1.item(),
                }
            )

            # ROC AUC using pure PyTorch
            if len(torch.unique(true_labels)) == 2:
                metrics["auc"] = self._compute_auc_torch(true_labels, probs)
        else:
            # Multi-class classification
            preds = pred_logits.argmax(dim=-1)
            correct = (preds == true_labels).float()
            metrics["accuracy"] = correct.mean().item()

            # Per-class accuracy
            if num_classes <= 20:
                for class_idx in range(num_classes):
                    class_mask = true_labels == class_idx
                    if class_mask.sum() > 0:
                        class_acc = (
                            (preds[class_mask] == true_labels[class_mask])
                            .float()
                            .mean()
                        )
                        metrics[f"acc_class_{class_idx}"] = class_acc.item()

            # Macro metrics
            macro_precision, macro_recall, macro_f1 = self._compute_macro_metrics_torch(
                preds, true_labels, num_classes
            )
            metrics.update(
                {
                    "macro_precision": macro_precision,
                    "macro_recall": macro_recall,
                    "macro_f1": macro_f1,
                }
            )

        return metrics

    def _regression_metrics(
        self, logits: Tensor, batch: Union[Data, HeteroData, Batch], stage: str
    ) -> Dict[str, float]:
        """Regression metrics using pure PyTorch."""
        # Extract predictions and labels
        if (
            hasattr(batch, f"{stage}_mask")
            and getattr(batch, f"{stage}_mask") is not None
        ):
            mask = getattr(batch, f"{stage}_mask")
            pred_values = logits[mask]
            true_values = batch.y[mask] if hasattr(batch, "y") else None
        else:
            pred_values = logits.squeeze()
            true_values = batch.y.squeeze() if hasattr(batch, "y") else None

        if true_values is None:
            return {"mse": 0.0, "mae": 0.0}

        # Basic regression metrics
        mse = F.mse_loss(pred_values, true_values)
        mae = F.l1_loss(pred_values, true_values)
        rmse = torch.sqrt(mse)

        # R² score
        ss_res = ((true_values - pred_values) ** 2).sum()
        ss_tot = ((true_values - true_values.mean()) ** 2).sum()
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        # MAPE (Mean Absolute Percentage Error)
        mape = (
            torch.mean(
                torch.abs((true_values - pred_values) / (torch.abs(true_values) + 1e-8))
            )
            * 100
        )

        # Explained variance
        explained_var = 1 - torch.var(true_values - pred_values) / (
            torch.var(true_values) + 1e-8
        )

        return {
            "mse": mse.item(),
            "mae": mae.item(),
            "rmse": rmse.item(),
            "r2": r2.item(),
            "mape": mape.item(),
            "explained_variance": explained_var.item(),
        }

    def _link_prediction_metrics(
        self, logits: Tensor, batch: Union[Data, HeteroData, Batch]
    ) -> Dict[str, float]:
        """Link prediction metrics using pure PyTorch."""
        # Split positive and negative predictions
        if hasattr(batch, "pos_edge_index"):
            pos_logits = logits[: batch.pos_edge_index.size(1)]
            neg_logits = logits[batch.pos_edge_index.size(1) :]

            pos_probs = torch.sigmoid(pos_logits)
            neg_probs = torch.sigmoid(neg_logits)

            pos_preds = (pos_probs > 0.5).float()
            neg_preds = (neg_probs > 0.5).float()

            pos_acc = pos_preds.mean().item()
            neg_acc = (1 - neg_preds).mean().item()

            # Overall metrics
            all_probs = torch.cat([pos_probs, neg_probs])
            all_labels = torch.cat(
                [torch.ones_like(pos_probs), torch.zeros_like(neg_probs)]
            )

            return {
                "pos_accuracy": pos_acc,
                "neg_accuracy": neg_acc,
                "link_accuracy": (pos_acc + neg_acc) / 2,
                "link_auc": self._compute_auc_torch(all_labels, all_probs),
            }
        else:
            return {"link_accuracy": 0.0}

    def _compute_auc_torch(self, labels: Tensor, scores: Tensor) -> float:
        """Compute ROC AUC using pure PyTorch (trapezoidal rule)."""
        # Sort by scores
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_labels = labels[sorted_indices]

        # Compute TPR and FPR
        n_pos = (labels == 1).sum().float()
        n_neg = (labels == 0).sum().float()

        if n_pos == 0 or n_neg == 0:
            return 0.5  # Random performance

        tp = torch.cumsum(sorted_labels, dim=0).float()
        fp = torch.cumsum(1 - sorted_labels, dim=0).float()

        tpr = tp / n_pos
        fpr = fp / n_neg

        # Add (0,0) point
        tpr = torch.cat([torch.tensor([0.0], device=tpr.device), tpr])
        fpr = torch.cat([torch.tensor([0.0], device=fpr.device), fpr])

        # Compute AUC using trapezoidal rule
        auc = torch.trapz(tpr, fpr).item()
        return auc

    def _compute_macro_metrics_torch(
        self, preds: Tensor, labels: Tensor, num_classes: int
    ) -> tuple[float, float, float]:
        """Compute macro-averaged precision, recall, F1 using pure PyTorch."""
        precisions = []
        recalls = []
        f1s = []

        for class_idx in range(num_classes):
            class_preds = preds == class_idx
            class_labels = labels == class_idx

            tp = (class_preds & class_labels).float().sum()
            fp = (class_preds & ~class_labels).float().sum()
            fn = (~class_preds & class_labels).float().sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            precisions.append(precision.item())
            recalls.append(recall.item())
            f1s.append(f1.item())

        return (
            sum(precisions) / len(precisions),
            sum(recalls) / len(recalls),
            sum(f1s) / len(f1s),
        )
