"""
Training-Modul für AstroGNN

Implementiert einen einfachen aber effektiven Trainer für das AstroGNN Modell.
Nutzt PyTorch Lightning für saubere Trainingsloops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from pathlib import Path
from typing import Dict, Any, Optional
import wandb
from tensordict import TensorDict

from .model import AstroPointCloudGNN
from .config import Config


class AstroGNNTrainer(pl.LightningModule):
    """
    PyTorch Lightning Module für AstroGNN Training
    
    Features:
    - Automatisches Logging mit TensorBoard/WandB
    - Learning Rate Scheduling
    - Metriken für Klassifikation
    - Checkpoint Saving
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Modell
        self.model = AstroPointCloudGNN(
            num_features=config.model.input_features,
            num_classes=config.model.output_classes,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
            k_neighbors=config.model.k_neighbors
        )
        
        # Metriken
        self.train_acc = Accuracy(task="multiclass", num_classes=config.model.output_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=config.model.output_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=config.model.output_classes)
        
        self.val_f1 = F1Score(task="multiclass", num_classes=config.model.output_classes, average="macro")
        self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=config.model.output_classes)
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Für Logging
        self.validation_outputs = []
        self.test_outputs = []
    
    def forward(self, batch):
        """Forward pass"""
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        # Forward pass
        logits = self(batch)
        loss = self.criterion(logits, batch.y)
        
        # Metriken
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, batch.y)
        
        # Logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        # Forward pass
        logits = self(batch)
        loss = self.criterion(logits, batch.y)
        
        # Metriken
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, batch.y)
        f1 = self.val_f1(preds, batch.y)
        
        # Speichere für epoch_end
        self.validation_outputs.append({
            'loss': loss,
            'preds': preds,
            'targets': batch.y
        })
        
        # Logging
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_epoch=True, prog_bar=True)
        self.log('val/f1', f1, on_epoch=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Am Ende der Validierungs Epoche"""
        if not self.validation_outputs:
            return
            
        # Sammle alle Predictions und Targets
        all_preds = torch.cat([x['preds'] for x in self.validation_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_outputs])
        
        # Confusion Matrix
        cm = self.confusion_matrix(all_preds, all_targets)
        
        # Log Confusion Matrix wenn WandB aktiv
        if self.logger and hasattr(self.logger, 'experiment'):
            try:
                class_names = [f"Class_{i}" for i in range(self.config.model.output_classes)]
                self.logger.experiment.log({
                    "val/confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=all_targets.cpu().numpy(),
                        preds=all_preds.cpu().numpy(),
                        class_names=class_names
                    )
                })
            except:
                pass  # Falls WandB nicht verfügbar
        
        # Clear outputs
        self.validation_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        # Forward pass
        logits = self(batch)
        
        # Metriken
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, batch.y)
        
        # Speichere für epoch_end
        self.test_outputs.append({
            'preds': preds,
            'targets': batch.y
        })
        
        # Logging
        self.log('test/acc', acc, on_epoch=True)
        
        return {'preds': preds, 'targets': batch.y}
    
    def on_test_epoch_end(self):
        """Am Ende der Test Epoch"""
        if not self.test_outputs:
            return
            
        # Sammle alle Predictions
        all_preds = torch.cat([x['preds'] for x in self.test_outputs])
        all_targets = torch.cat([x['targets'] for x in self.test_outputs])
        
        # Finale Confusion Matrix
        cm = self.confusion_matrix(all_preds, all_targets)
        
        # Speichere Confusion Matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm.cpu().numpy(), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Test Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        output_dir = Path(self.config.output_path) / self.config.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'confusion_matrix.png')
        plt.close()
        
        # Clear outputs
        self.test_outputs.clear()
    
    def configure_optimizers(self):
        """Konfiguriere Optimizer und Scheduler"""
        # Optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Scheduler
        if self.config.training.scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.epochs,
                eta_min=self.config.training.min_lr
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        elif self.config.training.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=self.config.training.patience // 2,
                factor=0.5,
                min_lr=self.config.training.min_lr
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch"
                }
            }
        else:
            return optimizer
    
    def on_train_start(self):
        """Zu Beginn des Trainings"""
        # Log Model Architecture
        if self.logger:
            self.logger.log_hyperparams(self.config.__dict__)
    
    def predict_step(self, batch, batch_idx):
        """Prediction step für Inference"""
        logits = self(batch)
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        
        return {
            'predictions': preds,
            'probabilities': probs,
            'embeddings': self.model.get_embeddings(batch)
        }


def train_model(config: Config):
    """
    Haupt-Trainingsfunktion
    
    Args:
        config: Konfigurations-Objekt
    """
    # Setze Seed für Reproduzierbarkeit
    pl.seed_everything(config.seed)
    
    # Erstelle Output-Verzeichnis
    output_dir = Path(config.output_path) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Speichere Config
    config.to_yaml(output_dir / "config.yaml")
    
    # Erstelle Trainer
    trainer_module = AstroGNNTrainer(config)
    
    # Callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="astro-gnn-{epoch:02d}-{val_acc:.3f}",
            monitor="val/acc",
            mode="max",
            save_top_k=3
        ),
        pl.callbacks.EarlyStopping(
            monitor="val/loss",
            patience=config.training.patience,
            mode="min"
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    ]
    
    # Logger
    logger = pl.loggers.TensorBoardLogger(
        save_dir=output_dir,
        name="tensorboard",
        default_hp_metric=False
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        accelerator="gpu" if config.device == "cuda" else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config.training.gradient_clip,
        deterministic=True,
        log_every_n_steps=10
    )
    
    return trainer, trainer_module
