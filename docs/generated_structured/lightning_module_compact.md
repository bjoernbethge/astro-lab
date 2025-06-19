# Lightning_Module Module

Auto-generated documentation for `astro_lab.training.lightning_module`

## Classes

### AstroLightningModule

Lightning wrapper for AstroLab models with automatic optimization.

#### Methods

**`forward(self, batch: Dict[str, Any]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]`**

Forward pass.

**`training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor`**

Training step.

**`validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor`**

Validation step.

**`test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor`**

Test step.

**`configure_optimizers(self) -> Union[torch.optim.adamw.AdamW, Dict[str, Any]]`**

Configure optimizer and scheduler.

**`predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Union[torch.Tensor, Dict[str, torch.Tensor]]`**

Prediction step.
