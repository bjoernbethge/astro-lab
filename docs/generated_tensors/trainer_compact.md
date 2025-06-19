# Trainer Module

Auto-generated documentation for `astro_lab.training.trainer`

## Classes

### AstroTrainer

High-level trainer for AstroLab models with integrated logging and optimization.

#### Methods

**`fit(self, train_dataloader: torch.utils.data.dataloader.DataLoader, val_dataloader: Optional[torch.utils.data.dataloader.DataLoader] = None, test_dataloader: Optional[torch.utils.data.dataloader.DataLoader] = None, log_hyperparameters: bool = True, log_model_architecture: bool = True, **fit_kwargs)`**

Train the model.

**`test(self, test_dataloader: torch.utils.data.dataloader.DataLoader)`**

Test the model.

**`predict(self, dataloader: torch.utils.data.dataloader.DataLoader)`**

Make predictions.

**`optimize_hyperparameters(self, model_factory: Any, train_dataloader: torch.utils.data.dataloader.DataLoader, val_dataloader: torch.utils.data.dataloader.DataLoader, n_trials: int = 50, timeout: Optional[int] = None, **optuna_kwargs)`**

Run hyperparameter optimization.

**`load_from_checkpoint(self, checkpoint_path: str)`**

Load model from checkpoint.

**`save_model(self, path: str)`**

Save model to path.

**`get_metrics(self) -> Dict[str, float]`**

Get training metrics.
