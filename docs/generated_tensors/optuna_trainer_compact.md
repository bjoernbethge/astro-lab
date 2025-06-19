# Optuna_Trainer Module

Auto-generated documentation for `astro_lab.training.optuna_trainer`

## Classes

### OptunaTrainer

Modern Optuna-based hyperparameter optimization with MLflow integration.

#### Methods

**`objective(self, trial: optuna.trial._trial.Trial) -> float`**

Optuna objective function for hyperparameter optimization.

**`optimize(self, n_trials: int = 100, timeout: Optional[float] = None) -> optuna.study.study.Study`**

Run hyperparameter optimization.

Args:
n_trials: Number of trials to run
timeout: Timeout in seconds (optional)

Returns:
Completed Optuna study

**`get_best_model(self) -> astro_lab.training.lightning_module.AstroLightningModule`**

Create and return the best model found during optimization.

**`save_study(self, filepath: str)`**

Save the study to a file.
