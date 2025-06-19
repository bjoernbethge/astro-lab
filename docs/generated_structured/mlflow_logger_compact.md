# Mlflow_Logger Module

Auto-generated documentation for `astro_lab.training.mlflow_logger`

## Functions

### setup_mlflow_experiment(experiment_name: str, tracking_uri: Optional[str] = None, artifact_location: Optional[str] = None) -> str

Setup MLflow experiment for AstroLab training.

## Classes

### AstroMLflowLogger

Enhanced MLflow logger for astronomical models.

#### Methods

**`log_model_architecture(self, model: torch.nn.modules.module.Module)`**

Log model architecture details.

**`log_hyperparameters(self, params: Dict[str, Any])`**

Log hyperparameters with astronomical context.

**`log_dataset_info(self, dataset_info: Dict[str, Any])`**

Log dataset information.

**`log_survey_info(self, survey: str, bands: Optional[list] = None)`**

Log astronomical survey information.

**`log_final_model(self, model: torch.nn.modules.module.Module, model_name: str = 'astro_model')`**

Log final trained model.

**`log_predictions(self, predictions_file: str)`**

Log prediction results as artifacts.

**`log_confusion_matrix(self, cm_path: str)`**

Log confusion matrix plot.

**`end_run(self)`**

End MLflow run with cleanup.
