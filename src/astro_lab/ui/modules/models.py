"""
Models Module - UI components for model management and training
==============================================================

Direct integration with AstroLab model and training classes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import marimo as mo
import mlflow

from astro_lab.cli.config import load_and_prepare_training_config
from astro_lab.models.core import (
    create_lightning_model,
    create_preset_model,
    list_lightning_models,
    list_presets,
)
from astro_lab.training import AstroTrainer


class ModelsModule:
    """UI module for model management and training."""

    def __init__(self):
        """Initialize models module with state management."""
        self.state, self.set_state = mo.state(
            {
                "available_models": list_lightning_models(),
                "available_presets": list_presets(),
                "current_model": None,
                "trainer": None,
                "training_status": "idle",  # idle, training, completed, failed
                "training_progress": 0,
                "training_logs": [],
                "mlflow_runs": [],
            }
        )

        # Refresh MLflow runs
        self._refresh_mlflow_runs()

    def create_model_lab(self) -> mo.Html:
        """Create model lab component."""
        state = self.state()

        # Model selector
        model_type = mo.ui.radio(
            options=["Preset", "Custom Model"],
            value="Preset",
            label="Model Type",
        )

        # Dynamic selector based on type
        if model_type.value == "Preset":
            model_selector = mo.ui.dropdown(
                options=list(state()["available_presets"].keys()),
                label="Select Preset",
            )
        else:
            model_selector = mo.ui.dropdown(
                options=list(state()["available_models"].keys()),
                label="Select Model",
            )

        # Training configuration
        config_section = self._create_training_config()

        # Training controls
        controls = self._create_training_controls(
            model_type.value, model_selector.value
        )

        # Training status
        status = self._create_training_status()

        # Model info
        model_info = self._create_model_info()

        return mo.vstack(
            [
                mo.md("## 🤖 Model Lab"),
                mo.hstack([model_type, model_selector]),
                config_section,
                controls,
                status,
                model_info,
            ]
        )

    def create_experiment_tracker(self) -> mo.Html:
        """Create MLflow experiment tracker."""
        state = self.state()

        # Refresh button
        refresh_btn = mo.ui.button(
            "🔄 Refresh",
            on_click=self._refresh_mlflow_runs,
        )

        # Runs table
        runs_table = self._create_runs_table()

        # Run details
        run_details = self._create_run_details()

        return mo.vstack(
            [
                mo.md("## 📊 Experiment Tracker"),
                mo.hstack(
                    [
                        mo.md("MLflow Experiments"),
                        refresh_btn,
                    ]
                ),
                runs_table,
                run_details,
            ]
        )

    def create_model_registry(self) -> mo.Html:
        """Create model registry interface."""
        # List registered models
        models = self._get_registered_models()

        if not models:
            return mo.vstack(
                [
                    mo.md("## 📦 Model Registry"),
                    mo.Html("""
                    <div class="info-message">
                        No models registered yet. Train a model to get started.
                    </div>
                """),
                ]
            )

        # Create model cards
        model_cards = []
        for model in models:
            card = self._create_model_card(model)
            model_cards.append(card)

        return mo.vstack(
            [
                mo.md("## 📦 Model Registry"),
                mo.Html(
                    """
                <div class="model-grid">
                    {cards}
                </div>
            """.format(cards="".join(model_cards))
                ),
            ]
        )

    def _create_training_config(self) -> mo.Html:
        """Create training configuration form."""
        # Basic hyperparameters
        epochs = mo.ui.slider(
            value=50,
            start=1,
            stop=500,
            step=1,
            label="Max Epochs",
        )

        batch_size = mo.ui.slider(
            value=32,
            start=1,
            stop=256,
            step=1,
            label="Batch Size",
        )

        learning_rate = mo.ui.number(
            value=0.001,
            start=0.00001,
            stop=1.0,
            step=0.0001,
            label="Learning Rate",
        )

        # Advanced settings
        optimizer = mo.ui.dropdown(
            options=["adamw", "adam", "sgd", "rmsprop"],
            value="adamw",
            label="Optimizer",
        )

        scheduler = mo.ui.dropdown(
            options=["cosine", "step", "exponential", "onecycle", "none"],
            value="cosine",
            label="LR Scheduler",
        )

        precision = mo.ui.dropdown(
            options=["16-mixed", "bf16-mixed", "32-true"],
            value="16-mixed",
            label="Precision",
        )

        # Store config in state
        config = {
            "max_epochs": epochs.value,
            "batch_size": batch_size.value,
            "learning_rate": learning_rate.value,
            "optimizer": optimizer.value,
            "scheduler": scheduler.value,
            "precision": precision.value,
        }

        self.set_state(lambda s: {**s, "training_config": config})

        return mo.accordion(
            {
                "⚙️ Training Configuration": mo.vstack(
                    [
                        mo.hstack([epochs, batch_size]),
                        mo.hstack([learning_rate, optimizer]),
                        mo.hstack([scheduler, precision]),
                    ]
                ),
            }
        )

    def _create_training_controls(self, model_type: str, model_name: str) -> mo.Html:
        """Create training control buttons."""
        state = self.state()

        # Train button
        train_btn = mo.ui.button(
            "🚀 Start Training",
            on_click=lambda: self._start_training(model_type, model_name),
            disabled=state()["training_status"] == "training" or not model_name,
            variant="primary",
        )

        # Stop button
        stop_btn = mo.ui.button(
            "⏹️ Stop Training",
            on_click=self._stop_training,
            disabled=state()["training_status"] != "training",
            variant="danger",
        )

        # Resume button
        resume_btn = mo.ui.button(
            "▶️ Resume",
            on_click=self._resume_training,
            disabled=state()["training_status"] != "stopped",
        )

        return mo.hstack([train_btn, stop_btn, resume_btn])

    def _create_training_status(self) -> mo.Html:
        """Create training status display."""
        state = self.state()
        status = state()["training_status"]
        progress = state()["training_progress"]

        # Status indicator
        status_emoji = {
            "idle": "⏸️",
            "training": "🔄",
            "completed": "✅",
            "failed": "❌",
            "stopped": "⏹️",
        }

        # Progress bar
        progress_bar = (
            mo.Html(f"""
            <div class="progress-container">
                <label>Training Progress</label>
                <progress value="{progress}" max="100"></progress>
                <span>{progress}%</span>
            </div>
        """)
            if status == "training"
            else mo.Html("")
        )

        # Training logs
        logs = state.get("training_logs", [])
        logs_display = (
            mo.Html(f"""
            <div class="training-logs">
                <h4>Training Logs</h4>
                <pre>{"\\n".join(logs[-10:])}</pre>
            </div>
        """)
            if logs
            else mo.Html("")
        )

        return mo.vstack(
            [
                mo.md(f"### Status: {status_emoji[status]} {status.capitalize()}"),
                progress_bar,
                logs_display,
            ]
        )

    def _start_training(self, model_type: str, model_name: str):
        """Start model training."""
        if not model_name:
            return

        state = self.state()
        config = state.get("training_config", {})

        # Get data module from data module
        from .data import data_module

        dm_state = data_module.state()
        datamodule = dm_state.get("datamodule")

        if not datamodule:
            mo.output.append(mo.md("❌ No data loaded! Please load data first."))
            return

        # Prepare full config
        full_config = {
            **config,
            "dataset": dm_state.get("current_dataset", "gaia"),
        }

        if model_type == "Preset":
            full_config["preset"] = model_name
        else:
            full_config["model"] = model_name

        # Create trainer
        try:
            trainer = AstroTrainer(full_config)
            trainer.datamodule = datamodule  # Use existing datamodule

            self.set_state(
                lambda s: {
                    **s,
                    "trainer": trainer,
                    "training_status": "training",
                    "training_progress": 0,
                    "training_logs": ["Starting training..."],
                }
            )

            # Start training (in real app, this would be async)
            mo.output.append(mo.md("🚀 Training started!"))

            # Note: In production, training should run in background thread
            success = trainer.train()

            if success:
                self.set_state(
                    lambda s: {
                        **s,
                        "training_status": "completed",
                        "training_progress": 100,
                    }
                )
                mo.output.append(mo.md("✅ Training completed successfully!"))
            else:
                self.set_state(
                    lambda s: {
                        **s,
                        "training_status": "failed",
                    }
                )
                mo.output.append(mo.md("❌ Training failed!"))

            # Refresh MLflow runs
            self._refresh_mlflow_runs()

        except Exception as e:
            self.set_state(
                lambda s: {
                    **s,
                    "training_status": "failed",
                    "training_logs": s["training_logs"] + [f"Error: {str(e)}"],
                }
            )
            mo.output.append(mo.md(f"❌ Training error: {str(e)}"))

    def _stop_training(self):
        """Stop current training."""
        state = self.state()
        trainer = state.get("trainer")

        if trainer and trainer.trainer:
            # In real implementation, this would signal the trainer to stop
            self.set_state(
                lambda s: {
                    **s,
                    "training_status": "stopped",
                }
            )
            mo.output.append(mo.md("⏹️ Training stopped."))

    def _resume_training(self):
        """Resume training from checkpoint."""
        # In real implementation, this would resume from last checkpoint
        mo.output.append(mo.md("▶️ Resume functionality not yet implemented."))

    def _create_model_info(self) -> mo.Html:
        """Display current model information."""
        state = self.state()
        trainer = state.get("trainer")

        if not trainer or not trainer.model:
            return mo.Html("")

        model = trainer.model

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return mo.vstack(
            [
                mo.md("### Model Information"),
                mo.md(f"**Architecture:** {model.__class__.__name__}"),
                mo.md(f"**Total Parameters:** {total_params:,}"),
                mo.md(f"**Trainable Parameters:** {trainable_params:,}"),
                mo.md(
                    f"**Input Features:** {getattr(model, 'in_features', 'Unknown')}"
                ),
                mo.md(
                    f"**Output Classes:** {getattr(model, 'num_classes', 'Unknown')}"
                ),
            ]
        )

    def _refresh_mlflow_runs(self):
        """Refresh MLflow experiment runs."""
        try:
            # Get all experiments
            experiments = mlflow.search_experiments()

            all_runs = []
            for exp in experiments:
                runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                for _, run in runs.iterrows():
                    all_runs.append(
                        {
                            "run_id": run["run_id"],
                            "experiment": exp.name,
                            "status": run["status"],
                            "start_time": run["start_time"],
                            "metrics": {
                                "val_loss": run.get("metrics.val_loss", None),
                                "val_acc": run.get("metrics.val_acc", None),
                            },
                            "params": {
                                "model": run.get("params.model", "Unknown"),
                                "dataset": run.get("params.dataset", "Unknown"),
                                "epochs": run.get("params.max_epochs", None),
                            },
                        }
                    )

            self.set_state(lambda s: {**s, "mlflow_runs": all_runs})

        except Exception as e:
            mo.output.append(mo.md(f"⚠️ Could not refresh MLflow runs: {str(e)}"))

    def _create_runs_table(self) -> mo.Html:
        """Create MLflow runs table."""
        state = self.state()
        runs = state.get("mlflow_runs", [])

        if not runs:
            return mo.Html("""
                <div class="info-message">
                    No experiment runs found. Train a model to create runs.
                </div>
            """)

        # Create table HTML
        rows = []
        for run in runs[:10]:  # Show latest 10
            rows.append(f"""
                <tr>
                    <td>{run["experiment"]}</td>
                    <td>{run["params"]["model"]}</td>
                    <td>{run["params"]["dataset"]}</td>
                    <td>{run["status"]}</td>
                    <td>{run["metrics"]["val_loss"]:.4f if run['metrics']['val_loss'] else 'N/A'}</td>
                    <td>{run["metrics"]["val_acc"]:.3f if run['metrics']['val_acc'] else 'N/A'}</td>
                    <td>
                        <button onclick="viewRun('{run["run_id"]}')">View</button>
                    </td>
                </tr>
            """)

        return mo.Html(f"""
            <table class="runs-table">
                <thead>
                    <tr>
                        <th>Experiment</th>
                        <th>Model</th>
                        <th>Dataset</th>
                        <th>Status</th>
                        <th>Val Loss</th>
                        <th>Val Acc</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        """)

    def _create_run_details(self) -> mo.Html:
        """Create run details view."""
        # Placeholder for selected run details
        return mo.Html("")

    def _get_registered_models(self) -> List[Dict]:
        """Get registered models from MLflow."""
        try:
            client = mlflow.tracking.MlflowClient()
            models = []

            for model in client.list_registered_models():
                latest_version = max(
                    model.latest_versions, key=lambda v: int(v.version)
                )
                models.append(
                    {
                        "name": model.name,
                        "version": latest_version.version,
                        "stage": latest_version.current_stage,
                        "description": model.description or "No description",
                    }
                )

            return models

        except Exception:
            return []

    def _create_model_card(self, model: Dict) -> str:
        """Create model card HTML."""
        return f"""
            <div class="model-card">
                <h4>{model["name"]}</h4>
                <p class="model-version">v{model["version"]} · {model["stage"]}</p>
                <p class="model-description">{model["description"]}</p>
                <div class="model-actions">
                    <button onclick="loadModel('{model["name"]}', {model["version"]})">Load</button>
                    <button onclick="deployModel('{model["name"]}', {model["version"]})">Deploy</button>
                </div>
            </div>
        """


# Create singleton instance
models_module = ModelsModule()
