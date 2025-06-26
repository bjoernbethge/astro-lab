"""
Training UI Module - Direct integration with AstroLab training
=============================================================

UI components that directly use AstroLab training classes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import marimo as mo
import mlflow
import yaml

from astro_lab.cli.config import load_and_prepare_training_config
from astro_lab.models.config import get_preset
from astro_lab.models.core import (
    list_lightning_models,
    list_presets,
)

# Direct imports from AstroLab
from astro_lab.training import AstroTrainer

from .data import get_current_datamodule


def training_dashboard() -> mo.Html:
    """Main training dashboard using AstroTrainer."""
    # State
    state, set_state = mo.state(
        {
            "trainer": None,
            "training_active": False,
            "training_config": {},
        }
    )

    # Model selection
    model_type = mo.ui.radio(
        options=["preset", "custom"],
        value="preset",
        label="Model Type",
    )

    # Dynamic model selector
    if model_type.value == "preset":
        available = list(list_presets().keys())
        model_select = mo.ui.dropdown(
            options=available,
            value=available[0] if available else None,
            label="Select Preset",
        )
    else:
        available = list(list_lightning_models().keys())
        model_select = mo.ui.dropdown(
            options=available,
            value=available[0] if available else None,
            label="Select Model",
        )

    # Training parameters
    epochs = mo.ui.slider(
        value=50,
        min=1,
        max=500,
        step=1,
        label="Epochs",
    )

    learning_rate = mo.ui.number(
        value=0.001,
        min=0.00001,
        max=1.0,
        step=0.0001,
        label="Learning Rate",
    )

    batch_size = mo.ui.slider(
        value=32,
        min=1,
        max=256,
        step=1,
        label="Batch Size",
    )

    optimizer = mo.ui.dropdown(
        options=["adamw", "adam", "sgd", "rmsprop"],
        value="adamw",
        label="Optimizer",
    )

    def start_training():
        """Start training using AstroTrainer."""
        # Get DataModule
        dm = get_current_datamodule()
        if not dm:
            mo.output.append(mo.md("❌ No DataModule loaded! Load data first."))
            return

        # Prepare config
        config = {
            "max_epochs": epochs.value,
            "learning_rate": learning_rate.value,
            "batch_size": batch_size.value,
            "optimizer": optimizer.value,
            "dataset": dm.survey,
            "num_workers": 0,  # For laptop
            "precision": "16-mixed",
        }

        if model_type.value == "preset":
            config["preset"] = model_select.value
        else:
            config["model"] = model_select.value

        try:
            # Create trainer
            trainer = AstroTrainer(config)
            trainer.datamodule = dm  # Use existing DataModule

            set_state(
                lambda s: {
                    **s,
                    "trainer": trainer,
                    "training_active": True,
                    "training_config": config,
                }
            )

            mo.output.append(mo.md("🚀 Starting training..."))

            # Create model
            trainer.create_model()
            trainer.create_trainer()

            # Note: In production, this should be async
            success = trainer.train()

            set_state(lambda s: {**s, "training_active": False})

            if success:
                mo.output.append(mo.md("✅ Training completed successfully!"))
            else:
                mo.output.append(mo.md("❌ Training failed!"))

        except Exception as e:
            set_state(lambda s: {**s, "training_active": False})
            mo.output.append(mo.md(f"❌ Error: {str(e)}"))

    train_btn = mo.ui.button(
        "🚀 Start Training",
        on_click=start_training,
        disabled=state()["training_active"],
        kind="primary",
    )

    # Display current status
    if state()["training_active"]:
        status = mo.md("🔄 **Training in progress...**")
    elif state()["trainer"]:
        status = mo.md("✅ **Training completed**")
    else:
        status = mo.md("⏸️ **Ready to train**")

    return mo.vstack(
        [
            mo.md("## 🤖 Training Dashboard"),
            mo.hstack([model_type, model_select]),
            mo.accordion(
                {
                    "⚙️ Training Parameters": mo.vstack(
                        [
                            epochs,
                            learning_rate,
                            mo.hstack([batch_size, optimizer]),
                        ]
                    )
                }
            ),
            train_btn,
            status,
        ]
    )


def model_selector() -> mo.Html:
    """Model selector with detailed information."""
    # State
    state, set_state = mo.state(
        {
            "selected_model": None,
            "model_info": None,
        }
    )

    # Model lists
    presets = list_presets()
    models = list_lightning_models()

    def show_model_info(model_type: str, model_name: str):
        """Show detailed model information."""
        if model_type == "preset":
            info = get_preset(model_name)
            set_state(
                lambda s: {
                    **s,
                    "selected_model": model_name,
                    "model_info": info,
                }
            )
        else:
            info = models.get(model_name, {})
            set_state(
                lambda s: {
                    **s,
                    "selected_model": model_name,
                    "model_info": {"name": model_name, "description": info},
                }
            )

    # Preset cards
    preset_cards = []
    for name, preset in presets.items():
        card = mo.Html(f"""
        <div class="model-card" style="cursor: pointer; padding: 1rem; border: 1px solid #ddd; border-radius: 8px; margin: 0.5rem;">
            <h4>{name}</h4>
            <p>{preset.get("description", "No description")}</p>
            <p><small>Model: {preset.get("model_name", "Unknown")}</small></p>
        </div>
        """)
        preset_cards.append(
            mo.ui.button(
                card,
                on_click=lambda n=name: show_model_info("preset", n),
                kind="neutral",
                full_width=True,
            )
        )

    # Model cards
    model_cards = []
    for name, model_class in models.items():
        card = mo.Html(f"""
        <div class="model-card" style="cursor: pointer; padding: 1rem; border: 1px solid #ddd; border-radius: 8px; margin: 0.5rem;">
            <h4>{name}</h4>
            <p>{model_class.__doc__ or "No description"}</p>
        </div>
        """)
        model_cards.append(
            mo.ui.button(
                card,
                on_click=lambda n=name: show_model_info("model", n),
                kind="neutral",
                full_width=True,
            )
        )

    # Info display
    info = state()["model_info"]
    if info:
        info_display = mo.md(f"""
        ### Selected: {state()["selected_model"]}
        
        {yaml.dump(info, default_flow_style=False)}
        """)
    else:
        info_display = mo.md("*Select a model to see details*")

    return mo.vstack(
        [
            mo.md("## 🎯 Model Selector"),
            mo.tabs(
                {
                    "🚀 Presets": mo.vstack(preset_cards),
                    "🏗️ Models": mo.vstack(model_cards),
                }
            ),
            info_display,
        ]
    )


def experiment_tracker() -> mo.Html:
    """MLflow experiment tracker."""
    # State
    state, set_state = mo.state(
        {
            "experiments": [],
            "runs": [],
            "selected_run": None,
        }
    )

    def refresh_experiments():
        """Refresh MLflow experiments."""
        try:
            experiments = mlflow.search_experiments()
            set_state(lambda s: {**s, "experiments": experiments})

            # Get all runs
            all_runs = []
            for exp in experiments:
                runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                all_runs.extend(runs.to_dict("records"))

            set_state(lambda s: {**s, "runs": all_runs[:20]})  # Last 20 runs
            mo.output.append(mo.md(f"✅ Found {len(all_runs)} runs"))

        except Exception as e:
            mo.output.append(mo.md(f"⚠️ MLflow error: {str(e)}"))

    refresh_btn = mo.ui.button(
        "🔄 Refresh",
        on_click=refresh_experiments,
    )

    # Runs table
    runs = state()["runs"]
    if runs:
        # Create simplified table data
        table_data = []
        for run in runs:
            table_data.append(
                {
                    "run_id": run.get("run_id", "")[:8],
                    "status": run.get("status", ""),
                    "model": run.get("params.model", "Unknown"),
                    "dataset": run.get("params.dataset", "Unknown"),
                    "val_loss": f"{run.get('metrics.val_loss', 0):.4f}"
                    if run.get("metrics.val_loss")
                    else "N/A",
                    "val_acc": f"{run.get('metrics.val_acc', 0):.3f}"
                    if run.get("metrics.val_acc")
                    else "N/A",
                }
            )

        runs_table = mo.ui.table(table_data)
    else:
        runs_table = mo.md("*No runs found. Train a model to create runs.*")

    return mo.vstack(
        [
            mo.md("## 📊 Experiment Tracker"),
            refresh_btn,
            runs_table,
        ]
    )


# Store trainer instance for sharing
_current_trainer = None


def get_current_trainer() -> Optional[AstroTrainer]:
    """Get current trainer instance."""
    return _current_trainer


def set_current_trainer(trainer: AstroTrainer):
    """Set current trainer instance."""
    global _current_trainer
    _current_trainer = trainer
