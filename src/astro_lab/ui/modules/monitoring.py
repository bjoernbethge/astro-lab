"""
Monitoring UI Module - System and training monitoring
===================================================

UI components for monitoring system resources and training progress.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import marimo as mo
import mlflow
import psutil
import torch

from astro_lab.memory import force_comprehensive_cleanup, get_memory_stats

from .training import get_current_trainer


def system_monitor() -> mo.Html:
    """Monitor system resources."""
    # Get system info
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()

    # CPU usage
    cpu_display = mo.Html(f"""
    <div style="margin: 1rem 0;">
        <h4>CPU Usage: {cpu_percent}%</h4>
        <div style="background: #e0e0e0; height: 20px; border-radius: 10px;">
            <div style="background: #4CAF50; width: {cpu_percent}%; height: 100%; border-radius: 10px;"></div>
        </div>
    </div>
    """)

    # Memory usage
    mem_percent = memory.percent
    mem_used = memory.used / (1024**3)  # GB
    mem_total = memory.total / (1024**3)  # GB

    memory_display = mo.Html(f"""
    <div style="margin: 1rem 0;">
        <h4>Memory: {mem_used:.1f} / {mem_total:.1f} GB ({mem_percent}%)</h4>
        <div style="background: #e0e0e0; height: 20px; border-radius: 10px;">
            <div style="background: #2196F3; width: {mem_percent}%; height: 100%; border-radius: 10px;"></div>
        </div>
    </div>
    """)

    # Disk usage
    disk = psutil.disk_usage("/")
    disk_display = mo.Html(f"""
    <div style="margin: 1rem 0;">
        <h4>Disk: {disk.used / (1024**3):.1f} / {disk.total / (1024**3):.1f} GB ({disk.percent}%)</h4>
        <div style="background: #e0e0e0; height: 20px; border-radius: 10px;">
            <div style="background: #FF9800; width: {disk.percent}%; height: 100%; border-radius: 10px;"></div>
        </div>
    </div>
    """)

    # Refresh button
    refresh_btn = mo.ui.button(
        label="🔄 Refresh",
        on_click=lambda: mo.output.append(
            mo.md(f"Refreshed at {datetime.now().strftime('%H:%M:%S')}")
        ),
    )

    # Memory cleanup button
    cleanup_btn = mo.ui.button(
        label="🧹 Clean Memory",
        on_click=lambda: _cleanup_memory(),
        kind="neutral",
    )

    return mo.vstack(
        [
            mo.md("## 💻 System Monitor"),
            mo.hstack([refresh_btn, cleanup_btn]),
            cpu_display,
            memory_display,
            disk_display,
        ]
    )


def gpu_monitor() -> mo.Html:
    """Monitor GPU resources using AstroLab memory management."""
    if not torch.cuda.is_available():
        return mo.vstack(
            [
                mo.md("## 🎮 GPU Monitor"),
                mo.md("*No GPU detected. Running on CPU.*"),
            ]
        )

    # Get GPU info
    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)

    # Get memory stats from AstroLab
    memory_stats = get_memory_stats()
    gpu_stats = memory_stats.get("gpu", {})

    # Memory display
    allocated_gb = gpu_stats.get("allocated_gb", 0)
    reserved_gb = gpu_stats.get("reserved_gb", 0)
    total_gb = gpu_stats.get("total_gb", 0)

    if total_gb > 0:
        allocated_percent = (allocated_gb / total_gb) * 100
        reserved_percent = (reserved_gb / total_gb) * 100
    else:
        allocated_percent = 0
        reserved_percent = 0

    gpu_display = mo.Html(f"""
    <div style="margin: 1rem 0;">
        <h3>{device_name}</h3>
        
        <h4>Allocated: {allocated_gb:.2f} / {total_gb:.2f} GB ({allocated_percent:.1f}%)</h4>
        <div style="background: #e0e0e0; height: 20px; border-radius: 10px; margin-bottom: 1rem;">
            <div style="background: #4CAF50; width: {allocated_percent}%; height: 100%; border-radius: 10px;"></div>
        </div>
        
        <h4>Reserved: {reserved_gb:.2f} / {total_gb:.2f} GB ({reserved_percent:.1f}%)</h4>
        <div style="background: #e0e0e0; height: 20px; border-radius: 10px;">
            <div style="background: #2196F3; width: {reserved_percent}%; height: 100%; border-radius: 10px;"></div>
        </div>
    </div>
    """)

    # Utilization (if nvidia-ml-py is available)
    try:
        import nvidia_ml_py as nvml

        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(device)
        utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = utilization.gpu

        util_display = mo.Html(f"""
        <div style="margin: 1rem 0;">
            <h4>GPU Utilization: {gpu_util}%</h4>
            <div style="background: #e0e0e0; height: 20px; border-radius: 10px;">
                <div style="background: #FF5722; width: {gpu_util}%; height: 100%; border-radius: 10px;"></div>
            </div>
        </div>
        """)
    except Exception:
        util_display = mo.md("*GPU utilization not available (install nvidia-ml-py)*")

    # Refresh button
    refresh_btn = mo.ui.button(
        label="🔄 Refresh",
        on_click=lambda: mo.output.append(
            mo.md(f"Refreshed at {datetime.now().strftime('%H:%M:%S')}")
        ),
    )

    # Clear GPU cache button
    clear_cache_btn = mo.ui.button(
        label="🧹 Clear GPU Cache",
        on_click=lambda: _clear_gpu_cache(),
        kind="neutral",
    )

    return mo.vstack(
        [
            mo.md("## 🎮 GPU Monitor"),
            mo.hstack([refresh_btn, clear_cache_btn]),
            gpu_display,
            util_display,
        ]
    )


def mlflow_dashboard() -> mo.Html:
    """MLflow tracking dashboard."""
    # State
    state, set_state = mo.state(
        {
            "experiments": [],
            "active_run": None,
            "metrics": {},
        }
    )

    def refresh_mlflow():
        """Refresh MLflow data."""
        try:
            # Get experiments
            experiments = mlflow.search_experiments()
            set_state(lambda s: {**s, "experiments": experiments})

            # Get active run if any
            active_run = mlflow.active_run()
            if active_run:
                set_state(lambda s: {**s, "active_run": active_run.info})

                # Get latest metrics
                client = mlflow.tracking.MlflowClient()
                metrics = client.get_run(active_run.info.run_id).data.metrics
                set_state(lambda s: {**s, "metrics": metrics})

            mo.output.append(mo.md("✅ MLflow data refreshed"))

        except Exception as e:
            mo.output.append(mo.md(f"⚠️ MLflow error: {str(e)}"))

    refresh_btn = mo.ui.button(
        label="🔄 Refresh",
        on_click=refresh_mlflow,
    )

    # Display active run
    active_run = state()["active_run"]
    if active_run:
        run_display = mo.md(f"""
        ### Active Run
        - **Run ID:** {active_run.run_id[:8]}
        - **Experiment:** {active_run.experiment_id}
        - **Status:** {active_run.status}
        """)

        # Display metrics
        metrics = state()["metrics"]
        if metrics:
            metrics_list = []
            for key, value in metrics.items():
                metrics_list.append(f"- **{key}:** {value:.4f}")

            metrics_display = mo.md(f"""
            ### Latest Metrics
            {chr(10).join(metrics_list)}
            """)
        else:
            metrics_display = mo.md("*No metrics recorded yet*")
    else:
        run_display = mo.md("*No active MLflow run*")
        metrics_display = mo.md("")

    # Experiments list
    experiments = state()["experiments"]
    if experiments:
        exp_list = []
        for exp in experiments[:5]:  # Show last 5
            exp_list.append(f"- {exp.name} (ID: {exp.experiment_id})")

        exp_display = mo.md(f"""
        ### Recent Experiments
        {chr(10).join(exp_list)}
        """)
    else:
        exp_display = mo.md("*No experiments found*")

    return mo.vstack(
        [
            mo.md("## 📊 MLflow Dashboard"),
            refresh_btn,
            run_display,
            metrics_display,
            exp_display,
        ]
    )


def training_monitor() -> mo.Html:
    """Monitor active training progress."""
    trainer = get_current_trainer()

    if not trainer:
        return mo.vstack(
            [
                mo.md("## 📈 Training Monitor"),
                mo.md("*No active training. Start training to see progress.*"),
            ]
        )

    # Training status
    if hasattr(trainer, "trainer") and trainer.trainer:
        current_epoch = trainer.trainer.current_epoch
        max_epochs = trainer.trainer.max_epochs
        progress = (current_epoch / max_epochs) * 100 if max_epochs > 0 else 0

        progress_display = mo.Html(f"""
        <div style="margin: 1rem 0;">
            <h4>Training Progress: Epoch {current_epoch} / {max_epochs}</h4>
            <div style="background: #e0e0e0; height: 30px; border-radius: 15px;">
                <div style="background: #4CAF50; width: {progress}%; height: 100%; border-radius: 15px;
                            display: flex; align-items: center; justify-content: center; color: white;">
                    {progress:.1f}%
                </div>
            </div>
        </div>
        """)
    else:
        progress_display = mo.md("*Training not started*")

    # Model info
    if trainer.model:
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )

        model_info = mo.md(f"""
        ### Model Information
        - **Architecture:** {trainer.model.__class__.__name__}
        - **Total Parameters:** {total_params:,}
        - **Trainable Parameters:** {trainable_params:,}
        - **Device:** {next(trainer.model.parameters()).device}
        """)
    else:
        model_info = mo.md("*No model loaded*")

    return mo.vstack(
        [
            mo.md("## 📈 Training Monitor"),
            progress_display,
            model_info,
        ]
    )


# Helper functions
def _cleanup_memory():
    """Clean up system memory."""
    try:
        force_comprehensive_cleanup()
        mo.output.append(mo.md("✅ Memory cleanup completed!"))
    except Exception as e:
        mo.output.append(mo.md(f"❌ Cleanup error: {str(e)}"))


def _clear_gpu_cache():
    """Clear GPU cache."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mo.output.append(mo.md("✅ GPU cache cleared!"))
        else:
            mo.output.append(mo.md("⚠️ No GPU available"))
    except Exception as e:
        mo.output.append(mo.md(f"❌ Error clearing GPU cache: {str(e)}"))
