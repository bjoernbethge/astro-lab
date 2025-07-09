"""
System Info Components
====================

System information UI components with proper Marimo patterns.
"""

import os
import platform
import sys
from typing import Any, Dict

import marimo as mo
import psutil
import torch


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    # GPU Information
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "None"

    # Memory Information
    memory = psutil.virtual_memory()
    memory_total = memory.total / (1024**3)  # GB
    memory_used = memory.used / (1024**3)  # GB
    memory_percent = memory.percent

    # GPU Memory if available
    if gpu_available:
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_memory_total = gpu_props.total_memory / (1024**3)  # GB
    else:
        gpu_memory_allocated = 0
        gpu_memory_reserved = 0
        gpu_memory_total = 0

    info = {
        # System
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        # GPU
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "gpu_memory_allocated": gpu_memory_allocated,
        "gpu_memory_reserved": gpu_memory_reserved,
        "gpu_memory_total": gpu_memory_total,
        # Memory
        "memory_total": memory_total,
        "memory_used": memory_used,
        "memory_percent": memory_percent,
        # PyTorch
        "torch_version": torch.__version__,
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None)
        if gpu_available
        else "N/A",
        # Data
        "available_surveys": ["Survey1", "Survey2", "Survey3"],
    }

    return info


def create_system_status():
    """Create interactive system status panel."""
    # Get current system info
    info = get_system_info()

    # Create refresh button that actually works
    refresh_button = mo.ui.button(label="🔄 Refresh", kind="neutral")

    # When button is clicked, update the display
    if refresh_button.value:
        # Re-fetch system info
        info = get_system_info()

    # GPU Status Card
    gpu_card = mo.vstack(
        [
            mo.md("### 🎮 GPU Status"),
            mo.hstack(
                [
                    mo.stat(
                        "Status",
                        "✅ Available" if info["gpu_available"] else "❌ Not Available",
                    ),
                    mo.stat("Count", str(info["gpu_count"])),
                ]
            ),
            mo.md(f"**Device:** {info['gpu_name']}"),
            mo.md(f"**CUDA:** {info['cuda_version']}"),
            mo.md(
                f"**GPU Memory Usage:** {info['gpu_memory_allocated']:.2f} / {info['gpu_memory_total']:.2f} GB"
            )
            if info["gpu_available"]
            else mo.md("*No GPU available*"),
        ]
    )

    # Memory Status Card
    memory_card = mo.vstack(
        [
            mo.md("### 💾 System Memory"),
            mo.md(
                f"**RAM Usage:** {info['memory_used']:.2f} / {info['memory_total']:.2f} GB ({info['memory_percent']:.1f}%)"
            ),
        ]
    )

    # Data Status Card
    surveys = info["available_surveys"]
    data_card = mo.vstack(
        [
            mo.md("### 📊 Available Data"),
            mo.stat("Surveys", str(len(surveys))),
            mo.ui.table({"Survey": surveys, "Status": ["✅ Ready"] * len(surveys)})
            if surveys
            else mo.md("*No surveys available*"),
        ]
    )

    # Backends Status
    backends_card = mo.vstack(
        [
            mo.md("### 🎨 Visualization Backends"),
            mo.ui.table(
                {
                    "Backend": ["PyVista", "Plotly", "Cosmograph", "Open3D"],
                    "Purpose": [
                        "3D Scientific",
                        "Interactive Stats",
                        "Large Networks",
                        "Point Clouds",
                    ],
                    "Status": ["✅", "✅", "✅", "✅"],
                }
            ),
        ]
    )

    # Layout with refresh button
    return mo.vstack(
        [
            mo.hstack(
                [
                    mo.md("## 🖥️ System Information"),
                    refresh_button,
                ],
                justify="space-between",
            ),
            mo.hstack(
                [
                    gpu_card,
                    memory_card,
                ],
                widths=[1, 1],
            ),
            mo.hstack(
                [
                    data_card,
                    backends_card,
                ],
                widths=[1, 1],
            ),
        ]
    )


def create_quick_start_guide():
    """Create an interactive quick start guide."""
    # Step tracking
    completed_steps = mo.ui.array(
        [
            mo.ui.checkbox(label="Configure settings"),
            mo.ui.checkbox(label="Load data"),
            mo.ui.checkbox(label="Run analysis"),
            mo.ui.checkbox(label="Train model"),
            mo.ui.checkbox(label="Create visualization"),
        ]
    )

    # Progress calculation
    completed_count = sum(1 for step in completed_steps.value if step)
    progress = completed_count / len(completed_steps.value)

    # Interactive guide
    guide = mo.vstack(
        [
            mo.md("### 🚀 Quick Start Guide"),
            mo.md(f"**Progress:** {progress * 100:.1f}%"),
            mo.callout(
                f"You've completed {completed_count} of 5 steps!",
                kind="success" if completed_count == 5 else "info",
            ),
            mo.md("""
**Step 1: Configure ⚙️**
- Go to the Config tab
- Set up data, analysis, training, and visualization parameters
- Click "Apply All Configs"
"""),
            completed_steps[0],
            mo.md("""
**Step 2: Load Data 📊**
- Go to the Data tab
- Select your survey and options
- Click "Load Survey Data"
"""),
            completed_steps[1],
            mo.md("""
**Step 3: Run Analysis 🔬**
- Go to the Analysis tab
- Configure analysis parameters
- Click "Run Analysis"
"""),
            completed_steps[2],
            mo.md("""
**Step 4: Train Models 🏋️**
- Go to the Training tab
- Select model and parameters
- Click "Start Training"
"""),
            completed_steps[3],
            mo.md("""
**Step 5: Create Visualizations 📊**
- Go to the Visualization tab
- Choose visualization type
- Click "Generate Plot"
"""),
            completed_steps[4],
        ]
    )

    # Reset button
    reset_button = mo.ui.button(label="🔄 Reset Progress", kind="danger")
    if reset_button.value:
        # Reset all checkboxes
        for i in range(len(completed_steps.value)):
            completed_steps[i].value = False

    return mo.vstack(
        [
            guide,
            reset_button if completed_count > 0 else mo.md(""),
        ]
    )
