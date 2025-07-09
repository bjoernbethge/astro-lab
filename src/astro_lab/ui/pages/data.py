"""
Data Page - Uses real COMPONENTS
"""

import marimo as mo

# Import the real, working COMPONENTS
from astro_lab.ui.components.data_loader import create_data_loader


def create_data_page():
    """Data Page - uses real Components"""

    # Use real, working Components
    data_loader_ui, status, preview, loaded_data = create_data_loader()

    return mo.vstack(
        [
            data_loader_ui,
            status,
            preview,
            mo.md("---"),
            mo.md("## 📊 Loaded Data Status"),
            create_data_status_display(loaded_data),
        ]
    )


def create_data_status_display(loaded_data):
    """Status of loaded data"""

    if loaded_data is not None and len(loaded_data) > 0:
        return mo.md(f"""
        ✅ **{len(loaded_data)} objects loaded**
        
        This data is available for:
        - 🌌 Cosmic Web Analysis
        - 🚀 Model Training
        - 🎨 Visualization
        """)
    else:
        return mo.md("📭 **No data loaded**")
