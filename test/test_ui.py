#!/usr/bin/env python3
"""
Test AstroLab UI
===============

Quick test to ensure the UI components work correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import marimo as mo
from src.astro_lab.ui import create_astrolab_dashboard

# Create simple test app
app = mo.App(width="full")

@app.cell
def test_dashboard():
    """Test the main dashboard."""
    print("Testing AstroLab UI...")
    
    # Create dashboard
    dashboard = create_astrolab_dashboard()
    
    print("✅ Dashboard created successfully!")
    return dashboard

@app.cell
def test_components():
    """Test individual components."""
    from src.astro_lab.ui.components import (
        ui_data_explorer,
        ui_visualization_studio,
        ui_analysis_center
    )
    
    print("Testing individual components...")
    
    # Test data explorer
    data_exp = ui_data_explorer()
    print("✅ Data Explorer OK")
    
    # Test visualization studio
    viz_studio = ui_visualization_studio()
    print("✅ Visualization Studio OK")
    
    # Test analysis center
    analysis = ui_analysis_center()
    print("✅ Analysis Center OK")
    
    return mo.vstack([
        mo.md("## Component Tests"),
        mo.accordion({
            "Data Explorer": data_exp,
            "Visualization Studio": viz_studio,
            "Analysis Center": analysis,
        })
    ])

if __name__ == "__main__":
    print("Starting AstroLab UI test...")
    app.run()
