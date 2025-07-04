"""
AstroLab UI Package - Reactive State Architecture
===============================================

Marimo-based user interface for astronomical data analysis.
Reactive architecture with central state management.

The UI provides:
- Central state management for all UI components
- Real data loading from AstroLab preprocessors
- Real analysis using CosmicWebAnalyzer, SpatialClustering, etc.
- Real visualizations using AstroLab widgets
- Interactive exploration with reactive data flow
"""

# Import central state
from .components import state

# Import dashboard
from .dashboard import create_astrolab_dashboard

# Version info
__version__ = "3.0.0"
__author__ = "Bjoern Bethge"

# Main exports
__all__ = [
    # Dashboard
    "create_astrolab_dashboard",
    # Central State
    "state",
]

# Package documentation
if __doc__ is None:
    __doc__ = ""
__doc__ += f"""

AstroLab UI v{__version__} - Reactive State Architecture
=====================================================

This UI package provides a complete web-based interface for AstroLab's
astronomical data analysis capabilities. It's built with Marimo for
interactive data exploration and visualization.

Architecture:
- state.py: Central state management for all UI elements
- components/: Reusable UI components
- pages/: Complete pages using central state
- dashboard.py: Main dashboard assembly

Key Features:
- Reactive state management with Marimo
- Real backend integration with AstroLab functions
- Interactive data loading and preprocessing
- Real-time analysis with Cosmic Web detection
- Visualization with multiple backends
- System monitoring and status
- Data flows automatically: loading → analysis → visualization

For usage examples, see the dashboard module and individual pages.
"""
