#!/usr/bin/env python3
"""
AstroLab Widget Example with Modular Backend System
Demonstrates the new intelligent backend selection for different data sizes and plot types.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from astro_lab.widgets import AstroLabWidget
from astro_lab.data.core import load_gaia_data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate the new modular backend system."""
    print("🚀 AstroLab Widget with Modular Backend System")
    print("=" * 50)
    
    # Create widget
    widget = AstroLabWidget()
    print("✅ Widget created successfully")
    
    # Load Gaia data
    print("\n📦 Loading Gaia data...")
    try:
        gaia_data = load_gaia_data(max_samples=50000)  # Reduced for faster testing
        print(f"✅ Loaded Gaia data: {type(gaia_data)}")
    except Exception as e:
        print(f"❌ Failed to load Gaia data: {e}")
        return
    
    # Test different backends and plot types
    print("\n🎨 Testing different backends and plot types:")
    
    # 1. Auto-select backend (should choose Open3D for large data)
    print("\n1️⃣ Auto-selecting backend (should choose Open3D):")
    try:
        result = widget.plot(gaia_data, plot_type="scatter")
        print(f"✅ Auto-backend result: {type(result)}")
    except Exception as e:
        print(f"❌ Auto-backend failed: {e}")
    
    # 2. Force Open3D backend
    print("\n2️⃣ Forcing Open3D backend:")
    try:
        result = widget.plot(gaia_data, plot_type="scatter", backend="open3d")
        print(f"✅ Open3D result: {type(result)}")
    except Exception as e:
        print(f"❌ Open3D failed: {e}")
    
    # 3. Force PyVista backend (with downsampling)
    print("\n3️⃣ Forcing PyVista backend:")
    try:
        result = widget.plot(gaia_data, plot_type="scatter", backend="pyvista", max_points=5000)
        print(f"✅ PyVista result: {type(result)}")
    except Exception as e:
        print(f"❌ PyVista failed: {e}")
    
    # 4. Force Plotly backend (for web visualization)
    print("\n4️⃣ Forcing Plotly backend:")
    try:
        result = widget.plot(gaia_data, plot_type="scatter", backend="plotly", max_points=1000)
        print(f"✅ Plotly result: {type(result)}")
    except Exception as e:
        print(f"❌ Plotly failed: {e}")
    
    # 5. Test Blender backend (if available)
    print("\n5️⃣ Testing Blender backend:")
    try:
        result = widget.plot(gaia_data, plot_type="mesh", backend="bpy", max_points=1000)
        print(f"✅ Blender result: {type(result)}")
    except Exception as e:
        print(f"❌ Blender failed: {e}")
    
    print("\n🎉 Backend system test completed!")
    print("\nKey Features:")
    print("- Intelligent backend selection based on data size")
    print("- Automatic downsampling for large datasets")
    print("- Support for Open3D, PyVista, Plotly, and Blender")
    print("- Single API for all visualization backends")


if __name__ == "__main__":
    main() 