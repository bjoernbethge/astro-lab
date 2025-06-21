#!/usr/bin/env python3
"""
🌌 Live Tensor Demo

Demonstrates real-time tensor operations and visualization
with live updates and interactive controls.
"""

import numpy as np
import time
import threading
from typing import Optional, Dict, Any

# Import AstroLab components
from astro_lab.widgets.astro_lab import AstroLabWidget
from astro_lab.tensors.spatial_3d import Spatial3DTensor

def demo_live_tensor_updates():
    """Demonstrates live tensor updates with real-time visualization."""
    print("🔄 Live Tensor Updates Demo")
    print("=" * 35)
    
    # Create widget
    widget = AstroLabWidget(num_galaxies=1000)
    
    # Create initial visualization
    plotter = widget.create_visualization()
    
    print("✅ Initial visualization created")
    print("🔄 Starting live updates...")
    
    # Simulate live data updates
    def update_loop():
        for i in range(10):
            # Generate new data
            new_data = np.random.randn(1000, 3) * (1 + i * 0.1)
            
            # Update tensor
            if hasattr(widget, 'galaxy_df'):
                # Update Polars DataFrame
                widget.galaxy_df = widget.galaxy_df.with_columns([
                    (pl.col("ra") + np.random.normal(0, 0.1, len(widget.galaxy_df))).alias("ra"),
                    (pl.col("dec") + np.random.normal(0, 0.1, len(widget.galaxy_df))).alias("dec"),
                    (pl.col("redshift") + np.random.normal(0, 0.01, len(widget.galaxy_df))).alias("redshift")
                ])
            
            print(f"📊 Update {i+1}/10: {len(new_data)} points")
            time.sleep(2)
    
    # Start update thread
    update_thread = threading.Thread(target=update_loop)
    update_thread.daemon = True
    update_thread.start()
    
    # Show visualization
    plotter.show()
    
    return widget

def demo_interactive_tensor_manipulation():
    """Demonstrates interactive tensor manipulation."""
    print("\n🎛️ Interactive Tensor Manipulation")
    print("=" * 40)
    
    widget = AstroLabWidget(num_galaxies=500)
    
    # Add interactive controls
    widget.add_interactive_controls()
    
    # Create visualization with controls
    plotter = widget.create_visualization()
    
    print("✅ Interactive controls added")
    print("   - Use sliders to adjust visualization")
    print("   - Real-time tensor updates")
    
    # Show interactive visualization
    plotter.show()
    
    return widget

def demo_tensor_analysis():
    """Demonstrates tensor analysis with live updates."""
    print("\n📈 Tensor Analysis Demo")
    print("=" * 25)
    
    widget = AstroLabWidget(num_galaxies=2000)
    
    # Initial analysis
    print("📊 Initial analysis:")
    widget.analyze_data()
    
    # Simulate progressive analysis
    def analysis_loop():
        for i in range(5):
            time.sleep(3)
            print(f"\n📊 Analysis update {i+1}/5:")
            
            # Add more data
            additional_data = np.random.randn(500, 3)
            # Update analysis
            widget.analyze_data()
    
    # Start analysis thread
    analysis_thread = threading.Thread(target=analysis_loop)
    analysis_thread.daemon = True
    analysis_thread.start()
    
    # Create visualization
    plotter = widget.create_visualization()
    plotter.show()
    
    return widget

if __name__ == "__main__":
    print("🌌 Live Tensor Demo Suite")
    print("=" * 40)
    
    # Run demos
    try:
        live_widget = demo_live_tensor_updates()
        print("✅ Live tensor updates demo completed")
    except Exception as e:
        print(f"❌ Live tensor updates demo failed: {e}")
    
    try:
        interactive_widget = demo_interactive_tensor_manipulation()
        print("✅ Interactive tensor manipulation demo completed")
    except Exception as e:
        print(f"❌ Interactive tensor manipulation demo failed: {e}")
    
    try:
        analysis_widget = demo_tensor_analysis()
        print("✅ Tensor analysis demo completed")
    except Exception as e:
        print(f"❌ Tensor analysis demo failed: {e}")
    
    print("\n🎉 All live tensor demos completed!")
    print("\n💡 Key Features:")
    print("   - Real-time tensor updates")
    print("   - Interactive controls")
    print("   - Live data analysis")
    print("   - Threaded operations") 