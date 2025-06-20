"""
AstroLab Operators Demo - Blender Integration Example
====================================================

Demonstrates the use of AstroLab operators for astronomical visualization
in Blender with procedural galaxy and nebula generation.
"""

import numpy as np
import torch
import polars as pl
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

from astro_lab.widgets.astro_lab import AstroLabWidget


def run_blender_demo(widget: AstroLabWidget):
    """
    Runs the main demonstration of the AstroLab Blender API.
    """
    print("\n--- Running Blender API Demo ---")

    # 1. Reset the scene using the core API
    print("Step 1: Resetting the scene.")
    widget.al.core['reset_scene']()

    # 2. Create a procedural galaxy using the advanced API
    print("Step 2: Creating a procedural spiral galaxy.")
    widget.al.advanced.create_procedural_galaxy(
        galaxy_type='spiral',
        num_stars=100000,
        radius=25
    )

    # 3. Set up a camera and light
    print("Step 3: Setting up camera and lighting.")
    widget.al.core['create_camera'](location=(0, -60, 20), target=(0, 0, 0))
    widget.al.core['create_light'](light_type='SUN', energy=5)

    # 4. Use the 3D plotter to draw a sine wave around the galaxy
    print("Step 4: Plotting a 3D sine wave with the Grease Pencil plotter.")
    t = np.linspace(0, 20, 500)
    x = np.cos(t) * 30
    y = np.sin(t) * 30
    z = np.sin(t * 5) * 5
    points = np.vstack((x, y, z)).T
    widget.al.plot_3d.create_3d_line_plot(points, name="SineWavePath", color=(1.0, 0.2, 0.8, 1.0), width=5.0)

    # 5. Access low-level bpy.ops to select all and smooth the sine wave
    print("Step 5: Accessing low-level bpy.ops to smooth the plot.")
    try:
        # De-select all first
        widget.ops.object.select_all(action='DESELECT')
        # Select our sine wave object
        if 'SineWavePath' in widget.data.objects:
            widget.data.objects['SineWavePath'].select_set(True)
            widget.context.view_layer.objects.active = widget.data.objects['SineWavePath'] # Failsafe
            # Switch to edit mode, select all, and smooth
            widget.ops.object.mode_set(mode='EDIT')
            widget.ops.gpencil.select_all(action='SELECT')
            widget.ops.gpencil.stroke_smooth(factor=0.8, repeat=5)
            widget.ops.object.mode_set(mode='OBJECT')
            print("   - Successfully smoothed the Grease Pencil object.")
        else:
            print("   - Could not find 'SineWavePath' to smooth.")
    except Exception as e:
        print(f"   - Could not execute low-level command: {e}")


    print("\n✅ Blender Demo Finished. Check your running Blender instance!")


if __name__ == "__main__":
    print("Initializing AstroLabWidget...")
    # Initialize with simulated data
    widget = AstroLabWidget()

    # Check if the Blender API is connected
    if widget.blender_available():
        run_blender_demo(widget)
    else:
        print("\n❌ Blender API not available.")
        print("Please ensure Blender is installed and the 'bpy' module is in your Python path.")
        print("You can do this by installing the 'bpy' pip package or running this script from within Blender.") 