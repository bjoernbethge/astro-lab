#!/usr/bin/env python3
"""
Run AstroLab UI Dashboard
========================

Start the modern AstroLab UI dashboard using Marimo.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import marimo as mo
from src.astro_lab.ui.app import app

if __name__ == "__main__":
    print("🌟 Starting AstroLab Dashboard...")
    print("📍 Open your browser at http://localhost:2718")
    print("⚡ Press Ctrl+C to stop")
    
    # Run the Marimo app
    app.run()
