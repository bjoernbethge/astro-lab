#!/usr/bin/env python
"""
AstroLab UI Launcher
===================

Launch the AstroLab interactive dashboard.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the AstroLab UI."""
    # Get the app path
    app_path = Path(__file__).parent / "src" / "astro_lab" / "ui" / "app.py"

    if not app_path.exists():
        print(f"Error: App file not found at {app_path}")
        sys.exit(1)

    print("🌌 Starting AstroLab UI...")
    print("=" * 50)
    print("📡 Loading astronomical data processing system")
    print("🧠 Initializing Graph Neural Networks")
    print("🎨 Preparing cosmic web visualizations")
    print("=" * 50)

    # Launch with marimo
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "marimo",
                "edit",
                str(app_path),
                "--port",
                "2718",
                "--watch",
            ]
        )
    except KeyboardInterrupt:
        print("\n👋 AstroLab UI stopped")
    except Exception as e:
        print(f"Error launching UI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║                                                   ║
    ║            🌌 AstroLab UI Dashboard 🌌            ║
    ║                                                   ║
    ║    Astro GNN Laboratory for Cosmic Web Analysis   ║
    ║                                                   ║
    ╚═══════════════════════════════════════════════════╝
    
    Access the dashboard at: http://localhost:2718
    
    Press Ctrl+C to stop the server.
    """)

    main()
