#!/usr/bin/env python3
"""
AstroLab CLI - Main Entry Point
"""

# Suppress NumPy warnings before ANY imports
import os
import warnings
warnings.filterwarnings("ignore", message=".*NumPy 1.x.*")
warnings.filterwarnings("ignore", message=".*compiled using NumPy 1.x.*") 
warnings.filterwarnings("ignore", message=".*numpy.core.multiarray.*")
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:numpy'

from . import main

if __name__ == "__main__":
    main() 