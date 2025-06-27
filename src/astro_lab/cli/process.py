#!/usr/bin/env python3
"""
AstroLab Process CLI - DEPRECATED
================================

This command is deprecated. Please use 'astro-lab preprocess' instead.
"""

import sys


def main(args=None) -> int:
    """Deprecated entry point - redirect to preprocess."""
    print("⚠️  WARNING: 'astro-lab process' is deprecated!")
    print("    Please use 'astro-lab preprocess' instead.")
    print("")
    print("    Example: astro-lab preprocess gaia sdss --max-samples 10000")
    print("")
    return 1
