#!/usr/bin/env python3
"""
AstroLab CLI Preprocessing Commands - DEPRECATED
================================================

⚠️  DEPRECATED: This module is deprecated. Use the unified CLI instead:

    # Old way (deprecated):
    python -m astro_lab.cli.preprocessing preprocess input.parquet --survey gaia

    # New way (recommended):
    uv run python -m astro_lab.cli preprocess input.parquet --config gaia

The new unified CLI provides the same functionality with a cleaner interface.
"""

import sys
import warnings


def main():
    """Show deprecation warning and exit."""
    warnings.warn(
        "astro_lab.cli.preprocessing is deprecated. Use 'uv run python -m astro_lab.cli preprocess' instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    print("⚠️  DEPRECATED: This preprocessing CLI is deprecated!")
    print()
    print("🚀 Use the new unified CLI instead:")
    print()
    print("   # Process all surveys:")
    print("   uv run python -m astro_lab.cli preprocess")
    print()
    print("   # Process specific file:")
    print(
        "   uv run python -m astro_lab.cli preprocess data/catalog.parquet --config gaia"
    )
    print()
    print("   # Show statistics:")
    print(
        "   uv run python -m astro_lab.cli preprocess data/catalog.parquet --stats-only"
    )
    print()
    print("   # Process with custom parameters:")
    print(
        "   uv run python -m astro_lab.cli preprocess --surveys gaia nsa --k-neighbors 8"
    )
    print()
    print("💡 See 'uv run python -m astro_lab.cli preprocess --help' for all options!")

    sys.exit(1)


if __name__ == "__main__":
    main()
