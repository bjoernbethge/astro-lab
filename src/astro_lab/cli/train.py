#!/usr/bin/env python3
"""
AstroLab Training CLI (Lightning Edition) - Simplified 2025
=========================================================

CLI for training astronomical ML models using Lightning.
Jetzt minimal, robust und benutzerfreundlich!
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

from astro_lab.models.core import list_presets
from astro_lab.training import train_model

from .config import load_and_prepare_training_config


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for training CLI (minimal version)."""
    parser = argparse.ArgumentParser(
        description="Train astronomical ML models with AstroLab (2025 minimal edition)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python -m src.astro_lab.cli.train --preset graph_classifier_small --dataset gaia
  python -m src.astro_lab.cli.train --preset node_classifier_medium --dataset sdss --epochs 10
  python -m src.astro_lab.cli.train --list-presets
""",
    )
    parser.add_argument(
        "--preset",
        type=str,
        help="Name des Model-Presets (z.B. graph_classifier_small, node_classifier_medium)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gaia",
        choices=["gaia", "sdss", "nsa", "tng50", "exoplanet", "rrlyrae", "linear"],
        help="Datensatz (default: gaia)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Anzahl Trainings-Epochen (optional, überschreibt Preset)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Zeigt alle verfügbaren Presets mit Beschreibung",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    return parser


def print_presets():
    presets = list_presets()
    print("\nVerfügbare Presets:")
    for name, desc in presets.items():
        print(f"  {name:28} {desc}")
    print(
        "\nBeispiel: python -m src.astro_lab.cli.train --preset graph_classifier_small --dataset gaia\n"
    )


def main(args=None) -> int:
    """Main CLI function - minimal, robust, preset-basiert."""
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    logger = setup_logging(args.verbose)

    if hasattr(args, "list_presets") and args.list_presets:
        print_presets()
        return 0

    # Unterstützung für beide CLI-Varianten
    preset = getattr(args, "preset", None)
    model = getattr(args, "model", None)

    # Wenn --model verwendet wurde, verwende es als Preset
    if model and not preset:
        preset = model

    if not preset:
        logger.error(
            "❌ Du musst ein Preset/Model mit --preset oder --model angeben! Beispiel: --preset graph_classifier_small"
        )
        print_presets()
        return 1

    # Hardware defaults
    if torch.cuda.is_available():
        accelerator = "gpu"
        precision = "16-mixed"
        num_workers = getattr(args, "num_workers", 4)
    else:
        accelerator = "cpu"
        precision = "32-true"
        num_workers = 0

    # Preset-Config laden und überschreiben
    cli_overrides = {
        "dataset": getattr(args, "dataset", "gaia"),
        "accelerator": accelerator,
        "precision": precision,
        "num_workers": num_workers,
    }

    # Zusätzliche CLI-Argumente verarbeiten
    if hasattr(args, "epochs") and args.epochs:
        cli_overrides["epochs"] = args.epochs
    if hasattr(args, "batch_size") and args.batch_size:
        cli_overrides["batch_size"] = args.batch_size
    if hasattr(args, "learning_rate") and args.learning_rate:
        cli_overrides["learning_rate"] = args.learning_rate
    if hasattr(args, "max_samples") and args.max_samples:
        cli_overrides["max_samples"] = args.max_samples
    if hasattr(args, "devices") and args.devices:
        cli_overrides["devices"] = args.devices
    if hasattr(args, "precision") and args.precision:
        cli_overrides["precision"] = args.precision
    if hasattr(args, "num_features") and args.num_features:
        cli_overrides["num_features"] = args.num_features

    config = load_and_prepare_training_config(
        preset=preset, cli_overrides=cli_overrides
    )

    logger.info(
        f"Training mit Preset '{preset}' auf Datensatz '{cli_overrides['dataset']}' für {config.get('epochs', 10)} Epochen"
    )
    logger.info(
        f"Hardware: {accelerator}, Precision: {precision}, num_workers: {num_workers}"
    )
    logger.info(f"Alle Parameter: {config}")

    try:
        success = train_model(config)
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Training fehlgeschlagen: {e}")
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
