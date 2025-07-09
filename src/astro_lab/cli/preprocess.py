"""
Preprocessing CLI commands for AstroLab.
"""

import logging
from pathlib import Path

import polars as pl
import torch

from astro_lab.config import get_optimal_batch_size

from ..config import get_data_paths

logger = logging.getLogger(__name__)


def preprocess_survey(args) -> int:
    """Preprocess a single survey."""
    survey = args.survey

    print(f"{'=' * 60}")
    print(f"Preprocessing {survey.upper()} data")
    print(f"{'=' * 60}")

    try:
        # Get preprocessor instance
        from astro_lab.data.preprocessors import (
            DESPreprocessor,
            ExoplanetPreprocessor,
            GaiaPreprocessor,
            LINEARPreprocessor,
            NSAPreprocessor,
            RRLyraePreprocessor,
            SDSSPreprocessor,
            TNG50Preprocessor,
            TwoMASSPreprocessor,
            WISEPreprocessor,
        )

        # Map survey names to preprocessor classes
        preprocessor_map = {
            "gaia": GaiaPreprocessor,
            "sdss": SDSSPreprocessor,
            "nsa": NSAPreprocessor,
            "tng50": TNG50Preprocessor,
            "exoplanet": ExoplanetPreprocessor,
            "twomass": TwoMASSPreprocessor,
            "wise": WISEPreprocessor,
            "des": DESPreprocessor,
            "linear": LINEARPreprocessor,
            "rrlyrae": RRLyraePreprocessor,
        }

        if survey not in preprocessor_map:
            raise ValueError(f"Survey '{survey}' not supported")

        preprocessor = preprocessor_map[survey]()

        # Check if data exists
        data_path = preprocessor._find_data_file()

        if not data_path.exists():
            print(f"\n❌ Error: No raw data found for {survey}")
            print(f"Expected location: {data_path}")
            print("\nPlease download the data first:")
            print(f"  astro-lab download {survey}")
            return 1

        print(f"\n📂 Raw data found: {data_path}")
        print(f"📊 File size: {data_path.stat().st_size / (1024**2):.1f} MB")

        # Check if already processed
        processed_dir = Path(get_data_paths()["processed_dir"]) / survey
        processed_file = processed_dir / f"{survey}.parquet"

        # Clean up processed files if --force is set
        if args.force:
            import os

            for ext in ["*.pt", "*.json"]:
                for f in processed_dir.glob(ext):
                    try:
                        os.remove(f)
                    except Exception:
                        pass

        if processed_file.exists() and not args.force:
            print(f"\n✅ Processed data already exists: {processed_file}")
            print("Use --force to reprocess")
            return 0

        # Run preprocessing
        print(f"\n🔄 Processing {survey} data...")
        # Load raw data
        if data_path.suffix == ".parquet":
            raw_df = pl.read_parquet(data_path)
        elif data_path.suffix == ".csv":
            raw_df = pl.read_csv(data_path)
        elif data_path.suffix == ".fits" and survey == "nsa":
            raw_df = preprocessor._load_fits_data(str(data_path))
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        df = preprocessor.preprocess(raw_df)

        # Save harmonized parquet
        processed_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(processed_file)

        # === Build ML-ready dataset (.pt) with sampling ===
        print(
            f"\n🔄 Building ML-ready dataset (.pt) with sampling strategy '{getattr(args, 'sampling_strategy', 'knn')}'..."
        )

        from astro_lab.data.dataset.astrolab import create_dataset

        # Collect sampling parameters
        sampling_kwargs = {}
        if hasattr(args, "k") and args.k is not None:
            sampling_kwargs["k"] = args.k
        if hasattr(args, "radius") and args.radius is not None:
            sampling_kwargs["radius"] = args.radius
        if hasattr(args, "num_subgraphs") and args.num_subgraphs is not None:
            sampling_kwargs["num_subgraphs"] = args.num_subgraphs
        if (
            hasattr(args, "points_per_subgraph")
            and args.points_per_subgraph is not None
        ):
            sampling_kwargs["points_per_subgraph"] = args.points_per_subgraph

        # Batch size and device
        batch_size = getattr(args, "batch_size", None)
        device = getattr(args, "device", None)
        if device is None:
            device = "cpu"
        if batch_size is None:
            # Use automatic detection based on device
            if device.startswith("cuda") and torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                batch_size = get_optimal_batch_size(gpu_memory_gb)
                print(
                    f"   (Auto-detected batch size: {batch_size} for {gpu_memory_gb:.1f} GB GPU, Device: {device})"
                )
            else:
                batch_size = get_optimal_batch_size(None)
                print(
                    f"   (Auto-detected batch size: {batch_size} for CPU, Device: {device})"
                )
        else:
            print(f"   (Batch size: {batch_size}, Device: {device})")

        # Create dataset with selected sampling strategy
        dataset = create_dataset(
            root=processed_dir,
            survey_name=survey,
            data_type=getattr(args, "type", "spatial"),
            sampling_strategy=getattr(args, "sampling_strategy", "knn"),
            sampler_kwargs=sampling_kwargs,
        )
        dataset.process()
        pt_path = dataset.processed_paths[0]
        print(f"\n✅ ML-ready dataset saved to: {pt_path}")
        print(
            f"   (Sampling: {getattr(args, 'sampling_strategy', 'knn')}, Params: {sampling_kwargs})"
        )
        print(f"   (Batch size: {batch_size}, Device: {device})")

        # Show results
        print("\n✅ Processing complete!")
        print(f"📊 Processed {len(df):,} rows")
        print(f"💾 Saved to: {processed_file}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Memory: {df.estimated_size() / (1024 * 1024):.1f} MB")

        return 0

    except Exception as e:
        import traceback

        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("\n📋 Stack trace:")
        traceback.print_exc()
        return 1


# Legacy function for compatibility
def add_preprocess_arguments(parser):
    """Add preprocess arguments to parser (legacy)."""
    parser.add_argument(
        "--surveys",
        nargs="+",
        choices=[
            "gaia",
            "sdss",
            "nsa",
            "tng50",
            "exoplanet",
            "twomass",
            "wise",
            "panstarrs",
            "des",
            "euclid",
        ],
        help="Surveys to preprocess",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to preprocess",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for preprocessed data",
    )


def main(args) -> int:
    """Legacy main function for old CLI structure."""
    if hasattr(args, "surveys") and args.surveys:
        # Process multiple surveys (old style)
        for survey in args.surveys:
            # Create new args object with single survey
            class NewArgs:
                def __init__(self, survey, force=False, max_samples=None):
                    self.survey = survey
                    self.force = force
                    self.max_samples = max_samples
                    self.verbose = getattr(args, "verbose", False)

            new_args = NewArgs(survey, force=False, max_samples=args.max_samples)
            result = preprocess_survey(new_args)

            if result is not None and result != 0:
                return int(result)

        return 0
    else:
        # New style - single survey
        result = preprocess_survey(args)
        return int(result) if result is not None else 0
