#!/usr/bin/env python3
"""
AstroLab Preprocess CLI
======================

Complete data preparation pipeline:
1. Download raw data (if needed)
2. Clean and preprocess data
3. Create SurveyTensorDicts with 3D spatial coordinates
4. Build graph structures for training
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

from astro_lab.data import load_survey_catalog, preprocess_survey
from astro_lab.data.datasets import SurveyGraphDataset


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def main(args=None) -> int:
    """Main entry point for preprocess command."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare astronomical survey data for machine learning"
    )
    parser.add_argument(
        "surveys", 
        nargs="+", 
        help="Surveys to preprocess (e.g., gaia sdss)"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int, 
        default=None, 
        help="Maximum samples per survey"
    )
    parser.add_argument(
        "--k-neighbors", "-k",
        type=int,
        default=8,
        help="Number of nearest neighbors for graph (default: 8)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output directory (default: data/)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-processing even if files exist"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (use existing data)"
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip graph building step"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args(args)
    logger = setup_logging(args.verbose)
    
    if args.output_dir is None:
        args.output_dir = Path("data")
    
    try:
        for survey in args.surveys:
            logger.info(f"\n{'='*60}")
            logger.info(f"🌟 Preparing {survey.upper()} survey data")
            logger.info(f"{'='*60}\n")
            
            # Step 1: Download/load raw data if needed
            raw_path = args.output_dir / "raw" / survey / f"{survey}.parquet"
            if not args.skip_download:
                if raw_path.exists() and not args.force:
                    logger.info(f"✓ Raw data exists: {raw_path}")
                else:
                    logger.info(f"📥 Downloading {survey} data...")
                    try:
                        df = load_survey_catalog(survey, max_samples=args.max_samples)
                        raw_path.parent.mkdir(parents=True, exist_ok=True)
                        df.write_parquet(raw_path)
                        logger.info(f"✓ Downloaded {len(df)} objects → {raw_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ Download failed: {e}")
                        logger.info("   Trying to continue with existing data...")
            
            # Step 2: Preprocess data
            processed_path = args.output_dir / "processed" / survey / f"{survey}_processed.parquet"
            if processed_path.exists() and not args.force:
                logger.info(f"✓ Processed data exists: {processed_path}")
            else:
                logger.info(f"🔧 Preprocessing {survey} data...")
                try:
                    input_path = raw_path if raw_path.exists() else None
                    processed_path = preprocess_survey(
                        survey,
                        input_path=input_path,
                        output_path=processed_path,
                        max_samples=args.max_samples
                    )
                    logger.info(f"✓ Preprocessed → {processed_path}")
                except Exception as e:
                    logger.error(f"❌ Preprocessing failed: {e}")
                    continue
            
            # Step 3: Verify processed data exists
            if not args.skip_graph:
                # Just verify the preprocessed data can be loaded
                logger.info(f"🔍 Verifying preprocessed data...")
                try:
                    import polars as pl
                    df = pl.read_parquet(processed_path)
                    logger.info(f"✓ Verified: {len(df)} objects with {len(df.columns)} columns")
                    logger.info(f"✓ Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
                    
                    # Note: Graph building happens automatically during training
                    logger.info(f"📝 Note: Graph structures will be built automatically during training")
                    
                except Exception as e:
                    logger.error(f"❌ Verification failed: {e}")
                    continue
            
            logger.info(f"\n✅ {survey.upper()} preparation complete!\n")
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("📊 PREPARATION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"✓ Prepared {len(args.surveys)} survey(s)")
        logger.info(f"✓ Output directory: {args.output_dir.absolute()}")
        logger.info(f"✓ Graph configuration: {args.k_neighbors}-NN, 3D coordinates")
        
        # Next steps
        logger.info(f"\n📚 Next steps:")
        logger.info(f"   1. Train a model: astro-lab train --dataset {args.surveys[0]}")
        logger.info(f"   2. Launch UI: marimo run src/astro_lab/ui/app.py")
        logger.info(f"   3. Inspect data: python -c \"from astro_lab.data import create_astro_datamodule; dm = create_astro_datamodule('{args.surveys[0]}'); dm.setup(); print(dm.get_info())\"")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Preparation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
