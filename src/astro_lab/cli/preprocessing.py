#!/usr/bin/env python3
"""
AstroLab CLI Preprocessing Commands
==================================

CLI commands for data preprocessing. Only handles argument parsing and command execution.
All actual preprocessing logic is in the data module.
"""

import logging
from pathlib import Path
from typing import Optional

import click
import polars as pl

from ..data.preprocessing import preprocess_catalog, create_graph_from_dataframe

logger = logging.getLogger(__name__)


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--survey", "-s", required=True, help="Survey type (gaia, sdss, nsa, linear)")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--max-samples", "-n", type=int, help="Maximum number of samples")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def preprocess(input_path: str, survey: str, output_dir: Optional[str], max_samples: Optional[int], verbose: bool):
    """
    Preprocess astronomical catalog data.
    
    INPUT_PATH: Path to input catalog file (.parquet, .csv)
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    logger.info(f"🔄 Starting preprocessing for {survey} catalog")
    
    try:
        # Call data module preprocessing function
        df = preprocess_catalog(
            input_path=input_path,
            survey_type=survey,
            max_samples=max_samples,
            output_dir=output_dir
        )
        
        logger.info(f"✅ Preprocessing completed: {len(df)} objects processed")
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        raise click.ClickException(str(e))


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--survey", "-s", required=True, help="Survey type (gaia, sdss, nsa, linear)")
@click.option("--output-path", "-o", type=click.Path(), help="Output graph file path")
@click.option("--k-neighbors", "-k", default=8, help="Number of nearest neighbors")
@click.option("--distance-threshold", "-d", default=50.0, help="Distance threshold for edges")
@click.option("--max-samples", "-n", type=int, help="Maximum number of samples")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def create_graph(input_path: str, survey: str, output_path: Optional[str], k_neighbors: int, 
                distance_threshold: float, max_samples: Optional[int], verbose: bool):
    """
    Create PyTorch Geometric graph from astronomical data.
    
    INPUT_PATH: Path to input data file (.parquet, .csv)
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    logger.info(f"🔄 Creating graph for {survey} data")
    
    try:
        # Load data first
        input_path_obj = Path(input_path)
        
        if input_path_obj.suffix == ".parquet":
            df = pl.read_parquet(input_path)
        elif input_path_obj.suffix == ".csv":
            df = pl.read_csv(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path_obj.suffix}")
        
        # Sample if requested
        if max_samples and len(df) > max_samples:
            df = df.sample(max_samples, seed=42)
            logger.info(f"📊 Sampled {max_samples} objects")
        
        # Determine output path
        if output_path is None:
            output_path = f"data/processed/{survey}/processed/{survey}_k{k_neighbors}.pt"
        
        # Call data module graph creation function
        graph_data = create_graph_from_dataframe(
            df=df,
            survey_type=survey,
            k_neighbors=k_neighbors,
            distance_threshold=distance_threshold,
            output_path=Path(output_path)
        )
        
        if graph_data:
            logger.info(f"✅ Graph created: {graph_data.num_nodes} nodes, {graph_data.edge_index.shape[1]} edges")
        else:
            logger.warning("⚠️ No graph data created")
            
    except Exception as e:
        logger.error(f"❌ Graph creation failed: {e}")
        raise click.ClickException(str(e))


@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--survey", "-s", required=True, help="Survey type (gaia, sdss, nsa, linear)")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--k-neighbors", "-k", default=8, help="Number of nearest neighbors")
@click.option("--distance-threshold", "-d", default=50.0, help="Distance threshold for edges")
@click.option("--max-samples", "-n", type=int, help="Maximum number of samples")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def process_all(input_dir: str, survey: str, output_dir: Optional[str], k_neighbors: int,
               distance_threshold: float, max_samples: Optional[int], verbose: bool):
    """
    Process all files in a directory.
    
    INPUT_DIR: Directory containing data files
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    logger.info(f"🔄 Processing all files in {input_dir}")
    
    try:
        input_path = Path(input_dir)
        data_files = list(input_path.glob("*.parquet")) + list(input_path.glob("*.csv"))
        
        if not data_files:
            raise ValueError(f"No data files found in {input_dir}")
        
        logger.info(f"📁 Found {len(data_files)} data files")
        
        for file_path in data_files:
            logger.info(f"🔄 Processing {file_path.name}")
            
            # Preprocess
            df = preprocess_catalog(
                input_path=str(file_path),
                survey_type=survey,
                max_samples=max_samples,
                output_dir=output_dir
            )
            
            # Create graph
            if output_dir:
                graph_path = Path(output_dir) / f"{file_path.stem}_k{k_neighbors}.pt"
            else:
                graph_path = Path(f"data/processed/{survey}/processed/{file_path.stem}_k{k_neighbors}.pt")
            
            graph_data = create_graph_from_dataframe(
                df=df,
                survey_type=survey,
                k_neighbors=k_neighbors,
                distance_threshold=distance_threshold,
                output_path=graph_path
            )
            
            if graph_data:
                logger.info(f"✅ Created graph: {graph_data.num_nodes} nodes")
        
        logger.info(f"✅ Processed {len(data_files)} files")
        
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        raise click.ClickException(str(e))
