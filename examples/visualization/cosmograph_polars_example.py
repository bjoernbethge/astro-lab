#!/usr/bin/env python3
"""
Cosmograph with Polars Example
==============================

Demonstrates how to use CosmographBridge with Polars DataFrames
for interactive 3D visualization of astronomical data.
"""

import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import numpy as np
import polars as pl
from astro_lab.utils.viz import CosmographBridge
from astro_lab.data.core import load_gaia_data, load_sdss_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_polars_data():
    """Create sample Polars DataFrame with 3D coordinates."""
    logger.info("📊 Creating sample Polars DataFrame")
    
    # Generate random 3D coordinates
    n_points = 100
    np.random.seed(42)  # For reproducible results
    
    coords = np.random.randn(n_points, 3) * 10  # 3D coordinates
    
    # Create Polars DataFrame
    df = pl.DataFrame({
        'id': [f'star_{i}' for i in range(n_points)],
        'x': coords[:, 0],
        'y': coords[:, 1],
        'z': coords[:, 2],
        'magnitude': np.random.uniform(10, 20, n_points),
        'color': np.random.choice(['red', 'blue', 'yellow', 'white'], n_points)
    })
    
    logger.info(f"✅ Created DataFrame with {len(df)} points")
    logger.info(f"📋 Columns: {list(df.columns)}")
    logger.info(f"📈 Schema: {df.schema}")
    
    return df


def demonstrate_cosmograph_with_polars():
    """Demonstrate Cosmograph visualization with Polars DataFrame."""
    logger.info("🌌 Demonstrating Cosmograph with Polars")
    logger.info("=" * 50)
    
    # Create sample data
    df = create_sample_polars_data()
    
    # Create CosmographBridge
    bridge = CosmographBridge()
    
    # Create visualization from Polars DataFrame
    logger.info("🎨 Creating Cosmograph visualization...")
    
    try:
        viz = bridge.from_polars_dataframe(
            df=df,
            x_col='x',
            y_col='y',
            z_col='z',
            id_col='id',
            radius=3.0,
            point_color='#ffd700',  # Gold color
            link_color='#666666',   # Gray links
            background_color='#000011',
            show_labels=True,
            show_top_labels_limit=5
        )
        
        logger.info("✅ Cosmograph visualization created successfully!")
        logger.info("🌐 Interactive 3D visualization is ready")
        logger.info("💡 You can rotate, zoom, and explore the data")
        
        return viz
        
    except Exception as e:
        logger.error(f"❌ Failed to create visualization: {e}")
        logger.info("💡 Make sure cosmograph is installed: pip install cosmograph")
        return None


def demonstrate_with_real_data():
    """Demonstrate with real astronomical data."""
    logger.info("🌟 Demonstrating with real Gaia data")
    logger.info("=" * 45)
    
    try:
        # Load Gaia data
        gaia_tensor = load_gaia_data(max_samples=200, return_tensor=True)
        
        # Access underlying Polars DataFrame
        if hasattr(gaia_tensor, 'data'):
            df = gaia_tensor.data
            
            # Create visualization
            bridge = CosmographBridge()
            viz = bridge.from_polars_dataframe(
                df=df,
                x_col='x',
                y_col='y',
                z_col='z',
                radius=5.0,
                point_color='#ffd700',  # Gold for stars
                show_labels=False,      # Too many points for labels
                point_size_range=[1, 4]
            )
            
            logger.info("✅ Real Gaia data visualization created!")
            return viz
            
    except Exception as e:
        logger.error(f"❌ Failed to load real data: {e}")
        return None


def demonstrate_data_operations():
    """Demonstrate Polars data operations before visualization."""
    logger.info("🔧 Demonstrating Polars data operations")
    logger.info("=" * 45)
    
    # Create sample data
    df = create_sample_polars_data()
    
    # Demonstrate filtering
    bright_stars = df.filter(df['magnitude'] < 15.0)
    logger.info(f"✨ Bright stars (mag < 15): {len(bright_stars)}")
    
    # Demonstrate grouping
    color_counts = df.group_by('color').count()
    logger.info(f"🎨 Color distribution: {color_counts}")
    
    # Demonstrate spatial filtering
    center_stars = df.filter(
        (df['x'].abs() < 5) & 
        (df['y'].abs() < 5) & 
        (df['z'].abs() < 5)
    )
    logger.info(f"📍 Stars near center: {len(center_stars)}")
    
    # Create visualization of filtered data
    if len(center_stars) > 0:
        bridge = CosmographBridge()
        viz = bridge.from_polars_dataframe(
            df=center_stars,
            x_col='x',
            y_col='y',
            z_col='z',
            radius=2.0,
            point_color='#00ff00',  # Green for center stars
            show_labels=True
        )
        
        logger.info("✅ Filtered data visualization created!")
        return viz
    
    return None


def main():
    """Main demonstration workflow."""
    logger.info("🌌 Cosmograph with Polars Example")
    logger.info("=" * 50)
    
    try:
        # Basic demonstration
        viz1 = demonstrate_cosmograph_with_polars()
        
        # Data operations demonstration
        viz2 = demonstrate_data_operations()
        
        # Real data demonstration (optional)
        viz3 = demonstrate_with_real_data()
        
        logger.info("🎉 All demonstrations completed!")
        logger.info("💡 Cosmograph visualizations are interactive 3D widgets")
        logger.info("🔧 Polars provides efficient data handling and operations")
        
        if viz1 or viz2 or viz3:
            logger.info("🌐 You can interact with the visualizations in your notebook/IDE")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main() 