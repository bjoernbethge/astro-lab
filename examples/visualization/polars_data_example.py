#!/usr/bin/env python3
"""
Polars Data Processing Example
==============================

Demonstrates efficient data handling with Polars DataFrames
for astronomical data processing and preparation for visualization.
"""

import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import numpy as np
import polars as pl
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
        'color': np.random.choice(['red', 'blue', 'yellow', 'white'], n_points),
        'distance': np.linalg.norm(coords, axis=1)
    })
    
    logger.info(f"✅ Created DataFrame with {len(df)} points")
    logger.info(f"📋 Columns: {list(df.columns)}")
    logger.info(f"📈 Schema: {df.schema}")
    
    return df


def demonstrate_polars_operations():
    """Demonstrate various Polars operations for data processing."""
    logger.info("🔧 Demonstrating Polars data operations")
    logger.info("=" * 50)
    
    # Create sample data
    df = create_sample_polars_data()
    
    # 1. Basic statistics
    logger.info("📊 Basic statistics:")
    logger.info(f"  Mean distance: {df['distance'].mean():.2f}")
    logger.info(f"  Distance std: {df['distance'].std():.2f}")
    logger.info(f"  Min magnitude: {df['magnitude'].min():.2f}")
    logger.info(f"  Max magnitude: {df['magnitude'].max():.2f}")
    
    # 2. Filtering
    logger.info("\n🔍 Filtering operations:")
    bright_stars = df.filter(df['magnitude'] < 15.0)
    logger.info(f"  Bright stars (mag < 15): {len(bright_stars)}")
    
    nearby_stars = df.filter(df['distance'] < 5.0)
    logger.info(f"  Nearby stars (dist < 5): {len(nearby_stars)}")
    
    # 3. Grouping and aggregation
    logger.info("\n📈 Grouping operations:")
    color_stats = df.group_by('color').agg([
        pl.count().alias('count'),
        pl.mean('magnitude').alias('avg_magnitude'),
        pl.mean('distance').alias('avg_distance')
    ])
    logger.info(f"  Color statistics: {color_stats}")
    
    # 4. Spatial operations
    logger.info("\n🌍 Spatial operations:")
    center_stars = df.filter(
        (df['x'].abs() < 5) & 
        (df['y'].abs() < 5) & 
        (df['z'].abs() < 5)
    )
    logger.info(f"  Stars near center: {len(center_stars)}")
    
    # 5. Data transformation
    logger.info("\n🔄 Data transformation:")
    df_with_bins = df.with_columns([
        (df['distance'] // 2).alias('distance_bin'),
        (df['magnitude'] < 15).alias('is_bright')
    ])
    logger.info(f"  Added distance bins and brightness flag")
    
    # 6. Sorting
    logger.info("\n📋 Sorting operations:")
    brightest_stars = df.sort('magnitude').head(5)
    logger.info(f"  Top 5 brightest stars: {brightest_stars.select(['id', 'magnitude'])}")
    
    return df


def demonstrate_with_real_data():
    """Demonstrate with real astronomical data."""
    logger.info("🌟 Demonstrating with real Gaia data")
    logger.info("=" * 45)
    
    try:
        # Load Gaia data
        gaia_tensor = load_gaia_data(max_samples=500, return_tensor=True)
        
        # Access underlying Polars DataFrame
        if hasattr(gaia_tensor, 'data'):
            df = gaia_tensor.data
            
            logger.info(f"✅ Loaded {len(df)} Gaia stars")
            logger.info(f"📋 Columns: {list(df.columns)}")
            
            # Demonstrate real data operations
            logger.info("\n🔍 Real data analysis:")
            
            # Filter bright stars
            bright_stars = df.filter(df['phot_g_mean_mag'] < 12.0)
            logger.info(f"  Bright stars (G < 12): {len(bright_stars)}")
            
            # Distance analysis
            if 'distance' in df.columns:
                nearby_stars = df.filter(df['distance'] < 1000)  # Within 1000 pc
                logger.info(f"  Nearby stars (< 1000 pc): {len(nearby_stars)}")
                
                distance_stats = df.select([
                    pl.mean('distance').alias('avg_distance'),
                    pl.std('distance').alias('std_distance'),
                    pl.min('distance').alias('min_distance'),
                    pl.max('distance').alias('max_distance')
                ])
                logger.info(f"  Distance statistics: {distance_stats}")
            
            # Magnitude distribution
            if 'phot_g_mean_mag' in df.columns:
                mag_bins = df.with_columns([
                    (df['phot_g_mean_mag'] // 2).alias('mag_bin')
                ]).group_by('mag_bin').count()
                logger.info(f"  Magnitude distribution: {mag_bins}")
            
            return df
            
    except Exception as e:
        logger.error(f"❌ Failed to load real data: {e}")
        return None


def demonstrate_data_export():
    """Demonstrate data export for visualization."""
    logger.info("💾 Demonstrating data export")
    logger.info("=" * 35)
    
    # Create sample data
    df = create_sample_polars_data()
    
    # Filter for visualization
    viz_data = df.filter(
        (df['magnitude'] < 15.0) & 
        (df['distance'] < 8.0)
    )
    
    logger.info(f"📊 Prepared {len(viz_data)} points for visualization")
    
    # Export to different formats
    try:
        # Export to CSV
        viz_data.write_csv("sample_data.csv")
        logger.info("✅ Exported to CSV: sample_data.csv")
        
        # Export to Parquet
        viz_data.write_parquet("sample_data.parquet")
        logger.info("✅ Exported to Parquet: sample_data.parquet")
        
        # Export to JSON
        viz_data.write_json("sample_data.json")
        logger.info("✅ Exported to JSON: sample_data.json")
        
        # Show data ready for Cosmograph
        logger.info("\n🎨 Data ready for Cosmograph visualization:")
        logger.info(f"  Points: {len(viz_data)}")
        logger.info(f"  Columns: {list(viz_data.columns)}")
        logger.info("  Use bridge.from_polars_dataframe(viz_data) for visualization")
        
    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
    
    return viz_data


def main():
    """Main demonstration workflow."""
    logger.info("🌌 Polars Data Processing Example")
    logger.info("=" * 50)
    
    try:
        # Basic Polars operations
        sample_df = demonstrate_polars_operations()
        
        # Real data demonstration
        real_df = demonstrate_with_real_data()
        
        # Data export demonstration
        viz_data = demonstrate_data_export()
        
        logger.info("\n🎉 All demonstrations completed!")
        logger.info("💡 Polars provides efficient data handling for astronomical data")
        logger.info("🔧 Data is ready for visualization with CosmographBridge")
        logger.info("📊 Export formats support various visualization tools")
        
        # Summary
        logger.info("\n📋 Summary:")
        logger.info(f"   - Sample data: {len(sample_df)} points")
        if real_df is not None:
            logger.info(f"   - Real data: {len(real_df)} Gaia stars")
        logger.info(f"   - Visualization data: {len(viz_data)} points")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main() 