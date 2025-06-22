#!/usr/bin/env python3
"""
AstroLab contextlib Memory Management Example
===========================================

Demonstrates the comprehensive contextlib-based memory management features
integrated throughout the AstroLab project.

This example shows:
- Memory tracking and cleanup contexts
- File processing with automatic resource management
- Batch processing with memory optimization
- Tensor operations with memory-efficient contexts
- Training with comprehensive memory management
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from astro_lab.data.manager import AstroDataManager
from astro_lab.data.processing import EnhancedDataProcessor, SimpleProcessingConfig

# Import AstroLab components
from astro_lab.tensors import FeatureTensor, Spatial3DTensor, SurveyTensor

# Import AstroLab memory management utilities
from astro_lab.utils.memory import (
    MemoryMonitor,
    batch_processing_context,
    comprehensive_cleanup_context,
    create_memory_efficient_context,
    file_processing_context,
    memory_tracking_context,
    model_training_context,
    pytorch_memory_context,
)


def example_1_basic_memory_tracking():
    """Example 1: Basic memory tracking with contextlib."""
    print("=" * 60)
    print("Example 1: Basic Memory Tracking")
    print("=" * 60)

    # Basic memory tracking context
    with memory_tracking_context("Basic tensor operations") as stats:
        # Create some tensors
        tensor1 = torch.randn(10000, 100)
        tensor2 = torch.randn(10000, 100)

        # Perform operations
        result = torch.matmul(tensor1, tensor2.T)

        print(
            f"✅ Created tensors: {tensor1.shape} @ {tensor2.T.shape} = {result.shape}"
        )

    # Memory statistics are automatically logged
    print(f"📊 Memory statistics: {stats}")
    print()


def example_2_comprehensive_cleanup():
    """Example 2: Comprehensive cleanup with PyTorch and system resources."""
    print("=" * 60)
    print("Example 2: Comprehensive Cleanup")
    print("=" * 60)

    with comprehensive_cleanup_context(
        "Comprehensive tensor operations",
        cleanup_pytorch=True,
        cleanup_matplotlib=True,
        force_gc=True,
    ) as stats:
        # Create large tensors
        large_tensor = torch.randn(50000, 200)

        # Move to GPU if available
        if torch.cuda.is_available():
            large_tensor = large_tensor.cuda()
            print(f"🚀 Moved tensor to GPU: {large_tensor.device}")

        # Perform memory-intensive operations
        for i in range(5):
            temp_tensor = large_tensor @ large_tensor.T
            print(f"  Iteration {i + 1}: {temp_tensor.shape}")
            # temp_tensor goes out of scope and will be cleaned up

    print(f"📊 Comprehensive cleanup completed: {stats}")
    print()


def example_3_astro_tensor_operations():
    """Example 3: AstroTensor operations with memory management."""
    print("=" * 60)
    print("Example 3: AstroTensor Memory Management")
    print("=" * 60)

    # Create astronomical data
    n_objects = 10000
    positions = np.random.randn(n_objects, 3) * 100  # 3D positions in Mpc

    # Create Spatial3DTensor with memory management
    with comprehensive_cleanup_context("Spatial3DTensor operations"):
        spatial_tensor = Spatial3DTensor(positions, unit="Mpc")

        print(f"📊 Created spatial tensor: {spatial_tensor.shape}")
        print(f"🔧 Memory info: {spatial_tensor.get_memory_info()['memory_mb']:.2f} MB")

        # Use memory-efficient context for operations
        with spatial_tensor.memory_efficient_context("Distance calculations"):
            # Calculate distances from origin
            distances = spatial_tensor.distances_from_origin()
            print(f"📏 Calculated distances: {distances.shape}")

            # Find nearest neighbors
            neighbors = spatial_tensor.find_nearest_neighbors(k=10)
            print(f"🔍 Found neighbors: {neighbors.shape}")

        # Use batch processing context for large operations
        with spatial_tensor.batch_processing_context(batch_size=1000) as batches:
            batch_results = []
            for i, batch in enumerate(batches):
                # Process each batch
                batch_mean = torch.mean(batch, dim=0)
                batch_results.append(batch_mean)
                if i < 3:  # Show first few batches
                    print(f"  Batch {i + 1}: mean = {batch_mean}")

            print(f"📊 Processed {len(batch_results)} batches")

    print()


def example_4_file_processing():
    """Example 4: File processing with memory management."""
    print("=" * 60)
    print("Example 4: File Processing with Memory Management")
    print("=" * 60)

    # Create sample data file
    sample_data = {
        "ra": np.random.uniform(0, 360, 5000),
        "dec": np.random.uniform(-90, 90, 5000),
        "mag_g": np.random.normal(20, 2, 5000),
        "mag_r": np.random.normal(19.5, 2, 5000),
        "redshift": np.random.exponential(0.1, 5000),
    }

    # Save sample data
    import polars as pl

    sample_df = pl.DataFrame(sample_data)
    sample_file = Path("temp_sample_data.parquet")
    sample_df.write_parquet(sample_file)

    try:
        # Process file with memory management
        with file_processing_context(
            file_path=sample_file, memory_limit_mb=500.0
        ) as processing_params:
            print(f"📂 Processing file: {sample_file}")
            print(f"🔧 Processing parameters: {processing_params}")

            # Load and process data
            df = pl.read_parquet(sample_file)

            # Create enhanced processor
            config = SimpleProcessingConfig(
                enable_feature_engineering=True,
                enable_clustering=True,
                memory_limit_mb=500.0,
            )

            processor = EnhancedDataProcessor(config)

            # Process with memory management
            results = processor.process(df)
            print(f"✅ Processing completed: {results['num_objects']} objects")

    finally:
        # Clean up temporary file
        if sample_file.exists():
            sample_file.unlink()
            print(f"🧹 Cleaned up temporary file: {sample_file}")

    print()


def example_5_batch_processing():
    """Example 5: Batch processing multiple files."""
    print("=" * 60)
    print("Example 5: Batch Processing with Memory Management")
    print("=" * 60)

    # Create multiple sample files
    sample_files = []
    for i in range(3):
        # Create sample data
        sample_data = {
            "x": np.random.randn(1000),
            "y": np.random.randn(1000),
            "z": np.random.randn(1000),
            "value": np.random.exponential(1, 1000),
        }

        # Save to file
        sample_df = pl.DataFrame(sample_data)
        sample_file = Path(f"temp_batch_data_{i}.parquet")
        sample_df.write_parquet(sample_file)
        sample_files.append(sample_file)

    try:
        # Batch process with memory management
        with batch_processing_context(
            total_items=len(sample_files), batch_size=2, memory_threshold_mb=300.0
        ) as batch_config:
            print(f"📊 Batch processing {len(sample_files)} files")
            print(f"🔧 Batch configuration: {batch_config}")

            # Process files in batches
            for i, file_path in enumerate(sample_files):
                with comprehensive_cleanup_context(f"File {i + 1}"):
                    # Load and process each file
                    df = pl.read_parquet(file_path)

                    # Convert to tensor
                    tensor_data = torch.from_numpy(df.to_numpy()).float()
                    feature_tensor = FeatureTensor(
                        data=tensor_data, feature_names=df.columns
                    )

                    print(f"  📄 File {i + 1}: {feature_tensor.shape}")

                    # Perform some operations
                    scaled_features = feature_tensor.scale_features(method="standard")
                    print(f"    🔧 Scaled features: {scaled_features.shape}")

    finally:
        # Clean up temporary files
        for sample_file in sample_files:
            if sample_file.exists():
                sample_file.unlink()
        print(f"🧹 Cleaned up {len(sample_files)} temporary files")

    print()


def example_6_model_training_context():
    """Example 6: Model training with memory management."""
    print("=" * 60)
    print("Example 6: Model Training Context")
    print("=" * 60)

    with model_training_context(
        model_name="SimpleTestModel",
        enable_mixed_precision=True,
        gradient_checkpointing=False,
    ) as training_config:
        print(f"🧠 Training configuration: {training_config}")

        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10),
            torch.nn.Softmax(dim=1),
        )

        # Create sample training data
        X = torch.randn(1000, 100)
        y = torch.randint(0, 10, (1000,))

        # Move to optimal device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        X = X.to(device)
        y = y.to(device)

        print(f"🚀 Model and data on device: {device}")

        # Training loop with memory management
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(5):
            with pytorch_memory_context(f"Training epoch {epoch + 1}"):
                # Forward pass
                outputs = model(X)
                loss = criterion(outputs, y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"  Epoch {epoch + 1}: Loss = {loss.item():.4f}")

        print(f"✅ Training completed with memory stats: {training_config['stats']}")

    print()


def example_7_memory_monitoring():
    """Example 7: Continuous memory monitoring."""
    print("=" * 60)
    print("Example 7: Memory Monitoring")
    print("=" * 60)

    # Create memory monitor
    monitor = MemoryMonitor(interval=0.5, threshold_mb=100.0)

    with monitor.monitoring_context():
        print("📊 Memory monitoring started...")

        # Perform memory-intensive operations
        tensors = []
        for i in range(10):
            # Create progressively larger tensors
            size = (i + 1) * 1000
            tensor = torch.randn(size, 100)
            tensors.append(tensor)

            print(f"  Created tensor {i + 1}: {tensor.shape}")

            # Small delay to see memory changes
            import time

            time.sleep(0.1)

        print(f"📊 Created {len(tensors)} tensors")

        # Clear tensors
        tensors.clear()

        print("🧹 Cleared all tensors")

    print(f"📊 Peak memory usage: {monitor.max_memory:.2f} MB")
    print()


def example_8_convenience_function():
    """Example 8: Using convenience functions."""
    print("=" * 60)
    print("Example 8: Convenience Functions")
    print("=" * 60)

    # Use the convenience function for memory-efficient operations
    with create_memory_efficient_context(
        "Convenience example", enable_pytorch=True, enable_monitoring=True
    ) as stats:
        print("🔧 Using convenience context manager")

        # Create some data
        data = torch.randn(5000, 50)

        # Create survey tensor
        survey_tensor = SurveyTensor(
            data=data,
            survey_name="test_survey",
            column_names=[f"col_{i}" for i in range(50)],
        )

        print(f"📊 Created survey tensor: {survey_tensor.shape}")

        # Use tensor's memory-efficient operations
        with survey_tensor.memory_efficient_context("Survey operations"):
            # Perform operations
            subset = survey_tensor.select_objects(slice(0, 1000))
            print(f"📂 Selected subset: {subset.shape}")

            # Calculate statistics
            mean_values = torch.mean(survey_tensor._data, dim=0)
            print(f"📈 Calculated means: {mean_values.shape}")

    print(f"📊 Convenience context completed: {stats}")
    print()


def main():
    """Run all contextlib memory management examples."""
    print("🚀 AstroLab contextlib Memory Management Examples")
    print("=" * 60)
    print()

    # Run all examples
    examples = [
        example_1_basic_memory_tracking,
        example_2_comprehensive_cleanup,
        example_3_astro_tensor_operations,
        example_4_file_processing,
        example_5_batch_processing,
        example_6_model_training_context,
        example_7_memory_monitoring,
        example_8_convenience_function,
    ]

    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
            print()

    print("=" * 60)
    print("✅ All contextlib memory management examples completed!")
    print("=" * 60)

    # Final memory cleanup
    with comprehensive_cleanup_context("Final cleanup"):
        print("🧹 Performing final memory cleanup...")

    print("🎉 Examples finished successfully!")


if __name__ == "__main__":
    main()
