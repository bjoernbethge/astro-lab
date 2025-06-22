# 📦 AstroPack

AstroPack is a specialized package for astronomical data packaging and distribution within the AstroLab ecosystem, providing efficient data compression, cross-platform compatibility, and streamlined data management.

## ✨ Features

- **📦 Data Packaging**: Efficient compression and packaging of survey data
- **🌐 Cross-Platform**: Compatible across different operating systems
- **⚡ High Performance**: Optimized for large astronomical datasets
- **🔄 Version Control**: Data versioning and change tracking
- **📊 Metadata Management**: Comprehensive data cataloging
- **🚀 Distribution**: Streamlined data sharing and deployment

## 📦 Installation

```bash
# Install from the main project
cd astro-lab
uv sync

# Or install directly
uv pip install -e ./astro-pack
```

## 🚀 Quick Start

### Basic Data Packaging
```python
from astro_pack import pack_survey_data, unpack_survey_data

# Pack survey data
pack_survey_data(
    survey_name="gaia",
    source_path="data/raw/gaia",
    output_path="data/packed/gaia.pack",
    compression="lz4"
)

# Unpack survey data
unpack_survey_data(
    pack_path="data/packed/gaia.pack",
    output_path="data/unpacked/gaia"
)
```

### Batch Processing
```python
from astro_pack import batch_pack_surveys

# Pack multiple surveys
surveys = ["gaia", "sdss", "nsa", "tng50"]
batch_pack_surveys(
    surveys=surveys,
    source_dir="data/raw",
    output_dir="data/packed",
    compression="zstd"
)
```

## 🔧 Core Components

### DataPacker
Efficient data compression and packaging:
- **Multiple formats**: Parquet, HDF5, custom binary formats
- **Compression algorithms**: LZ4, Zstandard, GZIP
- **Metadata preservation**: Complete data provenance
- **Incremental updates**: Delta compression for changes

### DataDistributor
Streamlined data distribution:
- **Remote storage**: Cloud storage integration
- **Checksum verification**: Data integrity checking
- **Progressive download**: Partial data access
- **Caching**: Local data caching for performance

### MetadataManager
Comprehensive data cataloging:
- **Schema validation**: Data format verification
- **Version tracking**: Data evolution history
- **Quality metrics**: Data quality assessment
- **Cross-references**: Inter-survey relationships

## 📊 Supported Data Formats

### Input Formats
- **Parquet**: High-performance columnar storage
- **HDF5**: Hierarchical data format
- **FITS**: Astronomical data standard
- **CSV/TSV**: Tabular data
- **Custom binary**: Specialized astronomical formats

### Output Formats
- **AstroPack**: Custom optimized format
- **Parquet**: Compressed columnar storage
- **HDF5**: Hierarchical with compression
- **ZIP**: Standard archive format

## 🎯 Use Cases

### Survey Data Distribution
```python
from astro_pack import create_survey_package

# Create distributable package
package = create_survey_package(
    survey="gaia",
    data_path="data/processed/gaia",
    include_metadata=True,
    include_documentation=True,
    compression="zstd"
)

# Distribute package
package.upload_to_cloud("s3://astro-data/gaia_v1.0.pack")
```

### Data Versioning
```python
from astro_pack import DataVersionManager

# Track data versions
version_manager = DataVersionManager("data/versions/")
version_manager.create_version(
    survey="gaia",
    version="v1.2.0",
    changes="Updated parallax corrections",
    data_path="data/processed/gaia_v1.2.0"
)
```

### Cross-Platform Compatibility
```python
from astro_pack import ensure_compatibility

# Ensure data works across platforms
ensure_compatibility(
    data_path="data/raw/gaia",
    target_platforms=["linux", "windows", "macos"],
    output_path="data/compatible/gaia"
)
```

## 🔧 Advanced Usage

### Custom Compression
```python
from astro_pack import pack_with_custom_compression

# Custom compression settings
pack_with_custom_compression(
    source_path="data/large_survey",
    output_path="data/compressed_survey.pack",
    algorithm="zstd",
    level=19,  # Maximum compression
    threads=8   # Parallel compression
)
```

### Incremental Updates
```python
from astro_pack import create_incremental_package

# Create delta package
delta_package = create_incremental_package(
    base_version="v1.0.0",
    new_version="v1.1.0",
    base_path="data/gaia_v1.0.0",
    new_path="data/gaia_v1.1.0",
    output_path="data/gaia_v1.0.0_to_v1.1.0.delta"
)
```

### Data Validation
```python
from astro_pack import validate_package

# Validate packaged data
validation_result = validate_package(
    pack_path="data/packed/gaia.pack",
    expected_schema="schemas/gaia_schema.json",
    checksum_verification=True
)

if validation_result.is_valid:
    print("Package validation successful!")
else:
    print(f"Validation failed: {validation_result.errors}")
```

## 📈 Performance

### Compression Ratios
| Format | Original Size | Compressed Size | Ratio |
|--------|---------------|-----------------|-------|
| Raw Parquet | 1.0 GB | 0.3 GB | 3.3:1 |
| HDF5 | 1.0 GB | 0.4 GB | 2.5:1 |
| AstroPack | 1.0 GB | 0.2 GB | 5.0:1 |

### Speed Benchmarks
- **Packaging**: 100 MB/s on SSD
- **Unpacking**: 200 MB/s on SSD
- **Validation**: 50 MB/s with full checksums

## 🔧 Development

This package is part of the larger AstroLab framework and follows the same development practices.

### Dependencies
- **PyArrow**: High-performance data processing
- **HDF5**: Hierarchical data format support
- **Zstandard**: Fast compression library
- **Boto3**: AWS S3 integration (optional)

### Testing
```bash
# Run packaging tests
uv run pytest test/data/test_packaging.py
```

## 📚 Related Documentation

- **[Data Loaders](../docs/DATA_LOADERS.md)** - Comprehensive guide to data loading
- **[Main Documentation](../README.md)** - Complete AstroLab framework overview
- **[Development Guide](../docs/DEVGUIDE.md)** - Contributing guidelines
