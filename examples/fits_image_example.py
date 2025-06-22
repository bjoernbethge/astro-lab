#!/usr/bin/env python3
"""
FITS Image Processing Example
Demonstrates extracting and visualizing image data from FITS files.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from astro_lab.data.utils import extract_fits_image_data, create_image_tensor_from_fits, visualize_fits_image
from astro_lab.widgets import AstroLabWidget
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate FITS image processing capabilities."""
    print("🔭 FITS Image Processing Example")
    print("=" * 40)
    
    # Check if we have any FITS files
    fits_files = list(Path("data/raw").glob("*.fits"))
    if not fits_files:
        print("❌ No FITS files found in data/raw/")
        print("   Please download some FITS files first.")
        return
    
    print(f"📁 Found {len(fits_files)} FITS files")
    
    # Process first FITS file
    fits_path = fits_files[0]
    print(f"\n🔍 Processing: {fits_path.name}")
    
    try:
        # Extract image data and metadata
        print("\n1️⃣ Extracting image data and metadata...")
        fits_data = extract_fits_image_data(fits_path)
        
        print(f"✅ Image shape: {fits_data['image_data'].shape}")
        print(f"✅ Data type: {fits_data['image_data'].dtype}")
        print(f"✅ Value range: {fits_data['metadata']['min_value']:.2f} to {fits_data['metadata']['max_value']:.2f}")
        
        # Show metadata
        print("\n📋 Metadata:")
        for key, value in fits_data['metadata'].items():
            if key not in ['filename', 'shape', 'dtype']:
                print(f"   {key}: {value}")
        
        # Create tensor
        print("\n2️⃣ Creating image tensor...")
        image_tensor = create_image_tensor_from_fits(fits_path, normalize=True)
        print(f"✅ Tensor shape: {image_tensor.shape}")
        print(f"✅ Tensor range: {image_tensor.min():.3f} to {image_tensor.max():.3f}")
        
        # Visualize with different backends
        print("\n3️⃣ Visualizing with different backends...")
        
        # Matplotlib visualization
        print("   📊 Matplotlib visualization:")
        try:
            fig = visualize_fits_image(fits_path, backend='matplotlib')
            print("   ✅ Matplotlib visualization created")
        except Exception as e:
            print(f"   ❌ Matplotlib failed: {e}")
        
        # Plotly visualization
        print("   🌐 Plotly visualization:")
        try:
            fig = visualize_fits_image(fits_path, backend='plotly')
            print("   ✅ Plotly visualization created")
        except Exception as e:
            print(f"   ❌ Plotly failed: {e}")
        
        # Test with AstroLab Widget
        print("\n4️⃣ Testing with AstroLab Widget...")
        try:
            widget = AstroLabWidget()
            
            # Convert image tensor to point cloud for 3D visualization
            # This is a creative approach - treating image pixels as 3D points
            h, w = image_tensor.shape
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            z = image_tensor.flatten()
            x = x.flatten().float()
            y = y.flatten().float()
            
            # Sample points for performance
            sample_size = min(10000, len(z))
            indices = torch.randperm(len(z))[:sample_size]
            
            coords = torch.stack([x[indices], y[indices], z[indices]], dim=1)
            
            # Create a simple tensor that the widget can handle
            from astro_lab.tensors.factory import TensorFactory
            image_survey_tensor = TensorFactory.create_survey(coords, survey_name='fits_image')
            
            print(f"   ✅ Created image survey tensor with {sample_size} points")
            
            # Visualize with Open3D
            print("   🎨 Open3D visualization:")
            result = widget.plot(image_survey_tensor, plot_type="scatter", backend="open3D")
            print(f"   ✅ Open3D result: {type(result)}")
            
        except Exception as e:
            print(f"   ❌ Widget test failed: {e}")
        
        print("\n🎉 FITS image processing completed!")
        print("\nKey Features:")
        print("- Extract image data and metadata from FITS files")
        print("- Convert to PyTorch tensors")
        print("- Visualize with multiple backends (Matplotlib, Plotly)")
        print("- 3D visualization of image data as point cloud")
        
    except Exception as e:
        print(f"❌ Failed to process FITS file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 