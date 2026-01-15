"""
Cross-Backend Workflow Example
=============================

Demonstrates advanced cross-backend workflows with:
- Image processing and texture generation
- PhotometricTensorDict and ImageTensorDict support
- Post-processing effects
- Cross-backend data transfer
"""

import torch

from astro_lab.tensors import (
    ImageTensorDict,
    LightcurveTensorDict,
    PhotometricTensorDict,
    SpatialTensorDict,
)
from astro_lab.widgets import (
    create_astronomical_texture,
    create_cross_backend_workflow,
    cross_bridge,
    quick_2d_plot,
    quick_3d_plot,
    quick_texture,
)


def create_sample_data():
    """Create sample TensorDicts for demonstration."""

    # Sample spatial data
    n_points = 1000
    coordinates = torch.randn(n_points, 3) * 100
    spatial_data = SpatialTensorDict(
        coordinates=coordinates,
        masses=torch.rand(n_points),
        velocities=torch.randn(n_points, 3),
    )

    # Sample photometric data
    n_stars = 500
    magnitudes = torch.randn(n_stars, 5)  # 5 bands
    fluxes = torch.exp(-magnitudes / 2.5)
    photometric_data = PhotometricTensorDict(
        magnitudes=magnitudes,
        fluxes=fluxes,
        positions=torch.randn(n_stars, 2),
        bands=["u", "g", "r", "i", "z"],
    )

    # Sample image data
    n_images = 10
    images = torch.randn(n_images, 3, 256, 256)  # RGB images
    image_data = ImageTensorDict(images=images, bands=["R", "G", "B"])

    # Sample lightcurve data
    n_objects = 50
    n_times = 100
    times = torch.linspace(0, 10, n_times)
    magnitudes_lc = torch.randn(n_objects, n_times, 1) + 15  # Base magnitude 15
    lightcurve_data = LightcurveTensorDict(
        times=times.unsqueeze(0).expand(n_objects, -1),
        magnitudes=magnitudes_lc,
        bands=["V"],
    )

    return spatial_data, photometric_data, image_data, lightcurve_data


def demonstrate_image_processing():
    """Demonstrate image processing capabilities."""
    print("=== Image Processing Demonstration ===")

    spatial_data, photometric_data, image_data, lightcurve_data = create_sample_data()

    # Process spatial data to density heatmap
    print("1. Creating density heatmap from spatial data...")
    density_img = cross_bridge.process_tensor_to_image(
        spatial_data,
        processing_type="density_heatmap",
        image_size=(512, 512),
        colormap="viridis",
    )
    print(f"   Created density heatmap: {density_img.image.shape}")

    # Process photometric data to HR diagram
    print("2. Creating HR diagram from photometric data...")
    hr_img = cross_bridge.process_tensor_to_image(
        photometric_data,
        processing_type="photometric",
        image_size=(512, 512),
        plot_type="hr_diagram",
        point_size=3,
    )
    print(f"   Created HR diagram: {hr_img.image.shape}")

    # Process image data
    print("3. Processing image data...")
    processed_img = cross_bridge.process_tensor_to_image(
        image_data, processing_type="image", image_size=(256, 256)
    )
    print(f"   Processed image: {processed_img.image.shape}")

    # Process lightcurve data
    print("4. Creating lightcurve plot...")
    lc_img = cross_bridge.process_tensor_to_image(
        lightcurve_data,
        processing_type="lightcurve",
        image_size=(512, 256),
        color=[255, 0, 0],
        thickness=2,
    )
    print(f"   Created lightcurve plot: {lc_img.image.shape}")

    return density_img, hr_img, processed_img, lc_img


def demonstrate_post_processing():
    """Demonstrate post-processing effects."""
    print("\n=== Post-Processing Demonstration ===")

    # Create a simple image for post-processing
    spatial_data, _, _, _ = create_sample_data()

    # Create base image
    base_img = cross_bridge.process_tensor_to_image(
        spatial_data, processing_type="density_heatmap", image_size=(256, 256)
    )

    # Apply different post-processing effects
    effects = [
        ("blur", {"blur_kernel": 5}),
        ("sharpen", {}),
        ("noise", {"noise_level": 0.1}),
        ("glow", {}),
        ("edge_detect", {}),
    ]

    processed_images = {}

    for effect_name, effect_params in effects:
        print(f"Applying {effect_name} effect...")
        processed_img = cross_bridge.apply_post_processing(
            base_img, [effect_name], **effect_params
        )
        processed_images[effect_name] = processed_img
        print(f"   Applied {effect_name}: {processed_img.image.shape}")

    return processed_images


def demonstrate_texture_generation():
    """Demonstrate texture generation from different data types."""
    print("\n=== Texture Generation Demonstration ===")

    spatial_data, photometric_data, image_data, lightcurve_data = create_sample_data()

    # Create different types of textures
    texture_configs = [
        ("spatial_diffuse", spatial_data, "diffuse", "density_heatmap"),
        ("spatial_emission", spatial_data, "emission", "density_heatmap"),
        ("photometric_normal", photometric_data, "normal", "photometric"),
        ("image_roughness", image_data, "roughness", "image"),
        ("lightcurve_emission", lightcurve_data, "emission", "lightcurve"),
    ]

    textures = {}

    for name, data, texture_type, processing_type in texture_configs:
        print(f"Creating {name} texture...")

        # Create texture with post-processing
        texture = create_astronomical_texture(
            data,
            processing_type=processing_type,
            texture_type=texture_type,
            post_processing=["blur", "glow"] if texture_type == "emission" else None,
            tileable=True,
        )

        textures[name] = texture
        print(
            f"   Created {name}: {texture.texture.shape}, type: {texture.texture_type}"
        )

    return textures


def demonstrate_cross_backend_workflow():
    """Demonstrate complete cross-backend workflow."""
    print("\n=== Cross-Backend Workflow Demonstration ===")

    spatial_data, _, _, _ = create_sample_data()

    # Define workflow steps
    workflow_steps = [
        {
            "step": "initial_processing",
            "backend": "pyvista",
            "processing": "density_heatmap",
            "post_processing": ["blur", "glow"],
            "output": "processed_image",
        },
        {
            "step": "texture_generation",
            "backend": "blender",
            "texture_type": "emission",
            "output": "texture",
        },
        {
            "step": "post_processing",
            "backend": "open3d",
            "effects": ["remove_outliers", "estimate_normals"],
            "output": "cleaned_data",
        },
        {
            "step": "final_export",
            "backend": "blender",
            "conversion": "mesh_to_scene",
            "output": "final_scene",
        },
    ]

    print("Executing cross-backend workflow...")
    results = create_cross_backend_workflow(spatial_data, workflow_steps)

    for step_name, step_result in results.items():
        print(f"   {step_name}: {list(step_result.keys())}")

    return results


def demonstrate_quick_functions():
    """Demonstrate quick access functions."""
    print("\n=== Quick Functions Demonstration ===")

    spatial_data, photometric_data, image_data, lightcurve_data = create_sample_data()

    # Quick 3D plot
    print("1. Quick 3D plot...")
    try:
        quick_3d_plot(spatial_data)
        print("   ✓ 3D visualization created")
    except Exception as e:
        print(f"   ✗ 3D visualization failed: {e}")

    # Quick 2D plot
    print("2. Quick 2D plot...")
    try:
        quick_2d_plot(photometric_data)
        print("   ✓ 2D visualization created")
    except Exception as e:
        print(f"   ✗ 2D visualization failed: {e}")

    # Quick texture
    print("3. Quick texture generation...")
    try:
        texture = quick_texture(spatial_data, texture_type="emission")
        print(f"   ✓ Texture created: {texture.texture.shape}")
    except Exception as e:
        print(f"   ✗ Texture generation failed: {e}")


def demonstrate_photometric_handling():
    """Demonstrate specific handling of PhotometricTensorDict."""
    print("\n=== PhotometricTensorDict Handling ===")

    _, photometric_data, _, _ = create_sample_data()

    # Different photometric visualizations
    photometric_configs = [
        ("hr_diagram", {"plot_type": "hr_diagram"}),
        ("color_color", {"plot_type": "color_color"}),
        ("magnitude_distribution", {"plot_type": "magnitude_dist"}),
    ]

    for plot_type, config in photometric_configs:
        print(f"Creating {plot_type}...")
        try:
            img = cross_bridge.process_tensor_to_image(
                photometric_data,
                processing_type="photometric",
                image_size=(512, 512),
                **config,
            )
            print(f"   ✓ {plot_type}: {img.image.shape}")
        except Exception as e:
            print(f"   ✗ {plot_type} failed: {e}")


def demonstrate_image_tensor_handling():
    """Demonstrate specific handling of ImageTensorDict."""
    print("\n=== ImageTensorDict Handling ===")

    _, _, image_data, _ = create_sample_data()

    # Different image processing approaches
    image_configs = [
        ("rgb_composite", {"composite_method": "rgb"}),
        ("false_color", {"composite_method": "false_color"}),
        ("enhanced", {"enhancement": "histogram_equalization"}),
    ]

    for method, config in image_configs:
        print(f"Processing image with {method}...")
        try:
            img = cross_bridge.process_tensor_to_image(
                image_data, processing_type="image", image_size=(256, 256), **config
            )
            print(f"   ✓ {method}: {img.image.shape}")
        except Exception as e:
            print(f"   ✗ {method} failed: {e}")


def main():
    """Run all demonstrations."""
    print("AstroLab Cross-Backend Workflow Demonstrations")
    print("=" * 50)

    try:
        # Basic demonstrations
        demonstrate_image_processing()
        demonstrate_post_processing()
        demonstrate_texture_generation()

        # Advanced demonstrations
        demonstrate_cross_backend_workflow()
        demonstrate_quick_functions()

        # Specialized demonstrations
        demonstrate_photometric_handling()
        demonstrate_image_tensor_handling()

        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
