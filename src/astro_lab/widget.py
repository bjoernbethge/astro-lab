"""
AstroLab Widget - Interactive Blender visualization widget for Jupyter notebooks.

This widget provides direct access to bpy.ops for enhanced usability and
allows real-time Blender scene manipulation from Jupyter notebooks.
"""

import base64
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import anywidget
    import traitlets

    ANYWIDGET_AVAILABLE = True
except ImportError:
    ANYWIDGET_AVAILABLE = False
    anywidget = None
    traitlets = None

# Modern CSS with enhanced UI
CSS = """
.astro-widget {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    color: white;
    max-width: 100%;
    margin: 10px 0;
}

.astro-header {
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 20px;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
}

.astro-controls {
    display: flex;
    justify-content: center;
    gap: 15px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}

.astro-button {
    background: rgba(255, 255, 255, 0.2);
    border: 2px solid rgba(255, 255, 255, 0.3);
    color: white;
    padding: 10px 20px;
    border-radius: 25px;
    cursor: pointer;
    transition: all 0.3s ease;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.astro-button:hover {
    background: rgba(255, 255, 255, 0.3);
    border-color: rgba(255, 255, 255, 0.5);
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
}

.astro-image-container {
    text-align: center;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
    min-height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.astro-image {
    max-width: 100%;
    max-height: 600px;
    border-radius: 8px;
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.4);
}

.astro-info {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin-top: 15px;
}

.astro-info-item {
    background: rgba(255, 255, 255, 0.1);
    padding: 10px;
    border-radius: 8px;
    text-align: center;
    backdrop-filter: blur(10px);
}

.astro-info-label {
    font-size: 12px;
    opacity: 0.8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 5px;
}

.astro-info-value {
    font-size: 16px;
    font-weight: bold;
}

.astro-placeholder {
    color: rgba(255, 255, 255, 0.6);
    font-style: italic;
    font-size: 18px;
}

.astro-ops-info {
    background: rgba(0, 255, 100, 0.2);
    border: 1px solid rgba(0, 255, 100, 0.3);
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 15px;
    text-align: center;
}

.astro-scene-controls {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
    margin-bottom: 20px;
}

.astro-quick-button {
    background: rgba(255, 255, 255, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.3);
    color: white;
    padding: 8px 12px;
    border-radius: 20px;
    cursor: pointer;
    transition: all 0.3s ease;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.astro-quick-button:hover {
    background: rgba(255, 255, 255, 0.25);
    transform: scale(1.05);
}
"""

JS = """
function render({ model, el }) {
    const container = document.createElement('div');
    container.className = 'astro-widget';
    
    const header = document.createElement('div');
    header.className = 'astro-header';
    header.textContent = '🚀 AstroLab Widget';
    container.appendChild(header);
    
    // Ops status indicator
    const opsInfo = document.createElement('div');
    opsInfo.className = 'astro-ops-info';
    opsInfo.innerHTML = '⚡ widget.ops - Direct Blender Access Available';
    container.appendChild(opsInfo);
    
    // Quick scene controls
    const sceneControls = document.createElement('div');
    sceneControls.className = 'astro-scene-controls';
    
    const quickButtons = [
        { text: 'Galaxy', action: 'galaxy' },
        { text: 'Solar System', action: 'solar_system' },
        { text: 'Reset Scene', action: 'reset' },
        { text: 'Quick Render', action: 'render' }
    ];
    
    quickButtons.forEach(btn => {
        const button = document.createElement('button');
        button.className = 'astro-quick-button';
        button.textContent = btn.text;
        button.onclick = () => {
            model.set('trigger_render', model.get('trigger_render') + 1);
            model.save_changes();
        };
        sceneControls.appendChild(button);
    });
    
    container.appendChild(sceneControls);
    
    // Main controls
    const controls = document.createElement('div');
    controls.className = 'astro-controls';
    
    const renderButton = document.createElement('button');
    renderButton.className = 'astro-button';
    renderButton.textContent = '🎬 Render Scene';
    renderButton.onclick = () => {
        model.set('trigger_render', model.get('trigger_render') + 1);
        model.save_changes();
    };
    controls.appendChild(renderButton);
    
    const clearButton = document.createElement('button');
    clearButton.className = 'astro-button';
    clearButton.textContent = '🧹 Clear';
    clearButton.onclick = () => {
        model.set('image_data', '');
        model.set('image_path', '');
        model.save_changes();
    };
    controls.appendChild(clearButton);
    
    container.appendChild(controls);
    
    // Image container
    const imageContainer = document.createElement('div');
    imageContainer.className = 'astro-image-container';
    
    const updateImage = () => {
        const imageData = model.get('image_data');
        const imagePath = model.get('image_path');
        
        if (imageData) {
            imageContainer.innerHTML = `<img src="data:image/png;base64,${imageData}" class="astro-image" alt="Rendered Image">`;
        } else if (imagePath) {
            imageContainer.innerHTML = `<img src="${imagePath}" class="astro-image" alt="Rendered Image">`;
        } else {
            imageContainer.innerHTML = '<div class="astro-placeholder">🌌 Ready for astronomical visualization</div>';
        }
    };
    
    updateImage();
    model.on('change:image_data', updateImage);
    model.on('change:image_path', updateImage);
    
    container.appendChild(imageContainer);
    
    // Info panel
    const infoPanel = document.createElement('div');
    infoPanel.className = 'astro-info';
    
    const updateInfo = () => {
        const renderTime = model.get('render_time');
        const resolution = model.get('resolution');
        const engine = model.get('render_engine');
        const samples = model.get('samples');
        
        infoPanel.innerHTML = `
            <div class="astro-info-item">
                <div class="astro-info-label">Render Time</div>
                <div class="astro-info-value">${renderTime.toFixed(2)}s</div>
            </div>
            <div class="astro-info-item">
                <div class="astro-info-label">Resolution</div>
                <div class="astro-info-value">${resolution || 'N/A'}</div>
            </div>
            <div class="astro-info-item">
                <div class="astro-info-label">Engine</div>
                <div class="astro-info-value">${engine}</div>
            </div>
            <div class="astro-info-item">
                <div class="astro-info-label">Samples</div>
                <div class="astro-info-value">${samples}</div>
            </div>
        `;
    };
    
    updateInfo();
    model.on('change:render_time', updateInfo);
    model.on('change:resolution', updateInfo);
    model.on('change:render_engine', updateInfo);
    model.on('change:samples', updateInfo);
    
    container.appendChild(infoPanel);
    el.appendChild(container);
}

export default { render };
"""

if ANYWIDGET_AVAILABLE:

    class AstroLabWidget(anywidget.AnyWidget):
        """
        AstroLab Widget for interactive Blender visualization in Jupyter notebooks.

        Features:
        - Direct bpy.ops access via widget.ops
        - Quick scene creation and rendering
        - Real-time scene information
        - Modern, responsive UI

        Usage:
            widget = AstroLabWidget()
            widget.ops.mesh.primitive_cube_add()  # Direct Blender ops access
            widget.quick_render()  # Render and display
        """

        _esm = JS
        _css = CSS

        # Widget Traits
        image_data = traitlets.Unicode("").tag(sync=True)
        image_path = traitlets.Unicode("").tag(sync=True)
        render_time = traitlets.Float(0.0).tag(sync=True)
        resolution = traitlets.Unicode("").tag(sync=True)
        render_engine = traitlets.Unicode("BLENDER_EEVEE_NEXT").tag(sync=True)
        samples = traitlets.Int(64).tag(sync=True)
        trigger_render = traitlets.Int(0).tag(sync=True)

        def __init__(self, **kwargs):
            if not ANYWIDGET_AVAILABLE:
                raise ImportError("anywidget is required for AstroLabWidget")
            super().__init__(**kwargs)

            # Initialize Blender integration
            self._init_blender_integration()

        def _init_blender_integration(self):
            """Initialize Blender integration and bpy.ops access."""
            try:
                import bpy

                self._bpy_available = True
                self.bpy = bpy
                self.ops = bpy.ops  # Direct ops access
                self.data = bpy.data
                self.context = bpy.context
                print("✅ Blender integration initialized - widget.ops available!")
            except ImportError:
                self._bpy_available = False
                self.bpy = None
                self.ops = None
                self.data = None
                self.context = None
                print("⚠️ Blender not available - ops functionality disabled")

        def display_image_from_path(
            self,
            image_path: Union[str, Path],
            render_time: float = 0.0,
            resolution: str = "",
            engine: str = "BLENDER_EEVEE_NEXT",
            samples: int = 64,
        ):
            """Display image from file path with render information."""
            try:
                image_path = Path(image_path)
                if not image_path.exists():
                    print(f"❌ Image not found: {image_path}")
                    return

                # Convert to absolute path for proper display
                abs_path = image_path.resolve()
                self.image_path = f"file:///{abs_path}"
                self.image_data = ""  # Clear base64 data when using path

                # Update render info
                self.render_time = render_time
                self.resolution = resolution
                self.render_engine = engine
                self.samples = samples

                print(f"🖼️ Image displayed: {abs_path}")

            except Exception as e:
                print(f"❌ Error displaying image: {e}")

        def display_image_from_base64(
            self,
            base64_data: str,
            render_time: float = 0.0,
            resolution: str = "",
            engine: str = "BLENDER_EEVEE_NEXT",
            samples: int = 64,
        ):
            """Display image from base64 data with render information."""
            try:
                self.image_data = base64_data
                self.image_path = ""  # Clear path when using base64

                # Update render info
                self.render_time = render_time
                self.resolution = resolution
                self.render_engine = engine
                self.samples = samples

                print("🖼️ Image displayed from base64 data")

            except Exception as e:
                print(f"❌ Error displaying base64 image: {e}")

        def display_image_from_file(
            self,
            file_path: Union[str, Path],
            as_base64: bool = False,
            render_time: float = 0.0,
            resolution: str = "",
            engine: str = "BLENDER_EEVEE_NEXT",
            samples: int = 64,
        ):
            """Display image from file, optionally converting to base64."""
            try:
                file_path = Path(file_path)
                if not file_path.exists():
                    print(f"❌ File not found: {file_path}")
                    return

                if as_base64:
                    with open(file_path, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode()
                    self.display_image_from_base64(
                        image_data, render_time, resolution, engine, samples
                    )
                else:
                    self.display_image_from_path(
                        file_path, render_time, resolution, engine, samples
                    )

            except Exception as e:
                print(f"❌ Error loading image file: {e}")

        def clear_image(self):
            """Clear the displayed image."""
            self.image_data = ""
            self.image_path = ""
            self.render_time = 0.0
            self.resolution = ""

        def quick_render(self, output_path: str = "results/widget_render.png"):
            """Quick render current scene and display in widget."""
            if not self._bpy_available:
                print("❌ Blender not available for rendering")
                return False

            try:
                import time
                from pathlib import Path

                # Ensure output directory exists
                output_path_obj = Path(output_path)
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)

                # Set render settings
                self.context.scene.render.filepath = str(output_path_obj)
                self.context.scene.render.image_settings.file_format = "PNG"

                # Record render time
                start_time = time.time()

                # Render
                self.ops.render.render(write_still=True)

                render_time = time.time() - start_time

                # Get render info
                scene = self.context.scene
                resolution = f"{scene.render.resolution_x}x{scene.render.resolution_y}"
                engine = scene.render.engine

                # Display in widget
                self.display_image_from_path(
                    output_path_obj,
                    render_time=render_time,
                    resolution=resolution,
                    engine=engine,
                    samples=getattr(scene.cycles, "samples", 64)
                    if engine == "CYCLES"
                    else 64,
                )

                print(f"🎬 Quick render completed: {output_path_obj}")
                return True

            except Exception as e:
                print(f"❌ Render error: {e}")
                return False

        def scene_info(self) -> Dict[str, Any]:
            """Get current scene information."""
            if not self._bpy_available:
                return {"error": "Blender not available"}

            try:
                scene = self.context.scene
                return {
                    "scene_name": scene.name,
                    "render_engine": scene.render.engine,
                    "resolution": f"{scene.render.resolution_x}x{scene.render.resolution_y}",
                    "frame_current": scene.frame_current,
                    "frame_start": scene.frame_start,
                    "frame_end": scene.frame_end,
                    "objects_count": len(self.data.objects),
                    "cameras_count": len(
                        [obj for obj in self.data.objects if obj.type == "CAMERA"]
                    ),
                    "lights_count": len(
                        [obj for obj in self.data.objects if obj.type == "LIGHT"]
                    ),
                    "meshes_count": len(
                        [obj for obj in self.data.objects if obj.type == "MESH"]
                    ),
                }
            except Exception as e:
                return {"error": str(e)}

        def reset_scene(self):
            """Reset scene using widget.ops."""
            if not self._bpy_available:
                print("❌ Blender not available")
                return False

            try:
                # Select all objects
                self.ops.object.select_all(action="SELECT")
                # Delete all selected objects
                self.ops.object.delete(use_global=False)
                print("🧹 Scene reset completed")
                return True
            except Exception as e:
                print(f"❌ Scene reset error: {e}")
                return False

        def add_camera(self, location=(5, -5, 3), rotation=(1.1, 0, 0.8)):
            """Add camera using widget.ops."""
            if not self._bpy_available:
                print("❌ Blender not available")
                return None

            try:
                self.ops.object.camera_add(location=location, rotation=rotation)
                camera = self.context.active_object
                self.context.scene.camera = camera
                print(f"📷 Camera added at {location}")
                return camera
            except Exception as e:
                print(f"❌ Camera add error: {e}")
                return None

        def add_light(self, light_type="SUN", location=(5, 5, 10), energy=1000):
            """Add light using widget.ops."""
            if not self._bpy_available:
                print("❌ Blender not available")
                return None

            try:
                self.ops.object.light_add(type=light_type, location=location)
                light = self.context.active_object
                light.data.energy = energy
                print(f"💡 {light_type} light added at {location}")
                return light
            except Exception as e:
                print(f"❌ Light add error: {e}")
                return None

        def create_astro_scene(self, scene_type="galaxy"):
            """Create predefined astronomical scenes."""
            if not self._bpy_available:
                print("❌ Blender not available")
                return False

            try:
                # Reset scene
                self.reset_scene()

                if scene_type == "galaxy":
                    # Add camera
                    self.add_camera(location=(10, -10, 8))
                    # Add lights
                    self.add_light("SUN", location=(5, -5, 10), energy=1000)
                    # Add some objects for galaxy simulation
                    for i in range(20):
                        import random

                        x = random.uniform(-5, 5)
                        y = random.uniform(-5, 5)
                        z = random.uniform(-1, 1)
                        self.ops.mesh.primitive_ico_sphere_add(
                            location=(x, y, z), radius=0.1
                        )

                elif scene_type == "solar_system":
                    # Add camera
                    self.add_camera(location=(15, -15, 10))
                    # Add sun (large sphere)
                    self.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0), radius=2)
                    # Add planets
                    for i in range(8):
                        distance = 3 + i * 1.5
                        self.ops.mesh.primitive_uv_sphere_add(
                            location=(distance, 0, 0), radius=0.2 + i * 0.1
                        )

                print(f"🌌 {scene_type} scene created")
                return True

            except Exception as e:
                print(f"❌ Scene creation error: {e}")
                return False

        # Convenience methods for common operations
        def add_cube(self, location=(0, 0, 0), size=2.0):
            """Add cube using widget.ops."""
            if self._bpy_available:
                self.ops.mesh.primitive_cube_add(location=location, size=size)
                return self.context.active_object
            return None

        def add_sphere(self, location=(0, 0, 0), radius=1.0):
            """Add sphere using widget.ops."""
            if self._bpy_available:
                self.ops.mesh.primitive_uv_sphere_add(location=location, radius=radius)
                return self.context.active_object
            return None

        def add_plane(self, location=(0, 0, 0), size=2.0):
            """Add plane using widget.ops."""
            if self._bpy_available:
                self.ops.mesh.primitive_plane_add(location=location, size=size)
                return self.context.active_object
            return None

else:
    # Stub class for when anywidget is not available
    class AstroLabWidgetStub:
        """Stub class when anywidget is not available."""

        def __init__(self, **kwargs):
            raise ImportError(
                "anywidget is required for AstroLabWidget. "
                "Install with: pip install anywidget"
            )

    AstroLabWidget = AstroLabWidgetStub
