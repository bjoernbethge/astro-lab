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


def _load_static_file(filename: str) -> str:
    """Load static file content from the widget package."""
    static_dir = Path(__file__).parent / "static"
    file_path = static_dir / filename

    if file_path.exists():
        return file_path.read_text(encoding="utf-8")
    else:
        print(f"⚠️ Static file not found: {file_path}")
        return ""


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

        _esm = _load_static_file("widget.js")
        _css = _load_static_file("widget.css")

        # Widget Traits
        image_data = traitlets.Unicode("").tag(sync=True)
        image_path = traitlets.Unicode("").tag(sync=True)
        render_time = traitlets.Float(0.0).tag(sync=True)
        resolution = traitlets.Unicode("").tag(sync=True)
        render_engine = traitlets.Unicode("BLENDER_EEVEE_NEXT").tag(sync=True)
        samples = traitlets.Int(64).tag(sync=True)
        trigger_render = traitlets.Int(0).tag(sync=True)

        # Ops Editor Traits
        request_ops_data = traitlets.Int(0).tag(sync=True)
        ops_data = traitlets.Unicode("").tag(sync=True)
        execute_op = traitlets.Unicode("").tag(sync=True)
        current_action = traitlets.Unicode("").tag(sync=True)

        def __init__(self, **kwargs):
            if not ANYWIDGET_AVAILABLE:
                raise ImportError("anywidget is required for AstroLabWidget")
            super().__init__(**kwargs)

            # Initialize Blender integration
            self._init_blender_integration()

            # Set up ops editor observers
            self.observe(self._on_request_ops_data, names=["request_ops_data"])
            self.observe(self._on_execute_op, names=["execute_op"])

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

        def _on_request_ops_data(self, change):
            """Handle request for ops data."""
            if not self._bpy_available:
                self.ops_data = '{"error": "Blender not available"}'
                return

            try:
                ops_info = self._get_ops_info()
                import json

                self.ops_data = json.dumps(ops_info)
            except Exception as e:
                self.ops_data = f'{{"error": "Failed to get ops data: {str(e)}"}}'

        def _on_execute_op(self, change):
            """Handle ops execution request."""
            if not self._bpy_available or not change["new"]:
                return

            try:
                import json

                request = json.loads(change["new"])
                category = request.get("category", "")
                operation = request.get("operation", "")
                parameters = request.get("parameters", {})

                # Execute the operation
                if hasattr(self.ops, category):
                    category_ops = getattr(self.ops, category)
                    if hasattr(category_ops, operation):
                        op_func = getattr(category_ops, operation)
                        result = op_func(**parameters)
                        print(
                            f"✅ Executed {category}.{operation} with params: {parameters}"
                        )

                        # Auto-render after execution
                        self.quick_render()
                    else:
                        print(f"❌ Operation {operation} not found in {category}")
                else:
                    print(f"❌ Category {category} not found in ops")

            except Exception as e:
                print(f"❌ Failed to execute operation: {e}")

        def _get_ops_info(self) -> Dict[str, Any]:
            """Get information about available Blender operations."""
            if not self._bpy_available:
                return {}

            ops_info = {}

            # Get common operation categories
            categories = [
                "mesh",
                "object",
                "material",
                "scene",
                "render",
                "curve",
                "surface",
                "text",
            ]

            for category in categories:
                if hasattr(self.ops, category):
                    category_ops = getattr(self.ops, category)
                    ops_list = []

                    # Get operations in this category
                    for op_name in dir(category_ops):
                        if not op_name.startswith("_"):
                            try:
                                op_func = getattr(category_ops, op_name)
                                if callable(op_func):
                                    # Try to get operation info
                                    op_info = {
                                        "name": op_name,
                                        "parameters": self._get_op_parameters(
                                            category, op_name
                                        ),
                                    }
                                    ops_list.append(op_info)
                            except:
                                continue

                    if ops_list:
                        ops_info[category] = ops_list

            return ops_info

        def _get_op_parameters(self, category: str, operation: str) -> list:
            """Get parameter information for a specific operation."""
            try:
                # This is a simplified version - in real Blender, you'd use bpy.ops introspection
                # For now, we'll provide common parameters for popular operations
                common_params = {
                    "mesh": {
                        "primitive_cube_add": [
                            {
                                "name": "size",
                                "type": "float",
                                "default": 2.0,
                                "min": 0.0,
                                "description": "Cube size",
                            },
                            {
                                "name": "location",
                                "type": "vector",
                                "default": [0, 0, 0],
                                "size": 3,
                                "description": "Location",
                            },
                            {
                                "name": "rotation",
                                "type": "vector",
                                "default": [0, 0, 0],
                                "size": 3,
                                "description": "Rotation",
                            },
                        ],
                        "primitive_uv_sphere_add": [
                            {
                                "name": "radius",
                                "type": "float",
                                "default": 1.0,
                                "min": 0.0,
                                "description": "Sphere radius",
                            },
                            {
                                "name": "location",
                                "type": "vector",
                                "default": [0, 0, 0],
                                "size": 3,
                                "description": "Location",
                            },
                            {
                                "name": "rings",
                                "type": "int",
                                "default": 32,
                                "min": 3,
                                "max": 100,
                                "description": "Number of rings",
                            },
                            {
                                "name": "segments",
                                "type": "int",
                                "default": 16,
                                "min": 3,
                                "max": 100,
                                "description": "Number of segments",
                            },
                        ],
                        "primitive_plane_add": [
                            {
                                "name": "size",
                                "type": "float",
                                "default": 2.0,
                                "min": 0.0,
                                "description": "Plane size",
                            },
                            {
                                "name": "location",
                                "type": "vector",
                                "default": [0, 0, 0],
                                "size": 3,
                                "description": "Location",
                            },
                        ],
                    },
                    "object": {
                        "select_all": [
                            {
                                "name": "action",
                                "type": "enum",
                                "default": "TOGGLE",
                                "items": [
                                    {"identifier": "TOGGLE", "name": "Toggle"},
                                    {"identifier": "SELECT", "name": "Select"},
                                    {"identifier": "DESELECT", "name": "Deselect"},
                                    {"identifier": "INVERT", "name": "Invert"},
                                ],
                                "description": "Selection action",
                            }
                        ],
                        "delete": [
                            {
                                "name": "use_global",
                                "type": "boolean",
                                "default": False,
                                "description": "Delete globally",
                            }
                        ],
                        "camera_add": [
                            {
                                "name": "location",
                                "type": "vector",
                                "default": [0, 0, 0],
                                "size": 3,
                                "description": "Camera location",
                            },
                            {
                                "name": "rotation",
                                "type": "vector",
                                "default": [0, 0, 0],
                                "size": 3,
                                "description": "Camera rotation",
                            },
                        ],
                        "light_add": [
                            {
                                "name": "type",
                                "type": "enum",
                                "default": "POINT",
                                "items": [
                                    {"identifier": "POINT", "name": "Point"},
                                    {"identifier": "SUN", "name": "Sun"},
                                    {"identifier": "SPOT", "name": "Spot"},
                                    {"identifier": "AREA", "name": "Area"},
                                ],
                                "description": "Light type",
                            },
                            {
                                "name": "location",
                                "type": "vector",
                                "default": [0, 0, 0],
                                "size": 3,
                                "description": "Light location",
                            },
                        ],
                    },
                    "render": {
                        "render": [
                            {
                                "name": "write_still",
                                "type": "boolean",
                                "default": True,
                                "description": "Write image to file",
                            }
                        ]
                    },
                }

                return common_params.get(category, {}).get(operation, [])

            except Exception as e:
                print(f"⚠️ Could not get parameters for {category}.{operation}: {e}")
                return []

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
