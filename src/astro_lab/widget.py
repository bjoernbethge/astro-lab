"""
AnyWidget for direct display of Blender renderings in Jupyter notebooks.

This widget displays rendered images directly in the notebook and enables
interactive controls for various rendering parameters. It can handle both
Render-Ergebnisse als auch Live-Viewport-Daten anzeigen.
"""

import base64
from pathlib import Path
from typing import Optional, Union

import traitlets

try:
    import anywidget

    ANYWIDGET_AVAILABLE = True
except ImportError:
    ANYWIDGET_AVAILABLE = False
    anywidget = None


class BlenderImageWidget(anywidget.AnyWidget if ANYWIDGET_AVAILABLE else object):
    """
    AnyWidget zur Anzeige von Blender-Renderings mit interaktiven Kontrollen.
    """

    _esm = """
    function render({ model, el }) {
        // Container für das Widget erstellen
        const container = document.createElement('div');
        container.style.cssText = `
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            border: 2px solid #333;
            border-radius: 10px;
            background: linear-gradient(135deg, #1e1e1e, #2d2d2d);
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        `;

        // Titel
        const title = document.createElement('h3');
        title.style.cssText = `
            color: #00d4ff;
            margin: 0 0 15px 0;
            text-align: center;
            text-shadow: 0 0 10px rgba(0,212,255,0.5);
        `;
        title.textContent = '🎨 Blender Astronomical Visualization';

        // Image Container
        const imageContainer = document.createElement('div');
        imageContainer.style.cssText = `
            position: relative;
            max-width: 100%;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 15px rgba(0,0,0,0.5);
        `;

        // Bild Element
        const img = document.createElement('img');
        img.style.cssText = `
            max-width: 100%;
            height: auto;
            display: block;
            border-radius: 8px;
        `;

        // Info Panel
        const infoPanel = document.createElement('div');
        infoPanel.style.cssText = `
            margin-top: 15px;
            padding: 10px;
            background: rgba(0,0,0,0.3);
            border-radius: 5px;
            color: #ccc;
            font-size: 14px;
            text-align: center;
            min-width: 300px;
        `;

        // Render Button
        const renderButton = document.createElement('button');
        renderButton.style.cssText = `
            margin-top: 15px;
            padding: 12px 24px;
            background: linear-gradient(45deg, #ff6b35, #f7931e);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255,107,53,0.3);
        `;
        renderButton.textContent = '🚀 New Render';

        // Button Hover Effekt
        renderButton.addEventListener('mouseenter', () => {
            renderButton.style.transform = 'translateY(-2px)';
            renderButton.style.boxShadow = '0 6px 20px rgba(255,107,53,0.4)';
        });
        renderButton.addEventListener('mouseleave', () => {
            renderButton.style.transform = 'translateY(0)';
            renderButton.style.boxShadow = '0 4px 15px rgba(255,107,53,0.3)';
        });

        // Event Listeners
        renderButton.addEventListener('click', () => {
            model.set('trigger_render', model.get('trigger_render') + 1);
            model.save_changes();
        });

        // Update Funktionen
        function updateImage() {
            const imageData = model.get('image_data');
            const imagePath = model.get('image_path');
            
            if (imageData) {
                img.src = `data:image/png;base64,${imageData}`;
                img.style.display = 'block';
            } else if (imagePath) {
                img.src = imagePath;
                img.style.display = 'block';
            } else {
                img.style.display = 'none';
            }
        }

        function updateInfo() {
            const renderTime = model.get('render_time');
            const resolution = model.get('resolution');
            const engine = model.get('render_engine');
            const samples = model.get('samples');
            
            let infoText = '';
            if (engine) infoText += `🎮 Engine: ${engine}<br>`;
            if (resolution) infoText += `📐 Resolution: ${resolution}<br>`;
            if (samples) infoText += `🔢 Samples: ${samples}<br>`;
            if (renderTime) infoText += `⏱️ Render Time: ${renderTime}s`;
            
            infoPanel.innerHTML = infoText || '📊 Rendering information will appear here';
        }

        // Model Change Listeners
        model.on('change:image_data', updateImage);
        model.on('change:image_path', updateImage);
        model.on('change:render_time', updateInfo);
        model.on('change:resolution', updateInfo);
        model.on('change:render_engine', updateInfo);
        model.on('change:samples', updateInfo);

        // Initial Updates
        updateImage();
        updateInfo();

        // Zusammenbauen
        imageContainer.appendChild(img);
        container.appendChild(title);
        container.appendChild(imageContainer);
        container.appendChild(infoPanel);
        container.appendChild(renderButton);
        
        el.appendChild(container);
    }
    export default { render };
    """

    _css = """
    .blender-widget {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .blender-widget img {
        transition: transform 0.3s ease;
    }
    
    .blender-widget img:hover {
        transform: scale(1.02);
    }
    """

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
            raise ImportError("anywidget is required for BlenderImageWidget")
        super().__init__(**kwargs)

    def display_image_from_path(
        self,
        image_path: Union[str, Path],
        render_time: float = 0.0,
        resolution: str = "",
        engine: str = "BLENDER_EEVEE_NEXT",
        samples: int = 64,
    ):
        """
        Display an image from a file.

        Args:
            image_path: Path to the image file
            render_time: Render time in seconds
            resolution: Resolution as string (e.g. "1920x1080")
            engine: Render engine name
            samples: Number of samples
        """
        image_path = Path(image_path)

        if image_path.exists():
            # Convert to Base64 for better compatibility
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            self.image_data = image_data
            self.image_path = ""  # Clear path when using data
            self.render_time = render_time
            self.resolution = resolution
            self.render_engine = engine
            self.samples = samples
        else:
            print(f"⚠️ Image file not found: {image_path}")

    def display_image_from_data(
        self,
        image_data: bytes,
        render_time: float = 0.0,
        resolution: str = "",
        engine: str = "BLENDER_EEVEE_NEXT",
        samples: int = 64,
    ):
        """
        Display an image from binary data.

        Args:
            image_data: Image data as bytes
            render_time: Render time in seconds
            resolution: Resolution as string
            engine: Render engine name
            samples: Number of samples
        """
        encoded_data = base64.b64encode(image_data).decode("utf-8")

        self.image_data = encoded_data
        self.image_path = ""
        self.render_time = render_time
        self.resolution = resolution
        self.render_engine = engine
        self.samples = samples

    def clear_image(self):
        """Clear the displayed image."""
        self.image_data = ""
        self.image_path = ""
        self.render_time = 0.0
        self.resolution = ""

    def capture_live_viewport(self):
        """
        Erfasse Live-Daten aus dem Blender-Viewport.

        Returns:
            Viewport-Screenshot und -Daten
        """
        try:
            from .utils.blender.viewport_capture import capture_viewport, get_scene_data

            # Screenshot aufnehmen
            viewport_image = capture_viewport()
            if viewport_image is not None:
                # Konvertiere zu bytes
                try:
                    import io

                    from PIL import Image

                    # Numpy array zu PIL Image
                    if viewport_image.shape[2] == 4:  # RGBA
                        pil_image = Image.fromarray(viewport_image, "RGBA")
                        pil_image = pil_image.convert("RGB")
                    else:
                        pil_image = Image.fromarray(viewport_image, "RGB")

                    # Zu bytes konvertieren
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format="PNG")
                    image_bytes = buffer.getvalue()

                    # Szenen-Daten holen
                    scene_data = get_scene_data()
                    viewport_info = scene_data.get("viewport_info", {})
                    viewport_size = viewport_info.get("viewport_size", (0, 0))

                    # Widget aktualisieren
                    self.display_image_from_data(
                        image_bytes,
                        render_time=0.0,
                        resolution=f"{viewport_size[0]}x{viewport_size[1]}",
                        engine="VIEWPORT_LIVE",
                        samples=0,
                    )

                    return viewport_image, scene_data
                except ImportError:
                    print("⚠️ PIL nicht verfügbar für Bildkonvertierung")

        except Exception as e:
            print(f"⚠️ Live Viewport Capture Error: {e}")

        return None, {}

    def capture_live_render(self):
        """
        Erfasse Live-Render-Daten aus Blender.

        Returns:
            Render-Daten und -Informationen
        """
        try:
            from .utils.blender.viewport_capture import capture_render

            # Render aufnehmen
            render_data = capture_render()

            if "beauty" in render_data:
                beauty_image = render_data["beauty"]

                # Konvertiere zu bytes
                try:
                    import io

                    import numpy as np
                    from PIL import Image

                    # Float zu uint8 konvertieren
                    if beauty_image.dtype != np.uint8:
                        beauty_image = (beauty_image * 255).astype(np.uint8)

                    # Zu PIL Image
                    if beauty_image.shape[2] == 4:  # RGBA
                        pil_image = Image.fromarray(beauty_image, "RGBA")
                        pil_image = pil_image.convert("RGB")
                    else:
                        pil_image = Image.fromarray(beauty_image, "RGB")

                    # Zu bytes
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format="PNG")
                    image_bytes = buffer.getvalue()

                    # Widget aktualisieren
                    self.display_image_from_data(
                        image_bytes,
                        render_time=0.0,
                        resolution=f"{beauty_image.shape[1]}x{beauty_image.shape[0]}",
                        engine="LIVE_RENDER",
                        samples=0,
                    )

                    return render_data
                except ImportError:
                    print("⚠️ PIL nicht verfügbar für Bildkonvertierung")

        except Exception as e:
            print(f"⚠️ Live Render Capture Error: {e}")

        return {}


# Stub class for when anywidget is not available
class BlenderImageWidgetStub:
    """Stub class when anywidget is not available."""

    def __init__(self, **kwargs):
        print("⚠️ anywidget not available - BlenderImageWidget disabled")

    def display_image_from_path(self, *args, **kwargs):
        print("⚠️ BlenderImageWidget not available")

    def display_image_from_data(self, *args, **kwargs):
        print("⚠️ BlenderImageWidget not available")

    def clear_image(self):
        print("⚠️ BlenderImageWidget not available")


# Export der entsprechenden Klasse
if ANYWIDGET_AVAILABLE:
    __all__ = ["BlenderImageWidget"]
else:
    BlenderImageWidget = BlenderImageWidgetStub
    __all__ = ["BlenderImageWidget"]
