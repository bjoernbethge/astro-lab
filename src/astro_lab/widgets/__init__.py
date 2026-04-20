"""
AstroLab Widget System - Enhanced Astronomical Visualization
==========================================================

Modernste, einheitliche API für alle Visualisierungs-Backends mit Enhanced Module Integration.
Das System kombiniert die Stärken verschiedener Engines optimal und nutzt Enhanced Features.

Quick Start:
-----------
```python
# Enhanced 3D Plots mit verbesserter Performance
from astro_lab.widgets import create_3d_scatter_plot
fig = create_3d_scatter_plot(coordinates, enhanced_colors=True)

# Enhanced Cosmic Web mit Physics Simulation
from astro_lab.widgets import CosmographBridge
bridge = CosmographBridge()
widget = bridge.create_cosmic_web_network(coordinates, scale=10, n_clusters=5)

# Enhanced Blender Rendering mit Publication Quality
from astro_lab.widgets import generate_cosmic_web_scene
result = generate_cosmic_web_scene(coordinates, render=True, enhanced_textures=True)
```

Enhanced Backend Übersicht:
---------------------------
🌟 **AlbPy (Blender)**: Enhanced photorealistic rendering, publication figures
📊 **Plotly**: Enhanced interactive web visualizations, optimierte Performance
🌌 **Cosmograph**: Enhanced large-scale networks mit intelligent physics
🔬 **PyVista (alpv)**: Enhanced scientific 3D analysis (optional)
☁️ **Open3D (alo3d)**: Enhanced real-time point clouds (optional)
⚡ **Enhanced Module**: Zero-copy tensors, advanced post-processing

Paket-Lage (was liegt wo)
-------------------------
- **``enhanced/``** — zentrale Tensor-/Backend-Konverter (``to_plotly``, ``to_pyvista``, …)
  und ``AstronomicalTensorBridge``; sinnvoller Einstieg für neue Backends.
- **``plotly/bridge.py``** — Kern-``AstronomicalPlotlyBridge`` (Figuren aus TensorDicts).
  **``plotly_bridge.py``** — dünne Facade + Convenience (``create_3d_scatter_plot``, …)
  aufbauend auf ``enhanced`` + ``plotly/bridge``.
- **``alcg/bridge.py``** — Cosmograph-Widget-Bridge. **``cosmograph_bridge.py``** —
  UI-Facade mit gleichem Klassennamen, wrappt AlcG + ``enhanced`` (kein Duplikat der
  Implementierung, aber zwei Einstiegsebenen).
- **``albpy_bridge.py``** — Blender-Szenen-API (Cosmic Web, Stellar Field, …) neben
  Paket **``albpy/``** (Node Groups, Operatoren, ``gpu_preferences``).
- **``alpv/tensor_bridge.py``** — PyVista-spezifische Hilfen; nutzt ``enhanced.to_pyvista``,
  nicht zu verwechseln mit ``enhanced/tensor_bridge.py``.
- **``tng50.py``** — Spezial-Visualizer für vorverarbeitete TNG50-Graphen (.pt).

Schwere Importe: dieses Paket startet AlbPy (inkl. GPU-Prefs) und baut u.a.
``enhanced_engine`` beim Import — für CLI-Only-Pfade ggf. gezielt Submodule importieren.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# AlbPy applies GPU prefs inside its package init before node groups load.
from . import albpy  # noqa: F401 — re-export; requires PyPI ``bpy`` matching Python version

from .albpy.gpu_preferences import GpuBackendName, prefer_gpu_backend

from . import alcg  # Enhanced Cosmograph System
from . import plotly  # Komplette Enhanced Plotly Implementation
from .albpy_bridge import (
    AlbPyBridge,
    create_blender_stellar_field,
    create_enhanced_galaxy_cluster,
    generate_cosmic_web_scene,
)
from .cosmograph_bridge import CosmographBridge

# Enhanced Module - Das Herzstück der Performance-Optimierung
from .enhanced import (  # Tensor Bridge für optimierte Datenkonvertierung; Backend Converters für nahtlose Integration; Processing Engines; Zero-Copy Performance; Orchestration Pipelines
    AstronomicalTensorBridge,
    ImageProcessor,
    PostProcessor,
    TextureGenerator,
    ZeroCopyTensorConverter,
    converter,
    orchestrate_pipeline,
    orchestrate_post_processing,
    orchestrate_texture_pipeline,
    tensor_bridge_context,
    to_blender,
    to_cosmograph,
    to_open3d,
    to_plotly,
    to_pyvista,
)

# Enhanced UI-Bridges (IMMER verfügbar mit Enhanced Features)
from .plotly_bridge import (
    create_3d_scatter_plot,
    create_cosmic_web_plot,
    create_hr_diagram,
    create_survey_comparison,
)

# Optionale erweiterte Backends
try:
    from . import alpv  # Enhanced PyVista backend

    ALPV_AVAILABLE = True
except ImportError:
    alpv = None
    ALPV_AVAILABLE = False

try:
    from . import alo3d  # Enhanced Open3D backend

    ALO3D_AVAILABLE = True
except ImportError:
    alo3d = None
    ALO3D_AVAILABLE = False

# Spezialisierte Enhanced Visualizer
from .tng50 import TNG50Visualizer


class EnhancedVisualizationEngine:
    """
    Enhanced Engine der intelligent die besten Backends kombiniert
    Mit vollständiger Enhanced Module Integration für optimale Performance

    Automatische Enhanced Backend-Auswahl:
    - Interactive Analysis → Enhanced Plotly mit Zero-Copy
    - Large Networks → Enhanced Cosmograph mit Physics
    - Publication Quality → Enhanced AlbPy mit Advanced Rendering
    - Scientific Analysis → Enhanced PyVista (falls verfügbar)
    - Point Clouds → Enhanced Open3D (falls verfügbar)
    """

    def __init__(self):
        # Enhanced Module Components
        self.tensor_bridge = AstronomicalTensorBridge()
        self.image_processor = ImageProcessor()
        self.post_processor = PostProcessor()
        self.texture_generator = TextureGenerator()
        self.converter = ZeroCopyTensorConverter()

        # Backend Availability
        self.plotly_available = True
        self.cosmograph_available = True
        self.blender_available = True
        self.pyvista_available = ALPV_AVAILABLE
        self.open3d_available = ALO3D_AVAILABLE

        logger.info("🎨 Enhanced Visualization Engine initialisiert:")
        logger.info("   📊 Plotly: ✅ Enhanced")
        logger.info("   🌌 Cosmograph: ✅ Enhanced")
        logger.info("   🌟 AlbPy (Blender): ✅ Enhanced")
        logger.info(
            f"   🔬 PyVista: {'✅ Enhanced' if self.pyvista_available else '⚠️ Optional'}"
        )
        logger.info(
            f"   ☁️ Open3D: {'✅ Enhanced' if self.open3d_available else '⚠️ Optional'}"
        )
        logger.info("   ⚡ Enhanced Module: ✅ Zero-Copy, Advanced Processing")

    def create_visualization(
        self,
        data: Any,
        visualization_type: str = "auto",
        backend: Optional[str] = None,
        enhanced: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Erstelle Enhanced Visualisierung mit automatischer Backend-Auswahl

        Args:
            data: Input data (coordinates, TensorDict, etc.)
            visualization_type: Art der Visualisierung
            backend: Spezifisches Backend oder None für auto-selection
            enhanced: Enhanced Features aktivieren
            **kwargs: Backend-spezifische Parameter

        Returns:
            Dict mit Visualisierungsergebnissen
        """

        results = {}

        if enhanced:
            # Enhanced Zero-Copy Tensor Conversion
            with tensor_bridge_context(self.tensor_bridge):
                if backend == "plotly" or (
                    backend is None and visualization_type in ["scatter", "analysis"]
                ):
                    results["plotly"] = self._create_enhanced_plotly(
                        data, visualization_type, **kwargs
                    )

                elif backend == "cosmograph" or (
                    backend is None and visualization_type in ["network", "cosmic_web"]
                ):
                    results["cosmograph"] = self._create_enhanced_cosmograph(
                        data, visualization_type, **kwargs
                    )

                elif backend == "blender" or (
                    backend is None and visualization_type in ["render", "publication"]
                ):
                    results["blender"] = self._create_enhanced_blender(
                        data, visualization_type, **kwargs
                    )

                elif backend == "pyvista" and self.pyvista_available:
                    results["pyvista"] = self._create_enhanced_pyvista(
                        data, visualization_type, **kwargs
                    )

                elif backend == "open3d" and self.open3d_available:
                    results["open3d"] = self._create_enhanced_open3d(
                        data, visualization_type, **kwargs
                    )

                else:
                    # Default: Enhanced Plotly
                    results["plotly"] = self._create_enhanced_plotly(
                        data, visualization_type, **kwargs
                    )
        else:
            # Standard processing without Enhanced features
            pass

        return results

    def _create_enhanced_plotly(self, data: Any, viz_type: str, **kwargs) -> Any:
        """Enhanced Plotly Visualization"""
        plotly_data = to_plotly(data, **kwargs)

        if viz_type == "cosmic_web":
            return create_cosmic_web_plot(data, enhanced_colors=True, **kwargs)
        else:
            return create_3d_scatter_plot(data, enhanced_colors=True, **kwargs)

    def _create_enhanced_cosmograph(self, data: Any, viz_type: str, **kwargs) -> Any:
        """Enhanced Cosmograph Visualization"""
        bridge = CosmographBridge()  # Bereits enhanced!
        return bridge.create_network_visualization(data, **kwargs)

    def _create_enhanced_blender(self, data: Any, viz_type: str, **kwargs) -> Any:
        """Enhanced Blender Visualization"""
        bridge = AlbPyBridge()  # Bereits enhanced!

        if viz_type == "cosmic_web":
            return bridge.generate_cosmic_web_scene(
                data, enhanced_textures=True, **kwargs
            )
        elif viz_type == "stellar":
            return bridge.create_stellar_field(data, **kwargs)
        else:
            return bridge.generate_cosmic_web_scene(data, **kwargs)

    def _create_enhanced_pyvista(self, data: Any, viz_type: str, **kwargs) -> Any:
        """Enhanced PyVista Visualization"""
        if self.pyvista_available:
            pyvista_data = to_pyvista(data, **kwargs)
            return self.image_processor.create_pyvista_visualization(
                pyvista_data, **kwargs
            )
        return None

    def _create_enhanced_open3d(self, data: Any, viz_type: str, **kwargs) -> Any:
        """Enhanced Open3D Visualization"""
        if self.open3d_available:
            open3d_data = to_open3d(data, **kwargs)
            return open3d_data  # Enhanced Open3D processing
        return None


# Globale Enhanced Engine Instance
enhanced_engine = EnhancedVisualizationEngine()


# Main Enhanced API Functions
def create_visualization(
    data: Any,
    backend: Optional[str] = None,
    visualization_type: str = "auto",
    enhanced: bool = True,
    **kwargs,
) -> Any:
    """
    Enhanced Hauptfunktion für Visualisierung

    Args:
        data: Input astronomical data
        backend: Spezifisches Backend oder None für auto-selection
        visualization_type: Art der Visualisierung
        enhanced: Enhanced Features aktivieren (empfohlen)
        **kwargs: Backend-spezifische Parameter

    Returns:
        Enhanced Visualization object(s)
    """

    return enhanced_engine.create_visualization(
        data=data,
        backend=backend,
        visualization_type=visualization_type,
        enhanced=enhanced,
        **kwargs,
    )


def plot_cosmic_web(
    data: Any, enhanced: bool = True, backend: str = "auto", **kwargs
) -> Dict[str, Any]:
    """
    Enhanced Cosmic Web Visualization mit automatischer Backend-Auswahl

    Args:
        data: Cosmic web data
        enhanced: Enhanced Features aktivieren
        backend: "auto", "plotly", "cosmograph", "blender"
        **kwargs: Enhanced Parameter

    Returns:
        Enhanced visualization results
    """

    if backend == "auto":
        # Smart backend selection
        data_size = len(data) if hasattr(data, "__len__") else 1000

        if data_size > 10000:
            backend = "cosmograph"  # Large networks
        elif kwargs.get("publication", False):
            backend = "blender"
        else:
            backend = "plotly"  # Interactive analysis

    return enhanced_engine.create_visualization(
        data=data,
        visualization_type="cosmic_web",
        backend=backend,
        enhanced=enhanced,
        **kwargs,
    )


def plot_stellar_data(data: Any, enhanced: bool = True, **kwargs) -> Dict[str, Any]:
    """Enhanced Stellar Data Visualization"""

    return enhanced_engine.create_visualization(
        data=data, visualization_type="stellar", enhanced=enhanced, **kwargs
    )


# Export Enhanced Components
__all__ = [
    # Enhanced Main Engine
    "EnhancedVisualizationEngine",
    "enhanced_engine",
    # Enhanced Main API
    "create_visualization",
    "plot_cosmic_web",
    "plot_stellar_data",
    # Enhanced UI Bridges
    "create_3d_scatter_plot",
    "create_cosmic_web_plot",
    "create_survey_comparison",
    "create_hr_diagram",
    "CosmographBridge",
    "AlbPyBridge",
    "generate_cosmic_web_scene",
    "create_blender_stellar_field",
    "create_enhanced_galaxy_cluster",
    # Enhanced Module Components
    "AstronomicalTensorBridge",
    "tensor_bridge_context",
    "ImageProcessor",
    "PostProcessor",
    "TextureGenerator",
    "ZeroCopyTensorConverter",
    "converter",
    # Enhanced Backend Converters
    "to_plotly",
    "to_cosmograph",
    "to_blender",
    "to_pyvista",
    "to_open3d",
    # Enhanced Orchestration
    "orchestrate_pipeline",
    "orchestrate_post_processing",
    "orchestrate_texture_pipeline",
    # Enhanced Backend Modules
    "albpy",
    "GpuBackendName",
    "prefer_gpu_backend",
    "plotly",  # Enhanced Plotly System
    "alcg",  # Enhanced Cosmograph System
    "alpv",  # Enhanced PyVista (optional)
    "alo3d",  # Enhanced Open3D (optional)
    "enhanced",  # Enhanced Module Core
    # Enhanced Specialized Visualizers
    "TNG50Visualizer",
]
