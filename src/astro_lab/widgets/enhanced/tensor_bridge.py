"""TensorDict / dict routing to the widgets ``create_visualization`` entry point."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .tensor_converters import converter


class AstronomicalTensorBridge:
    """Optional unit conversion, backend autodetect, then ``create_visualization``."""

    def __init__(self) -> None:
        self.converter = converter

    def to_visualization(self, tensordict: Any, backend: str = "auto", **kwargs: Any) -> Any:
        from .. import create_visualization

        if kwargs.get("convert_units", False):
            target_unit = kwargs.pop("target_unit", "pc")
            source_unit = kwargs.pop("source_unit", None)
            tensordict = self.converter.convert_tensordict_units(
                tensordict, target_unit, source_unit=source_unit
            )

        if backend == "auto":
            backend = self._auto_select_backend(tensordict, **kwargs)

        return create_visualization(tensordict, backend=backend, **kwargs)

    def _auto_select_backend(self, tensordict: Any, **kwargs: Any) -> str:
        if kwargs.get("photorealistic"):
            return "blender"
        if kwargs.get("web_export"):
            return "plotly"
        if kwargs.get("interactive", True):
            features = self.converter.extract_features(tensordict)
            coords = features.get("coordinates")
            if coords is not None and len(coords) > 100_000:
                return "open3d"
            return "pyvista"
        return "pyvista"


@contextmanager
def tensor_bridge_context(
    bridge: AstronomicalTensorBridge | None = None,
) -> Iterator[AstronomicalTensorBridge]:
    """Yield a bridge; pass ``bridge`` to reuse an engine-owned instance."""
    b = bridge or AstronomicalTensorBridge()
    yield b


__all__ = [
    "AstronomicalTensorBridge",
    "tensor_bridge_context",
]
