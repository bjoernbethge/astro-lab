"""Tools for ``astro_lab.widgets.enhanced`` (ImageProcessor, tensor bridges)."""

from __future__ import annotations

import json

from pydantic_ai import FunctionToolset

from .common import astro_lab_widgets_dir, parse_class_public_methods

enhanced_widgets_toolset = FunctionToolset()


@enhanced_widgets_toolset.tool
def widgets_image_processor_method_summary() -> str:
    """Public methods on ``ImageProcessor`` from ``enhanced/image_processing.py`` (AST)."""
    path = astro_lab_widgets_dir() / "enhanced" / "image_processing.py"
    if not path.is_file():
        return json.dumps({"error": "image_processing_not_found", "path": str(path)})
    methods = parse_class_public_methods(path, "ImageProcessor")
    return json.dumps(
        {
            "class": "astro_lab.widgets.enhanced.image_processing.ImageProcessor",
            "source": str(path),
            "methods": methods,
        },
        indent=2,
    )
