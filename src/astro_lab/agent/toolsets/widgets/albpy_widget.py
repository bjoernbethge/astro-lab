"""Tools for ``astro_lab.widgets.albpy`` (Blender AlbPy)."""

from __future__ import annotations

import json

from pydantic_ai import FunctionToolset

from .common import astro_lab_widgets_dir, parse_dunder_all

albpy_widgets_toolset = FunctionToolset()


@albpy_widgets_toolset.tool
def widgets_list_albpy_export_names() -> str:
    """Parse ``__all__`` from ``astro_lab/widgets/albpy/__init__.py`` (no bpy import)."""
    init_path = astro_lab_widgets_dir() / "albpy" / "__init__.py"
    if not init_path.is_file():
        return json.dumps({"error": "albpy_init_not_found", "path": str(init_path)})
    exports = parse_dunder_all(init_path)
    return json.dumps(
        {"module": "astro_lab.widgets.albpy", "source": str(init_path), "exports": exports},
        indent=2,
    )
