"""Tools for ``astro_lab.widgets.alo3d`` (Open3D)."""

from __future__ import annotations

import json

from pydantic_ai import FunctionToolset

from .common import astro_lab_widgets_dir, parse_dunder_all

alo3d_widgets_toolset = FunctionToolset()


@alo3d_widgets_toolset.tool
def widgets_list_alo3d_export_names() -> str:
    """Parse ``__all__`` from ``astro_lab/widgets/alo3d/__init__.py`` (no open3d import)."""
    init_path = astro_lab_widgets_dir() / "alo3d" / "__init__.py"
    if not init_path.is_file():
        return json.dumps({"error": "alo3d_init_not_found", "path": str(init_path)})
    exports = parse_dunder_all(init_path)
    return json.dumps(
        {"module": "astro_lab.widgets.alo3d", "source": str(init_path), "exports": exports},
        indent=2,
    )
