"""Tools for ``astro_lab.widgets.alcg`` (Cosmograph)."""

from __future__ import annotations

import json

from pydantic_ai import FunctionToolset

from .common import astro_lab_widgets_dir, parse_dunder_all

alcg_widgets_toolset = FunctionToolset()


@alcg_widgets_toolset.tool
def widgets_list_alcg_export_names() -> str:
    """Parse ``__all__`` from ``astro_lab/widgets/alcg/__init__.py`` (no widget package import)."""
    init_path = astro_lab_widgets_dir() / "alcg" / "__init__.py"
    if not init_path.is_file():
        return json.dumps({"error": "alcg_init_not_found", "path": str(init_path)})
    exports = parse_dunder_all(init_path)
    return json.dumps(
        {"module": "astro_lab.widgets.alcg", "source": str(init_path), "exports": exports},
        indent=2,
    )
