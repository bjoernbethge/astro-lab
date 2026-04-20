"""Tools for ``astro_lab.widgets.plotly`` and root ``plotly_bridge``."""

from __future__ import annotations

import json

from pydantic_ai import FunctionToolset

from .common import astro_lab_widgets_dir, parse_create_functions_signatures, parse_dunder_all

plotly_widgets_toolset = FunctionToolset()


@plotly_widgets_toolset.tool
def widgets_list_plotly_bridge_functions() -> str:
    """List ``create_*`` function definitions in ``widgets/plotly_bridge.py`` (AST; no import)."""
    path = astro_lab_widgets_dir() / "plotly_bridge.py"
    if not path.is_file():
        return json.dumps({"error": "plotly_bridge_not_found", "path": str(path)})
    rows = parse_create_functions_signatures(path)
    return json.dumps({"source": str(path), "functions": rows}, indent=2)


@plotly_widgets_toolset.tool
def widgets_list_plotly_package_exports() -> str:
    """Parse ``__all__`` from ``astro_lab/widgets/plotly/__init__.py``."""
    init_path = astro_lab_widgets_dir() / "plotly" / "__init__.py"
    if not init_path.is_file():
        return json.dumps({"error": "plotly_init_not_found", "path": str(init_path)})
    exports = parse_dunder_all(init_path)
    return json.dumps(
        {"module": "astro_lab.widgets.plotly", "source": str(init_path), "exports": exports},
        indent=2,
    )
