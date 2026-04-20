"""AST helpers for widget toolsets (no ``import astro_lab.widgets``)."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


def astro_lab_widgets_dir() -> Path:
    spec = importlib.util.find_spec("astro_lab")
    if spec is None or not spec.origin:
        raise RuntimeError("Cannot locate astro_lab package")
    return Path(spec.origin).resolve().parent / "widgets"


def parse_dunder_all(py_path: Path) -> list[str]:
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    names: list[str] = []
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            names.append(elt.value)
                    return names
    return []


def parse_create_functions_signatures(py_path: Path) -> list[dict[str, str]]:
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    out: list[dict[str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("create_"):
            continue
        args = ast.unparse(node.args) if hasattr(ast, "unparse") else "(...)"
        out.append({"name": node.name, "signature": f"{node.name}({args})"})
    return sorted(out, key=lambda x: x["name"])


def parse_class_public_methods(py_path: Path, class_name: str) -> list[str]:
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            methods: list[str] = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                    methods.append(item.name)
            return sorted(methods)
    return []
