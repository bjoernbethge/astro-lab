"""
Generate and update documentation automatically based on project structure.

This script scans the src/astro_lab directory and creates corresponding
documentation files for mkdocstrings, then builds the documentation.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

import yaml


def find_python_modules(src_path: Path) -> Dict[str, Set[str]]:
    """Find all Python modules and their submodules."""
    modules = {}

    for root, dirs, files in os.walk(src_path):
        # Skip __pycache__ and .mypy_cache directories
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith("__pycache__") and not d.startswith(".mypy_cache")
        ]

        root_path = Path(root)

        # Check if this directory contains an __init__.py file
        if "__init__.py" in files:
            # Get relative path from src/astro_lab
            rel_path = root_path.relative_to(src_path)

            # Convert path to module name
            if rel_path == Path("."):
                module_name = "astro_lab"
            else:
                module_name = f"astro_lab.{'.'.join(rel_path.parts)}"

            # Find Python files in this directory (excluding __init__.py)
            py_files = [f for f in files if f.endswith(".py") and f != "__init__.py"]
            submodules = {f[:-3] for f in py_files}  # Remove .py extension

            modules[module_name] = submodules

    return modules


def validate_module_exists(module_name: str, src_path: Path) -> bool:
    """Check if a module actually exists and can be imported."""
    try:
        # Convert module name to path
        if module_name == "astro_lab":
            module_path = src_path / "__init__.py"
        else:
            parts = module_name.split(".")
            if parts[0] != "astro_lab":
                return False
            module_path = src_path / "/".join(parts[1:]) / "__init__.py"

        return module_path.exists()
    except Exception:
        return False


def create_api_doc_content(module_name: str) -> str:
    """Create content for an API documentation file."""
    return f"""# {module_name}

::: {module_name}
    handler: python
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true
      separate_signature: true
      show_inheritance_diagram: true
      modernize_annotations: true
      group_by_category: true
      show_docstring_examples: true
      show_docstring_attributes: true
      show_docstring_parameters: true
      show_docstring_returns: true
      show_docstring_raises: true
      show_docstring_warns: true
      show_docstring_yields: true
      show_docstring_classes: true
      show_docstring_functions: true
      show_docstring_modules: true
      show_docstring_other_parameters: true
      show_if_no_docstring: false
      show_labels: true
      show_object_full_path: false
      show_root_full_path: false
      show_root_heading: true
      show_root_members_full_path: false
      show_root_toc_entry: true
      members_order: alphabetical
      summary:
        attributes: true
        classes: true
        functions: true
        modules: true
      docstring_style: google
      docstring_section_style: table
      line_length: 88
      unwrap_annotated: true
      signature_crossrefs: true
      show_overloads: true
      show_bases: true
      backlinks: tree
      filters:
        - '!^_'
"""


def generate_api_index_content(modules: Dict[str, Set[str]]) -> str:
    """Generate content for the main API index page."""
    content = """# API Reference

The API is organized by module. Each main module has its own page:

"""

    # Sort modules for consistent ordering
    sorted_modules = sorted(modules.keys())

    for module in sorted_modules:
        content += f"- [{module}](api/{module}.md)\n"

    content += "\nEach page documents the respective module and all contained classes, functions, and submodules.\n"

    return content


def update_mkdocs_nav(modules: Dict[str, Set[str]]) -> List[Dict]:
    """Generate navigation structure for mkdocs.yml."""
    api_nav = [{"Übersicht": "api.md"}]

    # Sort modules for consistent ordering
    sorted_modules = sorted(modules.keys())

    for module in sorted_modules:
        api_nav.append({module: f"api/{module}.md"})

    return api_nav


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)

    return result


def check_dependencies():
    """Check if required tools are available."""
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML is required but not installed.")
        sys.exit(1)

    # Check if mkdocs is available
    result = run_command("uv run mkdocs --version", check=False)
    if result.returncode != 0:
        print("Error: mkdocs is not available. Please install it with: uv add mkdocs")
        sys.exit(1)


def generate_api_docs():
    """Generate API documentation files."""
    # Define paths
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src" / "astro_lab"
    docs_api_path = project_root / "docs" / "api"
    mkdocs_yml = project_root / "mkdocs.yml"

    # Ensure API docs directory exists
    docs_api_path.mkdir(exist_ok=True)

    # Find all modules
    modules = find_python_modules(src_path)

    # Filter out non-existent modules
    valid_modules = {}
    for module_name, submodules in modules.items():
        if validate_module_exists(module_name, src_path):
            valid_modules[module_name] = submodules
            print(f"Found valid module: {module_name}")
        else:
            print(f"Skipping invalid module: {module_name}")

    print(f"Generating documentation for {len(valid_modules)} modules...")

    # Generate API documentation files
    for module in valid_modules:
        api_file = docs_api_path / f"{module}.md"
        content = create_api_doc_content(module)

        with open(api_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Generated: {api_file}")

    # Update API index
    api_index = project_root / "docs" / "api.md"
    index_content = generate_api_index_content(valid_modules)

    with open(api_index, "w", encoding="utf-8") as f:
        f.write(index_content)
    print(f"Updated: {api_index}")

    # Update mkdocs.yml navigation
    try:
        with open(mkdocs_yml, "r", encoding="utf-8") as f:
            mkdocs_config = yaml.safe_load(f)

        # Check if config was loaded properly
        if mkdocs_config and "nav" in mkdocs_config:
            # Update API Reference navigation
            api_nav = update_mkdocs_nav(valid_modules)

            # Find and update the API Reference section
            for i, nav_item in enumerate(mkdocs_config["nav"]):
                if isinstance(nav_item, dict) and "API Reference" in nav_item:
                    mkdocs_config["nav"][i]["API Reference"] = api_nav
                    break

            # Write back to file
            with open(mkdocs_yml, "w", encoding="utf-8") as f:
                yaml.dump(
                    mkdocs_config,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
            print(f"Updated: {mkdocs_yml}")

    except Exception as e:
        print(f"Warning: Could not update mkdocs.yml: {e}")


def update_documentation():
    """Update all documentation."""
    print("Checking dependencies...")
    check_dependencies()

    print("Generating API documentation...")
    generate_api_docs()

    print("Building documentation...")
    run_command("uv run mkdocs build --clean")


def serve_docs():
    """Start development server for documentation."""
    run_command("uv run mkdocs serve", check=False)


def deploy_docs():
    """Deploy documentation to GitHub Pages."""
    update_documentation()
    run_command("uv run mkdocs gh-deploy --force")


def main():
    """Main CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(description="AstroLab Documentation Manager")
    parser.add_argument(
        "command", choices=["update", "serve", "deploy"], help="Command to run"
    )

    args = parser.parse_args()

    if args.command == "update":
        update_documentation()
    elif args.command == "serve":
        serve_docs()
    elif args.command == "deploy":
        deploy_docs()


if __name__ == "__main__":
    main()
