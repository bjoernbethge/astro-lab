#!/usr/bin/env python3
"""
Command-line interface for automatic documentation generation.
"""

import argparse
from pathlib import Path

from .generator import DocumentationGenerator
from .utils import generate_index_page, save_documentation


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate documentation from Pydantic schemas and Python packages",
        epilog="""
Examples:
  # Generate docs from schemas
  uv run python -m docs.auto.cli --source docs/auto/schemas --output docs/generated

  # Generate compact docs for a single module
  uv run python -m docs.auto.cli --module data_schemas --format compact

  # List all packages with Pydantic models
  uv run python -m docs.auto.cli --list-packages

  # Generate flat package documentation
  uv run python -m docs.auto.cli --package bpy --format compact

  # Generate STRUCTURED package documentation (recommended!)
  uv run python -m docs.auto.cli --package bpy --format compact --structured

  # Generate both flat and structured formats
  uv run python -m docs.auto.cli --package torch_geometric --format both --structured

The --structured option creates organized subdirectories:
  package_NAME/
    ├── README.md           # Package overview
    ├── submodule1.md       # Main submodule docs
    ├── submodule2.md
    └── submodule1/         # Nested submodules
        ├── nested1.md
        └── nested2.md

This makes large packages like 'bpy' (70+ ops files) or 'torch_geometric' (390+ files)
much more navigable and organized!
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default="src/astro_lab",
        help="Source directory to scan (default: src/astro_lab)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="docs/auto",
        help="Output directory for documentation (default: docs/auto)"
    )
    
    parser.add_argument(
        "--module",
        type=str,
        help="Generate docs for specific module only"
    )
    
    parser.add_argument(
        "--package",
        type=str,
        help="Generate docs for installed package (e.g., --package pydantic)"
    )
    
    parser.add_argument(
        "--list-packages",
        action="store_true",
        help="List all installed packages with Pydantic models"
    )
    
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Generate structured package documentation with submodules in separate files"
    )
    
    parser.add_argument(
        "--schemas-only",
        action="store_true",
        help="Generate only Pydantic schema documentation"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["detailed", "compact", "both"],
        default="detailed",
        help="Documentation format (detailed=with JSON, compact=markdown only, both=generate both)"
    )
    
    parser.add_argument(
        "--index",
        action="store_true",
        help="Generate index page only"
    )
    
    args = parser.parse_args()
    
    source_root = Path(args.source)
    output_root = Path(args.output)
    
    if not source_root.exists():
        print(f"❌ Source directory not found: {source_root}")
        return 1
    
    # Create documentation generator
    generator = DocumentationGenerator(source_root)
    
    if args.index:
        # Generate index page only
        print("📝 Generating index page...")
        index_content = generate_index_page(output_root)
        save_documentation(index_content, output_root / "README.md")
        print(f"✅ Index page saved to: {output_root / 'README.md'}")
        return 0
    
    if args.list_packages:
        # List packages with Pydantic models
        print("🔍 Scanning installed packages for Pydantic models...")
        generator = DocumentationGenerator(source_root)
        packages_with_models = generator.find_packages_with_pydantic()
        
        if packages_with_models:
            print("📦 Found packages with Pydantic models:")
            for pkg_name, model_count in packages_with_models.items():
                print(f"  - {pkg_name}: {model_count} models")
        else:
            print("❌ No packages with Pydantic models found")
        
        return 0
    
    if args.package:
        # Generate docs for installed package
        print(f"📝 Generating documentation for package {args.package}...")
        generator = DocumentationGenerator(source_root)
        
        if args.structured:
            # Generate structured documentation
            generator.generate_structured_package_docs(args.package, output_root, args.format)
            return 0
        
        if args.format == "both":
            # Generate both formats
            docs_content_detailed = generator.generate_package_docs(args.package, "detailed")
            docs_content_compact = generator.generate_package_docs(args.package, "compact")
            
            output_path_detailed = output_root / f"package_{args.package}.md"
            output_path_compact = output_root / f"package_{args.package}_compact.md"
            
            save_documentation(docs_content_detailed, output_path_detailed)
            save_documentation(docs_content_compact, output_path_compact)
            
            print(f"✅ Detailed documentation saved to: {output_path_detailed}")
            print(f"✅ Compact documentation saved to: {output_path_compact}")
        else:
            docs_content = generator.generate_package_docs(args.package, args.format)
            
            suffix = "_compact" if args.format == "compact" else ""
            output_path = output_root / f"package_{args.package}{suffix}.md"
            save_documentation(docs_content, output_path)
            
            print(f"✅ Documentation saved to: {output_path}")
        
        return 0
    
    if args.module:
        # Generate docs for specific module
        module_path = source_root / f"{args.module}.py"
        if not module_path.exists():
            print(f"❌ Module not found: {module_path}")
            return 1
        
        print(f"📝 Generating documentation for {args.module}...")
        
        if args.format == "both":
            # Generate both formats
            docs_content_detailed = generator.generate_module_docs(module_path, "detailed")
            docs_content_compact = generator.generate_module_docs(module_path, "compact")
            
            output_path_detailed = output_root / f"{args.module}.md"
            output_path_compact = output_root / f"{args.module}_compact.md"
            
            save_documentation(docs_content_detailed, output_path_detailed)
            save_documentation(docs_content_compact, output_path_compact)
            
            print(f"✅ Detailed documentation saved to: {output_path_detailed}")
            print(f"✅ Compact documentation saved to: {output_path_compact}")
        else:
            docs_content = generator.generate_module_docs(module_path, args.format)
            
            suffix = "_compact" if args.format == "compact" else ""
            output_path = output_root / f"{args.module}{suffix}.md"
            save_documentation(docs_content, output_path)
            
            print(f"✅ Documentation saved to: {output_path}")
        
        # Don't generate index for single module
        return 0
    
    # Generate all documentation
    print("🚀 Starting automatic documentation generation...")
    generator.generate_all_docs(output_root, args.format)
    
    # Generate index page only if we're documenting the main source root
    if source_root.name == "astro_lab" or args.source == "src/astro_lab":
        print("📝 Generating main index page...")
        index_content = generate_index_page(output_root)
        save_documentation(index_content, output_root / "README.md")
    else:
        print("📝 Skipping index page for subdirectory documentation...")
    
    print("🎉 Documentation generation complete!")
    print(f"📁 Documentation available at: {output_root}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 