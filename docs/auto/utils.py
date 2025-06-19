"""
Utility functions for documentation generation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Type

from pydantic import BaseModel


def extract_pydantic_schemas(module_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Extract Pydantic schemas from a Python module.
    
    Parameters
    ----------
    module_path : Path
        Path to the Python module
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping model names to their schemas
    """
    try:
        # Import module dynamically
        module_name = _path_to_module_name(module_path)
        module = __import__(module_name, fromlist=[''])
        
        schemas = {}
        
        # Find all Pydantic models
        for name, obj in vars(module).items():
            if (isinstance(obj, type) and 
                issubclass(obj, BaseModel) and 
                hasattr(obj, 'model_json_schema')):
                
                schemas[name] = {
                    'schema': obj.model_json_schema(),
                    'docstring': obj.__doc__,
                    'module': module_name
                }
        
        return schemas
        
    except Exception as e:
        print(f"Warning: Could not extract schemas from {module_path}: {e}")
        return {}


def generate_schema_docs(schemas: Dict[str, Dict[str, Any]], format_type: str = "detailed") -> str:
    """
    Generate markdown documentation from Pydantic schemas.
    
    Parameters
    ----------
    schemas : Dict[str, Dict[str, Any]]
        Dictionary of schemas to document
    format_type : str
        Format type: "detailed" (with JSON), "compact" (markdown only)
        
    Returns
    -------
    str
        Generated markdown documentation
    """
    if format_type == "compact":
        return generate_compact_schema_docs(schemas)
    else:
        return generate_detailed_schema_docs(schemas)


def generate_compact_schema_docs(schemas: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate compact markdown documentation from Pydantic schemas.
    
    Parameters
    ----------
    schemas : Dict[str, Dict[str, Any]]
        Dictionary of schemas to document
        
    Returns
    -------
    str
        Generated compact markdown documentation
    """
    docs = []
    
    for model_name, model_info in schemas.items():
        docs.append(f"## {model_name}")
        docs.append("")
        
        # Add docstring if available
        if model_info.get('docstring'):
            docs.append(model_info['docstring'])
            docs.append("")
        
        # Add schema information in compact format
        schema = model_info['schema']
        
        if 'properties' in schema:
            docs.append("### Parameters")
            docs.append("")
            
            for prop_name, prop_info in schema['properties'].items():
                # Parameter name and type
                prop_type = _format_type(prop_info)
                
                # Format parameter line
                param_line = f"**`{prop_name}`** *({prop_type})*"
                
                # Add default value
                if 'default' in prop_info:
                    param_line += f" = `{prop_info['default']}`"
                
                docs.append(param_line)
                
                # Description
                if 'description' in prop_info:
                    docs.append(f"  {prop_info['description']}")
                
                # Constraints in compact format
                constraints = _format_constraints(prop_info)
                if constraints:
                    docs.append(f"  *{constraints}*")
                
                docs.append("")
        
        # Add usage example
        docs.append("### Usage")
        docs.append("")
        docs.append("```python")
        
        # Use the correct module path from the schema info
        module_path = model_info.get('module', 'unknown_module')
        docs.append(f"from {module_path} import {model_name}")
        docs.append("")
        
        # Add example parameters
        if 'properties' in schema:
            required_params = []
            optional_params = []
            
            for prop_name, prop_info in schema['properties'].items():
                if 'default' not in prop_info:
                    example_value = _get_example_value(prop_info)
                    required_params.append(f"    {prop_name}={example_value}")
                else:
                    example_value = _get_example_value(prop_info)
                    optional_params.append(f"    # {prop_name}={example_value}")
            
            docs.append(f"config = {model_name}(")
            if required_params:
                docs.append(",\n".join(required_params))
            if optional_params:
                docs.append("")
                docs.append("    # Optional parameters:")
                docs.append("\n".join(optional_params))
            docs.append(")")
        else:
            docs.append(f"config = {model_name}()")
        
        docs.append("```")
        docs.append("")
    
    return "\n".join(docs)


def generate_detailed_schema_docs(schemas: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate detailed markdown documentation from Pydantic schemas.
    This is the original implementation with full JSON schemas.
    """
    docs = []
    
    for model_name, model_info in schemas.items():
        docs.append(f"## {model_name}")
        docs.append("")
        
        # Add docstring if available
        if model_info.get('docstring'):
            docs.append(model_info['docstring'])
            docs.append("")
        
        # Add schema information
        schema = model_info['schema']
        
        if 'properties' in schema:
            docs.append("### Parameters")
            docs.append("")
            
            for prop_name, prop_info in schema['properties'].items():
                # Parameter name and type
                prop_type = prop_info.get('type', 'unknown')
                if 'anyOf' in prop_info:
                    # Handle Optional types
                    types = [t.get('type', 'unknown') for t in prop_info['anyOf'] if 'type' in t]
                    prop_type = f"Optional[{', '.join(types)}]"
                
                docs.append(f"**{prop_name}** ({prop_type})")
                
                # Description
                if 'description' in prop_info:
                    docs.append(f": {prop_info['description']}")
                
                # Default value
                if 'default' in prop_info:
                    docs.append(f" (default: `{prop_info['default']}`)")
                
                # Constraints
                constraints = []
                if 'minimum' in prop_info:
                    constraints.append(f"min: {prop_info['minimum']}")
                if 'maximum' in prop_info:
                    constraints.append(f"max: {prop_info['maximum']}")
                if 'minLength' in prop_info:
                    constraints.append(f"min length: {prop_info['minLength']}")
                if 'maxLength' in prop_info:
                    constraints.append(f"max length: {prop_info['maxLength']}")
                
                if constraints:
                    docs.append(f" ({', '.join(constraints)})")
                
                docs.append("")
        
        # Add JSON schema
        docs.append("### JSON Schema")
        docs.append("")
        docs.append("```json")
        docs.append(json.dumps(schema, indent=2))
        docs.append("```")
        docs.append("")
    
    return "\n".join(docs)


def _format_type(prop_info: Dict[str, Any]) -> str:
    """Format property type for compact display."""
    if 'anyOf' in prop_info:
        types = [t.get('type', 'unknown') for t in prop_info['anyOf'] if 'type' in t]
        if 'null' in [t.get('type') for t in prop_info['anyOf']]:
            non_null_types = [t for t in types if t != 'null']
            if non_null_types:
                return f"Optional[{non_null_types[0]}]"
        return f"Union[{', '.join(types)}]"
    return prop_info.get('type', 'unknown')


def _format_constraints(prop_info: Dict[str, Any]) -> str:
    """Format constraints for compact display."""
    constraints = []
    if 'minimum' in prop_info:
        constraints.append(f"≥{prop_info['minimum']}")
    if 'maximum' in prop_info:
        constraints.append(f"≤{prop_info['maximum']}")
    if 'exclusiveMinimum' in prop_info:
        constraints.append(f">{prop_info['exclusiveMinimum']}")
    if 'exclusiveMaximum' in prop_info:
        constraints.append(f"<{prop_info['exclusiveMaximum']}")
    if 'minLength' in prop_info:
        constraints.append(f"min len: {prop_info['minLength']}")
    if 'maxLength' in prop_info:
        constraints.append(f"max len: {prop_info['maxLength']}")
    
    return ", ".join(constraints)


def _get_example_value(prop_info: Dict[str, Any]) -> str:
    """Get example value for a property."""
    prop_type = prop_info.get('type', 'unknown')
    
    if prop_type == 'string':
        return '"example"'
    elif prop_type == 'integer':
        return str(prop_info.get('minimum', 1))
    elif prop_type == 'number':
        return str(prop_info.get('minimum', 1.0))
    elif prop_type == 'boolean':
        return str(prop_info.get('default', True))
    elif prop_type == 'array':
        return '[]'
    elif prop_type == 'object':
        return '{}'
    else:
        return 'None'


def create_docs_structure(source_root: Path, docs_root: Path) -> None:
    """
    Create documentation directory structure mirroring source code.
    
    Parameters
    ----------
    source_root : Path
        Root directory of source code
    docs_root : Path
        Root directory for documentation
    """
    # Create base docs directory
    docs_root.mkdir(parents=True, exist_ok=True)
    
    # Mirror directory structure
    for item in source_root.rglob("*"):
        if item.is_dir() and not item.name.startswith("__"):
            # Create corresponding docs directory
            rel_path = item.relative_to(source_root)
            doc_dir = docs_root / rel_path
            doc_dir.mkdir(parents=True, exist_ok=True)


def save_documentation(docs_content: str, output_path: Path) -> None:
    """
    Save documentation content to file.
    
    Parameters
    ----------
    docs_content : str
        Documentation content to save
    output_path : Path
        Output file path
    """
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save content
    output_path.write_text(docs_content, encoding="utf-8")


def generate_index_page(docs_root: Path) -> str:
    """
    Generate an index page for the documentation.
    
    Parameters
    ----------
    docs_root : Path
        Root directory of documentation
        
    Returns
    -------
    str
        Generated index page content
    """
    docs = []
    docs.append("# AstroLab Auto-Generated Documentation")
    docs.append("")
    docs.append("This documentation is automatically generated from Pydantic schemas and docstrings in the AstroLab codebase.")
    docs.append("")
    docs.append("## Module Structure")
    docs.append("")
    
    # Find all documentation files
    md_files = list(docs_root.rglob("*.md"))
    md_files = [f for f in md_files if f.name != "README.md"]
    
    # Group by directory
    by_directory = {}
    for md_file in md_files:
        rel_path = md_file.relative_to(docs_root)
        dir_name = str(rel_path.parent) if rel_path.parent != Path(".") else "root"
        
        if dir_name not in by_directory:
            by_directory[dir_name] = []
        by_directory[dir_name].append(rel_path)
    
    # Generate directory tree
    for dir_name, files in sorted(by_directory.items()):
        docs.append(f"### {dir_name}")
        docs.append("")
        
        for file_path in sorted(files):
            module_name = file_path.stem
            docs.append(f"- [{module_name}]({file_path})")
        
        docs.append("")
    
    docs.append("## Schema Overview")
    docs.append("")
    docs.append("The documentation includes the following types of schemas:")
    docs.append("")
    docs.append("- **Data Schemas**: Configuration for datasets and data loaders")
    docs.append("- **Tensor Schemas**: Configuration for astronomical tensor types")
    docs.append("- **Model Schemas**: Configuration for machine learning models")
    docs.append("- **Training Schemas**: Configuration for training procedures")
    docs.append("- **Utility Schemas**: Configuration for visualization and utilities")
    docs.append("")
    
    return "\n".join(docs)


def _path_to_module_name(module_path: Path) -> str:
    """Convert file path to Python module name."""
    try:
        # Try relative to current working directory first
        relative_path = module_path.relative_to(Path.cwd())
    except ValueError:
        # If that fails, try using the path as-is
        relative_path = module_path
    
    module_parts = relative_path.with_suffix('').parts
    return '.'.join(module_parts) 