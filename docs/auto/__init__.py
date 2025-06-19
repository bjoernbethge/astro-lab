"""
Automatic documentation generation for AstroLab.

This module automatically generates documentation from Pydantic schemas
and other structured data types in the AstroLab codebase.
"""

from .generator import DocumentationGenerator
from .utils import extract_pydantic_schemas, generate_schema_docs, create_docs_structure, save_documentation

__all__ = [
    "DocumentationGenerator",
    "extract_pydantic_schemas", 
    "generate_schema_docs",
    "create_docs_structure",
    "save_documentation"
] 