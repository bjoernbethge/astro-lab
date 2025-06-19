"""
Main documentation generator for AstroLab modules.
"""

import ast
import importlib
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel


class DocumentationGenerator:
    """
    Generates documentation from Python modules with Pydantic models.
    
    This class scans AstroLab modules for Pydantic models, functions,
    and classes, then generates structured documentation.
    """
    
    def __init__(self, source_root: Union[str, Path] = "src/astro_lab"):
        """
        Initialize the documentation generator.
        
        Parameters
        ----------
        source_root : str or Path
            Root directory of the source code to document
        """
        self.source_root = Path(source_root)
        self.docs_root = Path("docs/auto")
        self.discovered_models: Dict[str, Type[BaseModel]] = {}
        self.discovered_functions: Dict[str, Any] = {}
        self.discovered_classes: Dict[str, Type] = {}
    
    def scan_module(self, module_path: Path) -> Dict[str, Any]:
        """
        Scan a module for Pydantic models, functions, classes, and constants.
        
        Parameters
        ----------
        module_path : Path
            Path to the Python module to scan
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing discovered elements
        """
        try:
            # Import module dynamically
            module_name = self._path_to_module_name(module_path)
            
            # Add the source root to sys.path if not already there
            source_parent = self.source_root.parent
            if str(source_parent) not in sys.path:
                sys.path.insert(0, str(source_parent))
            
            module = importlib.import_module(module_name)
            
            elements = {
                'pydantic_models': [],
                'functions': [],
                'classes': [],
                'constants': []
            }
            
            for name, obj in inspect.getmembers(module):
                # Skip private members
                if name.startswith('_'):
                    continue
                    
                # Skip common imports that shouldn't be documented
                if name in ['Field', 'BaseModel', 'Optional', 'List', 'Dict', 'Any', 'Union']:
                    continue
                    
                if self._is_pydantic_model(obj):
                    try:
                        elements['pydantic_models'].append({
                            'name': name,
                            'obj': obj,
                            'schema': obj.model_json_schema(),
                            'docstring': inspect.getdoc(obj)
                        })
                    except Exception as e:
                        print(f"Warning: Could not get schema for {name}: {e}")
                elif inspect.isfunction(obj):
                    # Only include functions that are defined in this module
                    if self._is_function_defined_in_module(obj, module):
                        elements['functions'].append({
                            'name': name,
                            'obj': obj,
                            'signature': str(inspect.signature(obj)),
                            'docstring': inspect.getdoc(obj)
                        })
                elif inspect.isclass(obj):
                    # Only include classes that are defined in this module
                    if self._is_class_defined_in_module(obj, module):
                        elements['classes'].append({
                            'name': name,
                            'obj': obj,
                            'docstring': inspect.getdoc(obj),
                            'methods': self._get_class_methods(obj)
                        })
                elif isinstance(obj, (str, int, float, bool, list, dict)):
                    elements['constants'].append({
                        'name': name,
                        'value': obj,
                        'type': type(obj).__name__
                    })
            
            return elements
            
        except Exception as e:
            print(f"Warning: Could not scan module {module_path}: {e}")
            return {'pydantic_models': [], 'functions': [], 'classes': [], 'constants': []}
    
    def _path_to_module_name(self, module_path: Path) -> str:
        """Convert file path to Python module name."""
        try:
            # Try relative to current working directory first
            relative_path = module_path.relative_to(Path.cwd())
        except ValueError:
            # If that fails, try relative to source root
            try:
                relative_path = module_path.relative_to(self.source_root.parent)
            except ValueError:
                # Last resort: use the path as-is
                relative_path = module_path
        
        module_parts = relative_path.with_suffix('').parts
        return '.'.join(module_parts)
    
    def _is_pydantic_model(self, obj: Any) -> bool:
        """Check if an object is a Pydantic model."""
        try:
            return (inspect.isclass(obj) and 
                   issubclass(obj, BaseModel) and 
                   hasattr(obj, 'model_json_schema'))
        except TypeError:
            return False
    
    def _get_class_methods(self, cls: Type) -> List[Dict[str, Any]]:
        """Extract methods from a class that are actually defined in the class (not inherited)."""
        methods = []
        
        # Get methods that are defined directly in this class
        for name, method in cls.__dict__.items():
            # Skip private methods and attributes
            if name.startswith('_'):
                continue
                
            # Check if it's a callable (method/function)
            if not callable(method):
                continue
            
            # Skip class methods and static methods for now (they're still callable but different)
            if isinstance(method, (classmethod, staticmethod)):
                # Unwrap to get the actual function
                if isinstance(method, classmethod):
                    actual_method = method.__func__
                elif isinstance(method, staticmethod):
                    actual_method = method.__func__
                else:
                    actual_method = method
            else:
                actual_method = method
                
            try:
                signature = str(inspect.signature(actual_method))
                methods.append({
                    'name': name,
                    'signature': signature,
                    'docstring': inspect.getdoc(actual_method)
                })
            except (ValueError, TypeError):
                # Skip methods where signature can't be determined
                methods.append({
                    'name': name,
                    'signature': '(...)',
                    'docstring': inspect.getdoc(actual_method) if hasattr(actual_method, '__doc__') else None
                })
                
        return methods
    
    def generate_module_docs(self, module_path: Path, format_type: str = "detailed") -> str:
        """
        Generate markdown documentation for a module.
        
        Parameters
        ----------
        module_path : Path
            Path to the module to document
        format_type : str
            Format type: "detailed" (with JSON), "compact" (markdown only)
            
        Returns
        -------
        str
            Generated markdown documentation
        """
        elements = self.scan_module(module_path)
        
        # Generate markdown
        docs = []
        module_name = module_path.stem
        
        docs.append(f"# {module_name.title()} Module")
        docs.append("")
        docs.append(f"Auto-generated documentation for `{self._path_to_module_name(module_path)}`")
        docs.append("")
        
        # Pydantic Models (with their methods)
        if elements['pydantic_models']:
            # Convert to the format expected by generate_schema_docs
            schemas = {}
            for model in elements['pydantic_models']:
                schemas[model['name']] = {
                    'schema': model['schema'],
                    'docstring': model['docstring'],
                    'module': self._path_to_module_name(module_path)
                }
            
            from .utils import generate_schema_docs
            schema_docs = generate_schema_docs(schemas, format_type)
            docs.append(schema_docs)
            
            # Also document methods for Pydantic models
            docs.append("## Pydantic Model Methods")
            docs.append("")
            
            for model in elements['pydantic_models']:
                methods = self._get_class_methods(model['obj'])
                if methods:
                    docs.append(f"### {model['name']} Methods")
                    docs.append("")
                    for method in methods:
                        docs.append(f"**`{method['name']}{method['signature']}`**")
                        docs.append("")
                        if method['docstring']:
                            # Format docstring with proper indentation
                            docstring_lines = method['docstring'].split('\n')
                            for line in docstring_lines:
                                docs.append(line.strip())
                            docs.append("")
                        else:
                            docs.append("*No documentation available.*")
                            docs.append("")
        
        # Functions
        if elements['functions']:
            docs.append("## Functions")
            docs.append("")
            
            for func in elements['functions']:
                docs.append(f"### {func['name']}{func['signature']}")
                docs.append("")
                
                if func['docstring']:
                    docs.append(func['docstring'])
                    docs.append("")
        
        # Classes
        if elements['classes']:
            docs.append("## Classes")
            docs.append("")
            
            for cls in elements['classes']:
                docs.append(f"### {cls['name']}")
                docs.append("")
                
                if cls['docstring']:
                    docs.append(cls['docstring'])
                    docs.append("")
                
                if cls['methods']:
                    docs.append("#### Methods")
                    docs.append("")
                    for method in cls['methods']:
                        docs.append(f"**`{method['name']}{method['signature']}`**")
                        docs.append("")
                        if method['docstring']:
                            # Format docstring with proper indentation
                            docstring_lines = method['docstring'].split('\n')
                            for line in docstring_lines:
                                docs.append(line.strip())
                            docs.append("")
                        else:
                            docs.append("*No documentation available.*")
                            docs.append("")
        
        # Constants
        if elements['constants']:
            docs.append("## Constants")
            docs.append("")
            
            for const in elements['constants']:
                docs.append(f"- **{const['name']}** ({const['type']}): `{const['value']}`")
            docs.append("")
        
        return "\n".join(docs)
    
    def generate_all_docs(self, output_root: Optional[Path] = None, format_type: str = "detailed") -> None:
        """Generate documentation for all modules in the source tree."""
        if output_root is None:
            output_root = self.docs_root
        
        print(f"🔍 Scanning {self.source_root} for modules...")
        
        # Create docs directory structure
        output_root.mkdir(parents=True, exist_ok=True)
        
        # Find all Python files
        python_files = list(self.source_root.rglob("*.py"))
        python_files = [f for f in python_files if not f.name.startswith("__")]
        
        print(f"📝 Found {len(python_files)} Python files to document")
        
        for py_file in python_files:
            try:
                # Generate relative path for docs
                rel_path = py_file.relative_to(self.source_root)
                
                if format_type == "both":
                    # Generate both formats
                    docs_content_detailed = self.generate_module_docs(py_file, "detailed")
                    docs_content_compact = self.generate_module_docs(py_file, "compact")
                    
                    doc_path_detailed = output_root / rel_path.with_suffix(".md")
                    doc_path_compact = output_root / (rel_path.stem + "_compact.md")
                    
                    # Create directory if needed
                    doc_path_detailed.parent.mkdir(parents=True, exist_ok=True)
                    doc_path_compact.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save both versions
                    doc_path_detailed.write_text(docs_content_detailed, encoding="utf-8")
                    doc_path_compact.write_text(docs_content_compact, encoding="utf-8")
                    
                    print(f"✅ Generated docs: {doc_path_detailed}")
                    print(f"✅ Generated docs: {doc_path_compact}")
                else:
                    # Generate single format
                    docs_content = self.generate_module_docs(py_file, format_type)
                    
                    suffix = "_compact" if format_type == "compact" else ""
                    doc_path = output_root / (rel_path.stem + suffix + ".md")
                    
                    # Create directory if needed
                    doc_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save documentation
                    doc_path.write_text(docs_content, encoding="utf-8")
                    
                    print(f"✅ Generated docs: {doc_path}")
                
            except Exception as e:
                print(f"❌ Failed to generate docs for {py_file}: {e}")
        
        print(f"🎉 Documentation generation complete!")
        print(f"📁 Docs saved to: {output_root}")
    
    def _is_function_defined_in_module(self, func: Any, module: Any) -> bool:
        """Check if a function is defined in the given module (not imported)."""
        try:
            return (hasattr(func, '__module__') and 
                   func.__module__ == module.__name__)
        except (AttributeError, TypeError):
            return False
    
    def _is_class_defined_in_module(self, cls: Any, module: Any) -> bool:
        """Check if a class is defined in the given module (not imported)."""
        try:
            return (hasattr(cls, '__module__') and 
                   cls.__module__ == module.__name__)
        except (AttributeError, TypeError):
            return False
    
    def scan_package(self, package_name: str) -> Dict[str, Any]:
        """
        Scan an installed package for Pydantic models and other documentation.
        
        Parameters
        ----------
        package_name : str
            Name of the installed package to scan
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing discovered elements
        """
        try:
            # Import the package
            package = importlib.import_module(package_name)
            
            elements = {
                'pydantic_models': [],
                'functions': [],
                'classes': [],
                'constants': [],
                'submodules': [],
                'attributes': [],
                'c_extensions': []
            }
            
            # Get package info
            try:
                dist = importlib.metadata.distribution(package_name)
                elements['package_info'] = {
                    'version': dist.version,
                    'location': str(dist.locate_file('')) if hasattr(dist, 'locate_file') else 'unknown',
                    'summary': dist.metadata.get('Summary', None)
                }
            except:
                elements['package_info'] = {'version': 'unknown', 'location': 'unknown'}
            
            # Scan main package module
            for name, obj in inspect.getmembers(package):
                # Skip private members
                if name.startswith('_'):
                    continue
                    
                if self._is_pydantic_model(obj):
                    try:
                        elements['pydantic_models'].append({
                            'name': name,
                            'obj': obj,
                            'schema': obj.model_json_schema(),
                            'docstring': inspect.getdoc(obj),
                            'module': f"{package_name}.{name}"
                        })
                    except Exception as e:
                        print(f"Warning: Could not get schema for {package_name}.{name}: {e}")
                elif inspect.isfunction(obj):
                    if hasattr(obj, '__module__') and obj.__module__ and obj.__module__.startswith(package_name):
                        try:
                            elements['functions'].append({
                                'name': name,
                                'obj': obj,
                                'signature': str(inspect.signature(obj)),
                                'docstring': inspect.getdoc(obj),
                                'module': obj.__module__
                            })
                        except Exception as e:
                            # Handle C-functions that don't have signatures
                            elements['functions'].append({
                                'name': name,
                                'obj': obj,
                                'signature': '(...)',
                                'docstring': inspect.getdoc(obj),
                                'module': getattr(obj, '__module__', package_name)
                            })
                elif inspect.isclass(obj):
                    if hasattr(obj, '__module__') and obj.__module__ and obj.__module__.startswith(package_name):
                        elements['classes'].append({
                            'name': name,
                            'obj': obj,
                            'docstring': inspect.getdoc(obj),
                            'methods': self._get_class_methods(obj),
                            'module': obj.__module__
                        })
                elif inspect.ismodule(obj):
                    if hasattr(obj, '__name__') and obj.__name__.startswith(package_name):
                        # Enhanced submodule scanning
                        submodule_info = {
                            'name': name,
                            'module_name': obj.__name__,
                            'docstring': inspect.getdoc(obj),
                            'functions': [],
                            'classes': [],
                            'attributes': []
                        }
                        
                        # Scan submodule for interesting content
                        try:
                            # Dynamic discovery of important types (not hard-coded)
                            important_types = []
                            other_types = []
                            all_members = inspect.getmembers(obj)
                            
                            # Heuristics to identify important types
                            for name, member in all_members:
                                if name.startswith('_'):
                                    continue
                                
                                # Skip functions and modules
                                if inspect.isfunction(member) or inspect.ismodule(member):
                                    continue
                                
                                # Heuristics for "important" types:
                                is_important = False
                                
                                # 1. Common important type patterns
                                important_patterns = [
                                    # Core data types
                                    r'^(Object|Node|Element|Item|Entity)$',
                                    # Graphics/3D types
                                    r'^(Mesh|Material|Texture|Camera|Light|Scene)$',
                                    # Animation/Time types
                                    r'^(Action|Animation|Keyframe|Timeline)$',
                                    # Collection/Container types
                                    r'^(Collection|Group|Set|List)$',
                                    # System/Core types
                                    r'^(Context|Data|World|System)$',
                                    # Common class suffixes
                                    r'.*(?:Type|Class|Base|Manager|Handler|Controller)$'
                                ]
                                
                                # Check if name matches important patterns
                                import re
                                for pattern in important_patterns:
                                    if re.match(pattern, name, re.IGNORECASE):
                                        is_important = True
                                        break
                                
                                # 2. Type-based heuristics
                                type_str = str(type(member))
                                
                                # Look for metaclasses (often important base types)
                                if 'metaclass' in type_str.lower():
                                    is_important = True
                                
                                # Look for C extension types (common in packages like bpy, numpy)
                                if any(indicator in type_str for indicator in ['built-in', 'extension', 'wrapper']):
                                    is_important = True
                                
                                # 3. Documentation-based heuristics
                                docstring = inspect.getdoc(member)
                                if docstring:
                                    # Look for documentation patterns that suggest importance
                                    important_doc_patterns = [
                                        r'base class', r'main.*class', r'core.*type',
                                        r'represents.*object', r'fundamental.*type'
                                    ]
                                    for doc_pattern in important_doc_patterns:
                                        if re.search(doc_pattern, docstring.lower()):
                                            is_important = True
                                            break
                                
                                # 4. Attribute-based heuristics
                                try:
                                    # Types with many attributes are often important
                                    if hasattr(member, '__dict__') and len(getattr(member, '__dict__', {})) > 10:
                                        is_important = True
                                    
                                    # Types that are callable classes (not functions)
                                    if (hasattr(member, '__call__') and 
                                        hasattr(member, '__name__') and 
                                        not inspect.isfunction(member)):
                                        is_important = True
                                except:
                                    pass
                                
                                # Categorize the type
                                type_info = {
                                    'name': name,
                                    'type': type_str,
                                    'docstring': docstring,
                                    'has_methods': bool(getattr(member, '__dict__', {})),
                                    'is_callable': callable(member)
                                }
                                
                                if is_important:
                                    important_types.append(type_info)
                                else:
                                    other_types.append(type_info)
                            
                            # Sort by importance (name length and alphabetical)
                            important_types.sort(key=lambda x: (len(x['name']), x['name']))
                            other_types.sort(key=lambda x: (len(x['name']), x['name']))
                            
                            # Limit results to keep documentation manageable
                            if important_types:
                                submodule_info['important_types'] = important_types[:15]  # Top 15 important
                            
                            if other_types:
                                submodule_info['other_types'] = other_types[:10]  # Top 10 others
                            
                            for sub_name, sub_obj in inspect.getmembers(obj):
                                if sub_name.startswith('_'):
                                    continue
                                
                                if inspect.isfunction(sub_obj):
                                    try:
                                        submodule_info['functions'].append({
                                            'name': sub_name,
                                            'signature': str(inspect.signature(sub_obj)),
                                            'docstring': inspect.getdoc(sub_obj)
                                        })
                                    except:
                                        submodule_info['functions'].append({
                                            'name': sub_name,
                                            'signature': '(...)',
                                            'docstring': inspect.getdoc(sub_obj)
                                        })
                                elif inspect.isclass(sub_obj):
                                    submodule_info['classes'].append({
                                        'name': sub_name,
                                        'docstring': inspect.getdoc(sub_obj),
                                        'methods': self._get_class_methods(sub_obj)[:5]  # Limit to first 5 methods
                                    })
                                elif inspect.ismodule(sub_obj) and hasattr(sub_obj, '__name__') and sub_obj.__name__.startswith(package_name):
                                    # Deep scan for nested submodules (like bpy.ops.mesh, bpy.types.Object)
                                    nested_info = {
                                        'name': sub_name,
                                        'module_name': sub_obj.__name__,
                                        'docstring': inspect.getdoc(sub_obj),
                                        'functions': [],
                                        'classes': [],
                                        'operators': []  # Special for bpy.ops
                                    }
                                    
                                    try:
                                        # Scan nested submodule
                                        for nested_name, nested_obj in inspect.getmembers(sub_obj):
                                            if nested_name.startswith('_'):
                                                continue
                                            
                                            if inspect.isfunction(nested_obj):
                                                try:
                                                    nested_info['functions'].append({
                                                        'name': nested_name,
                                                        'signature': str(inspect.signature(nested_obj)),
                                                        'docstring': inspect.getdoc(nested_obj)
                                                    })
                                                except:
                                                    nested_info['functions'].append({
                                                        'name': nested_name,
                                                        'signature': '(...)',
                                                        'docstring': inspect.getdoc(nested_obj)
                                                    })
                                            elif inspect.isclass(nested_obj):
                                                class_info = {
                                                    'name': nested_name,
                                                    'docstring': inspect.getdoc(nested_obj),
                                                    'methods': []
                                                }
                                                
                                                # Get some methods for classes
                                                try:
                                                    methods = self._get_class_methods(nested_obj)[:3]
                                                    class_info['methods'] = methods
                                                except:
                                                    pass
                                                
                                                nested_info['classes'].append(class_info)
                                            elif hasattr(nested_obj, '__call__'):
                                                # Special handling for bpy.ops operators
                                                if 'ops' in sub_obj.__name__:
                                                    nested_info['operators'].append({
                                                        'name': nested_name,
                                                        'docstring': inspect.getdoc(nested_obj)
                                                    })
                                    
                                        # Only add if we found something interesting
                                        if (nested_info['functions'] or nested_info['classes'] or 
                                            nested_info['operators']):
                                            if 'nested_submodules' not in submodule_info:
                                                submodule_info['nested_submodules'] = []
                                            submodule_info['nested_submodules'].append(nested_info)
                                    
                                    except Exception as e:
                                        print(f"Warning: Could not scan nested submodule {sub_obj.__name__}: {e}")
                                
                                elif not inspect.ismodule(sub_obj) and not callable(sub_obj):
                                    # Document interesting attributes
                                    attr_type = type(sub_obj).__name__
                                    if attr_type not in ['module', 'builtin_function_or_method']:
                                        submodule_info['attributes'].append({
                                            'name': sub_name,
                                            'type': attr_type,
                                            'value': str(sub_obj)[:100] if len(str(sub_obj)) < 100 else str(sub_obj)[:100] + "..."
                                        })
                        except Exception as e:
                            print(f"Warning: Could not scan submodule {obj.__name__}: {e}")
                        
                        elements['submodules'].append(submodule_info)
                else:
                    # Document other interesting attributes (C-extensions, etc.)
                    attr_type = type(obj).__name__
                    if attr_type not in ['module', 'builtin_function_or_method', 'type']:
                        elements['attributes'].append({
                            'name': name,
                            'type': attr_type,
                            'value': str(obj)[:100] if len(str(obj)) < 100 else str(obj)[:100] + "...",
                            'docstring': inspect.getdoc(obj)
                        })
                    
                    # Special handling for C-extensions
                    if 'built-in' in str(type(obj)) or 'extension' in str(type(obj)):
                        elements['c_extensions'].append({
                            'name': name,
                            'type': str(type(obj)),
                            'docstring': inspect.getdoc(obj)
                        })
            
            return elements
            
        except Exception as e:
            print(f"Warning: Could not scan package {package_name}: {e}")
            return {'pydantic_models': [], 'functions': [], 'classes': [], 'constants': [], 'submodules': [], 'attributes': [], 'c_extensions': []}
    
    def generate_package_docs(self, package_name: str, format_type: str = "detailed") -> str:
        """
        Generate markdown documentation for an installed package.
        
        Parameters
        ----------
        package_name : str
            Name of the package to document
        format_type : str
            Format type: "detailed" (with JSON), "compact" (markdown only)
            
        Returns
        -------
        str
            Generated markdown documentation
        """
        elements = self.scan_package(package_name)
        
        # Generate markdown
        docs = []
        docs.append(f"# {package_name.title()} Package Documentation")
        docs.append("")
        docs.append(f"Auto-generated documentation for installed package `{package_name}`")
        docs.append("")
        
        # Package info
        if 'package_info' in elements:
            info = elements['package_info']
            docs.append("## Package Information")
            docs.append("")
            docs.append(f"- **Version**: {info.get('version', 'unknown')}")
            docs.append(f"- **Location**: {info.get('location', 'unknown')}")
            if info.get('summary'):
                docs.append(f"- **Summary**: {info['summary']}")
            docs.append("")
        
        # Submodules
        if elements['submodules']:
            docs.append("## Submodules")
            docs.append("")
            for submodule in elements['submodules']:
                docs.append(f"### {submodule['name']}")
                docs.append(f"Module: `{submodule['module_name']}`")
                docs.append("")
                if submodule['docstring']:
                    docs.append(submodule['docstring'])
                    docs.append("")
                
                # Document submodule functions
                if submodule.get('functions'):
                    docs.append("#### Functions")
                    docs.append("")
                    for func in submodule['functions'][:10]:  # Limit to first 10
                        docs.append(f"- **`{func['name']}{func['signature']}`**")
                        if func['docstring']:
                            # Take first line of docstring
                            first_line = func['docstring'].split('\n')[0]
                            docs.append(f"  {first_line}")
                        docs.append("")
                
                # Document submodule classes
                if submodule.get('classes'):
                    docs.append("#### Classes")
                    docs.append("")
                    for cls in submodule['classes'][:5]:  # Limit to first 5
                        docs.append(f"- **`{cls['name']}`**")
                        if cls['docstring']:
                            # Take first line of docstring
                            first_line = cls['docstring'].split('\n')[0]
                            docs.append(f"  {first_line}")
                        
                        # Show some methods if available
                        if cls.get('methods'):
                            method_names = [m['name'] for m in cls['methods'][:3]]
                            docs.append(f"  Methods: {', '.join(method_names)}")
                        docs.append("")
                
                # Document submodule attributes
                if submodule.get('attributes'):
                    docs.append("#### Attributes")
                    docs.append("")
                    for attr in submodule['attributes'][:5]:  # Limit to first 5
                        docs.append(f"- **`{attr['name']}`** ({attr['type']}): `{attr['value']}`")
                    docs.append("")
                
                # Document important types (dynamically discovered)
                if submodule.get('important_types'):
                    docs.append("#### Important Data Types")
                    docs.append("")
                    for itype in submodule['important_types']:
                        docs.append(f"- **`{itype['name']}`** ({itype['type']})")
                        if itype['docstring']:
                            # Take first line of docstring
                            first_line = itype['docstring'].split('\n')[0]
                            docs.append(f"  {first_line}")
                        
                        # Add additional info
                        info_parts = []
                        if itype.get('has_methods'):
                            info_parts.append("has methods")
                        if itype.get('is_callable'):
                            info_parts.append("callable")
                        
                        if info_parts:
                            docs.append(f"  *({', '.join(info_parts)})*")
                        docs.append("")
                
                # Document other types
                if submodule.get('other_types'):
                    docs.append("#### Other Data Types")
                    docs.append("")
                    for otype in submodule['other_types']:
                        docs.append(f"- **`{otype['name']}`** ({otype['type']})")
                        if otype['docstring']:
                            # Take first line of docstring
                            first_line = otype['docstring'].split('\n')[0]
                            docs.append(f"  {first_line}")
                    docs.append("")
                
                # Document nested submodules (like bpy.ops.mesh, bpy.types.Object)
                if submodule.get('nested_submodules'):
                    docs.append("#### Nested Submodules")
                    docs.append("")
                    for nested in submodule['nested_submodules'][:8]:  # Limit to first 8
                        docs.append(f"##### {nested['name']}")
                        docs.append(f"Module: `{nested['module_name']}`")
                        docs.append("")
                        
                        if nested['docstring']:
                            # Take first line of docstring
                            first_line = nested['docstring'].split('\n')[0]
                            docs.append(first_line)
                            docs.append("")
                        
                        # Show operators (special for bpy.ops)
                        if nested.get('operators'):
                            docs.append("**Operators:**")
                            op_names = [op['name'] for op in nested['operators'][:5]]
                            docs.append(f"`{', '.join(op_names)}`")
                            docs.append("")
                        
                        # Show functions
                        if nested.get('functions'):
                            func_names = [f['name'] for f in nested['functions'][:5]]
                            docs.append(f"**Functions:** `{', '.join(func_names)}`")
                            docs.append("")
                        
                        # Show classes
                        if nested.get('classes'):
                            class_names = [c['name'] for c in nested['classes'][:5]]
                            docs.append(f"**Classes:** `{', '.join(class_names)}`")
                            docs.append("")
                    docs.append("")
        
        # Pydantic Models (with their methods)
        if elements['pydantic_models']:
            # Convert to the format expected by generate_schema_docs
            schemas = {}
            for model in elements['pydantic_models']:
                schemas[model['name']] = {
                    'schema': model['schema'],
                    'docstring': model['docstring'],
                    'module': model.get('module', package_name)
                }
            
            from .utils import generate_schema_docs
            schema_docs = generate_schema_docs(schemas, format_type)
            docs.append(schema_docs)
            
            # Also document methods for Pydantic models
            docs.append("## Pydantic Model Methods")
            docs.append("")
            
            for model in elements['pydantic_models']:
                methods = self._get_class_methods(model['obj'])
                if methods:
                    docs.append(f"### {model['name']} Methods")
                    docs.append("")
                    for method in methods:
                        docs.append(f"**`{method['name']}{method['signature']}`**")
                        docs.append("")
                        if method['docstring']:
                            # Format docstring with proper indentation
                            docstring_lines = method['docstring'].split('\n')
                            for line in docstring_lines:
                                docs.append(line.strip())
                            docs.append("")
                        else:
                            docs.append("*No documentation available.*")
                            docs.append("")
        
        # Functions
        if elements['functions']:
            docs.append("## Functions")
            docs.append("")
            
            for func in elements['functions']:
                docs.append(f"### {func['name']}{func['signature']}")
                docs.append(f"Module: `{func.get('module', package_name)}`")
                docs.append("")
                
                if func['docstring']:
                    docs.append(func['docstring'])
                    docs.append("")
        
        # Classes
        if elements['classes']:
            docs.append("## Classes")
            docs.append("")
            
            for cls in elements['classes']:
                docs.append(f"### {cls['name']}")
                docs.append(f"Module: `{cls.get('module', package_name)}`")
                docs.append("")
                
                if cls['docstring']:
                    docs.append(cls['docstring'])
                    docs.append("")
                
                if cls['methods']:
                    docs.append("#### Methods")
                    docs.append("")
                    for method in cls['methods']:
                        docs.append(f"**`{method['name']}{method['signature']}`**")
                        docs.append("")
                        if method['docstring']:
                            # Format docstring with proper indentation
                            docstring_lines = method['docstring'].split('\n')
                            for line in docstring_lines:
                                docs.append(line.strip())
                            docs.append("")
                        else:
                            docs.append("*No documentation available.*")
                            docs.append("")
        
        # Attributes
        if elements['attributes']:
            docs.append("## Attributes")
            docs.append("")
            
            for attr in elements['attributes']:
                docs.append(f"### {attr['name']}")
                docs.append(f"Type: `{attr['type']}`")
                docs.append(f"Value: `{attr['value']}`")
                docs.append("")
                
                if attr['docstring']:
                    docs.append(attr['docstring'])
                    docs.append("")
        
        # C-Extensions
        if elements['c_extensions']:
            docs.append("## C-Extensions")
            docs.append("")
            
            for ext in elements['c_extensions']:
                docs.append(f"### {ext['name']}")
                docs.append(f"Type: `{ext['type']}`")
                docs.append("")
                
                if ext['docstring']:
                    docs.append(ext['docstring'])
                    docs.append("")
        
        return "\n".join(docs)
    
    def find_packages_with_pydantic(self) -> Dict[str, int]:
        """
        Find all installed packages that contain Pydantic models.
        
        Returns
        -------
        Dict[str, int]
            Dictionary mapping package names to number of Pydantic models found
        """
        packages_with_models = {}
        
        # Get all installed packages
        try:
            installed_packages = [d.metadata['Name'] for d in importlib.metadata.distributions()]
        except Exception as e:
            print(f"Warning: Could not get installed packages: {e}")
            return {}
        
        # Common packages that are likely to have Pydantic models
        priority_packages = [
            'pydantic', 'fastapi', 'sqlmodel', 'pydantic-settings',
            'astroml', 'astropy', 'torch_geometric', 'torch-geometric',
            'transformers', 'datasets', 'accelerate'
        ]
        
        # Check priority packages first
        for package_name in priority_packages:
            if package_name in installed_packages:
                try:
                    elements = self.scan_package(package_name)
                    model_count = len(elements.get('pydantic_models', []))
                    if model_count > 0:
                        packages_with_models[package_name] = model_count
                        print(f"✅ Found {model_count} Pydantic models in {package_name}")
                except Exception as e:
                    print(f"❌ Error scanning {package_name}: {e}")
        
        return packages_with_models
    
    def generate_structured_package_docs(self, package_name: str, output_root: Path, format_type: str = "detailed") -> None:
        """
        Generate structured documentation for a package with submodules in separate files.
        
        Parameters
        ----------
        package_name : str
            Name of the package to document
        output_root : Path
            Root output directory
        format_type : str
            Format type: "detailed" (with JSON), "compact" (markdown only)
        """
        print(f"📝 Generating structured documentation for package {package_name}...")
        
        # Create package directory
        package_dir = output_root / f"package_{package_name}"
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Scan the package
        elements = self.scan_package(package_name)
        
        # Generate main package overview
        overview_docs = self._generate_package_overview(package_name, elements, format_type)
        overview_path = package_dir / "README.md"
        overview_path.write_text(overview_docs, encoding="utf-8")
        print(f"✅ Package overview: {overview_path}")
        
        # Generate documentation for each submodule
        if elements.get('submodules'):
            for submodule in elements['submodules']:
                submodule_docs = self._generate_submodule_docs(package_name, submodule, format_type)
                
                # Create submodule file
                submodule_name = submodule['name']
                submodule_path = package_dir / f"{submodule_name}.md"
                submodule_path.write_text(submodule_docs, encoding="utf-8")
                print(f"✅ Submodule docs: {submodule_path}")
                
                # If submodule has nested submodules, create subdirectory
                if submodule.get('nested_submodules'):
                    nested_dir = package_dir / submodule_name
                    nested_dir.mkdir(parents=True, exist_ok=True)
                    
                    for nested in submodule['nested_submodules']:
                        nested_docs = self._generate_nested_submodule_docs(package_name, submodule, nested, format_type)
                        nested_path = nested_dir / f"{nested['name']}.md"
                        nested_path.write_text(nested_docs, encoding="utf-8")
                        print(f"✅ Nested submodule: {nested_path}")
        
        print(f"🎉 Structured documentation complete for {package_name}!")
        print(f"📁 Documentation available at: {package_dir}")
    
    def _generate_package_overview(self, package_name: str, elements: Dict[str, Any], format_type: str) -> str:
        """Generate the main package overview documentation."""
        docs = []
        docs.append(f"# {package_name.title()} Package")
        docs.append("")
        docs.append(f"Auto-generated structured documentation for package `{package_name}`")
        docs.append("")
        
        # Package info
        if 'package_info' in elements:
            info = elements['package_info']
            docs.append("## Package Information")
            docs.append("")
            docs.append(f"- **Version**: {info.get('version', 'unknown')}")
            docs.append(f"- **Location**: {info.get('location', 'unknown')}")
            if info.get('summary'):
                docs.append(f"- **Summary**: {info['summary']}")
            docs.append("")
        
        # Structure overview
        docs.append("## Package Structure")
        docs.append("")
        
        # Main level elements
        if elements.get('functions'):
            docs.append(f"### Functions ({len(elements['functions'])})")
            docs.append("Functions defined at package level.")
            docs.append("")
        
        if elements.get('classes'):
            docs.append(f"### Classes ({len(elements['classes'])})")
            docs.append("Classes defined at package level.")
            docs.append("")
        
        if elements.get('attributes'):
            docs.append(f"### Attributes ({len(elements['attributes'])})")
            docs.append("Package-level attributes and constants.")
            docs.append("")
        
        # Submodules overview
        if elements.get('submodules'):
            docs.append(f"## Submodules ({len(elements['submodules'])})")
            docs.append("")
            docs.append("Each submodule is documented in a separate file:")
            docs.append("")
            
            for submodule in elements['submodules']:
                docs.append(f"### [{submodule['name']}](./{submodule['name']}.md)")
                docs.append(f"Module: `{submodule['module_name']}`")
                docs.append("")
                
                if submodule['docstring']:
                    # Take first line of docstring
                    first_line = submodule['docstring'].split('\n')[0]
                    docs.append(first_line)
                    docs.append("")
                
                # Show counts
                counts = []
                if submodule.get('functions'):
                    counts.append(f"{len(submodule['functions'])} functions")
                if submodule.get('classes'):
                    counts.append(f"{len(submodule['classes'])} classes")
                if submodule.get('important_types'):
                    counts.append(f"{len(submodule['important_types'])} important types")
                if submodule.get('nested_submodules'):
                    counts.append(f"{len(submodule['nested_submodules'])} nested submodules")
                
                if counts:
                    docs.append(f"*Contains: {', '.join(counts)}*")
                    docs.append("")
        
        return "\n".join(docs)
    
    def _generate_submodule_docs(self, package_name: str, submodule: Dict[str, Any], format_type: str) -> str:
        """Generate documentation for a single submodule."""
        docs = []
        docs.append(f"# {submodule['name']} Submodule")
        docs.append("")
        docs.append(f"Part of the `{package_name}` package")
        docs.append(f"Module: `{submodule['module_name']}`")
        docs.append("")
        
        if submodule['docstring']:
            docs.append("## Description")
            docs.append("")
            docs.append(submodule['docstring'])
            docs.append("")
        
        # Functions
        if submodule.get('functions'):
            docs.append(f"## Functions ({len(submodule['functions'])})")
            docs.append("")
            for func in submodule['functions']:
                docs.append(f"### `{func['name']}{func['signature']}`")
                docs.append("")
                if func['docstring']:
                    docs.append(func['docstring'])
                    docs.append("")
        
        # Important types
        if submodule.get('important_types'):
            docs.append(f"## Important Data Types ({len(submodule['important_types'])})")
            docs.append("")
            for itype in submodule['important_types']:
                docs.append(f"### `{itype['name']}`")
                docs.append(f"**Type**: `{itype['type']}`")
                docs.append("")
                
                if itype['docstring']:
                    docs.append(itype['docstring'])
                    docs.append("")
                
                # Add additional info
                info_parts = []
                if itype.get('has_methods'):
                    info_parts.append("has methods")
                if itype.get('is_callable'):
                    info_parts.append("callable")
                
                if info_parts:
                    docs.append(f"*({', '.join(info_parts)})*")
                    docs.append("")
        
        # Classes
        if submodule.get('classes'):
            docs.append(f"## Classes ({len(submodule['classes'])})")
            docs.append("")
            for cls in submodule['classes']:
                docs.append(f"### `{cls['name']}`")
                docs.append("")
                
                if cls['docstring']:
                    docs.append(cls['docstring'])
                    docs.append("")
                
                if cls.get('methods'):
                    docs.append("#### Methods")
                    docs.append("")
                    for method in cls['methods']:
                        docs.append(f"- **`{method['name']}{method['signature']}`**")
                        if method['docstring']:
                            # Take first line
                            first_line = method['docstring'].split('\n')[0]
                            docs.append(f"  {first_line}")
                        docs.append("")
        
        # Nested submodules
        if submodule.get('nested_submodules'):
            docs.append(f"## Nested Submodules ({len(submodule['nested_submodules'])})")
            docs.append("")
            docs.append("Each nested submodule is documented in a separate file:")
            docs.append("")
            
            for nested in submodule['nested_submodules']:
                docs.append(f"### [{nested['name']}](./{submodule['name']}/{nested['name']}.md)")
                docs.append(f"Module: `{nested['module_name']}`")
                docs.append("")
                
                # Show what it contains
                contains = []
                if nested.get('functions'):
                    contains.append(f"{len(nested['functions'])} functions")
                if nested.get('classes'):
                    contains.append(f"{len(nested['classes'])} classes")
                if nested.get('operators'):
                    contains.append(f"{len(nested['operators'])} operators")
                
                if contains:
                    docs.append(f"*Contains: {', '.join(contains)}*")
                    docs.append("")
        
        return "\n".join(docs)
    
    def _generate_nested_submodule_docs(self, package_name: str, parent_submodule: Dict[str, Any], 
                                      nested: Dict[str, Any], format_type: str) -> str:
        """Generate documentation for a nested submodule."""
        docs = []
        docs.append(f"# {nested['name']}")
        docs.append("")
        docs.append(f"Part of `{package_name}.{parent_submodule['name']}`")
        docs.append(f"Module: `{nested['module_name']}`")
        docs.append("")
        
        if nested['docstring']:
            docs.append("## Description")
            docs.append("")
            docs.append(nested['docstring'])
            docs.append("")
        
        # Operators (special for bpy.ops)
        if nested.get('operators'):
            docs.append(f"## Operators ({len(nested['operators'])})")
            docs.append("")
            for op in nested['operators']:
                docs.append(f"### `{op['name']}`")
                docs.append("")
                if op['docstring']:
                    docs.append(op['docstring'])
                    docs.append("")
        
        # Functions
        if nested.get('functions'):
            docs.append(f"## Functions ({len(nested['functions'])})")
            docs.append("")
            for func in nested['functions']:
                docs.append(f"### `{func['name']}{func['signature']}`")
                docs.append("")
                if func['docstring']:
                    docs.append(func['docstring'])
                    docs.append("")
        
        # Classes
        if nested.get('classes'):
            docs.append(f"## Classes ({len(nested['classes'])})")
            docs.append("")
            for cls in nested['classes']:
                docs.append(f"### `{cls['name']}`")
                docs.append("")
                
                if cls['docstring']:
                    docs.append(cls['docstring'])
                    docs.append("")
                
                if cls.get('methods'):
                    docs.append("#### Methods")
                    docs.append("")
                    for method in cls['methods']:
                        docs.append(f"- **`{method['name']}{method['signature']}`**")
                        if method['docstring']:
                            first_line = method['docstring'].split('\n')[0]
                            docs.append(f"  {first_line}")
                        docs.append("")
        
        return "\n".join(docs) 