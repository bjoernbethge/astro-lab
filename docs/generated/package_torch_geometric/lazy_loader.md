# lazy_loader Submodule

Part of the `torch_geometric` package
Module: `torch_geometric.lazy_loader`

## Functions (1)

### `import_module(name, package=None)`

Import a module.

The 'package' argument is required when performing a relative import. It
specifies the package to use as the anchor point from which to resolve the
relative import to an absolute import.

## Important Data Types (5)

### `Any`
**Type**: `<class 'typing._AnyMeta'>`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

*(has methods, callable)*

### `Dict`
**Type**: `<class 'typing._SpecialGenericAlias'>`

A generic version of dict.

*(has methods, callable)*

### `List`
**Type**: `<class 'typing._SpecialGenericAlias'>`

A generic version of list.

*(has methods, callable)*

### `LazyLoader`
**Type**: `<class 'type'>`

Create a module object.

The name must be a string; the optional doc argument can have any type.

*(has methods, callable)*

### `ModuleType`
**Type**: `<class 'type'>`

Create a module object.

The name must be a string; the optional doc argument can have any type.

*(has methods, callable)*

## Classes (3)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `LazyLoader`

Create a module object.

The name must be a string; the optional doc argument can have any type.

### `ModuleType`

Create a module object.

The name must be a string; the optional doc argument can have any type.
