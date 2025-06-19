# Data_Schemas Module

Auto-generated documentation for `schemas.data_schemas`

## Pydantic Models

## DataLoaderConfigSchema

Configuration schema for PyTorch Geometric DataLoaders.

### Parameters

**batch_size** (integer)
: Batch size for data loading
 (default: `32`)
 (min: 1, max: 10000)

**shuffle** (boolean)
: Whether to shuffle the data
 (default: `True`)

**num_workers** (integer)
: Number of worker processes for data loading
 (default: `0`)
 (min: 0, max: 16)

**pin_memory** (boolean)
: Whether to pin memory for GPU transfer
 (default: `True`)

**use_gpu_optimization** (boolean)
: Whether to use GPU optimization if available
 (default: `True`)

### JSON Schema

```json
{
  "description": "Configuration schema for PyTorch Geometric DataLoaders.",
  "properties": {
    "batch_size": {
      "default": 32,
      "description": "Batch size for data loading",
      "maximum": 10000,
      "minimum": 1,
      "title": "Batch Size",
      "type": "integer"
    },
    "shuffle": {
      "default": true,
      "description": "Whether to shuffle the data",
      "title": "Shuffle",
      "type": "boolean"
    },
    "num_workers": {
      "default": 0,
      "description": "Number of worker processes for data loading",
      "maximum": 16,
      "minimum": 0,
      "title": "Num Workers",
      "type": "integer"
    },
    "pin_memory": {
      "default": true,
      "description": "Whether to pin memory for GPU transfer",
      "title": "Pin Memory",
      "type": "boolean"
    },
    "use_gpu_optimization": {
      "default": true,
      "description": "Whether to use GPU optimization if available",
      "title": "Use Gpu Optimization",
      "type": "boolean"
    }
  },
  "title": "DataLoaderConfigSchema",
  "type": "object"
}
```

## DatasetConfigSchema

Base configuration schema for all datasets.

### Parameters

**root** (Optional[string, null])
: Root directory for dataset files
 (default: `None`)

**transform** (Optional[string, null])
: Transform to apply to each sample
 (default: `None`)

**pre_transform** (Optional[string, null])
: Transform to apply before saving
 (default: `None`)

**pre_filter** (Optional[string, null])
: Filter to apply before saving
 (default: `None`)

### JSON Schema

```json
{
  "description": "Base configuration schema for all datasets.",
  "properties": {
    "root": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Root directory for dataset files",
      "title": "Root"
    },
    "transform": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Transform to apply to each sample",
      "title": "Transform"
    },
    "pre_transform": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Transform to apply before saving",
      "title": "Pre Transform"
    },
    "pre_filter": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Filter to apply before saving",
      "title": "Pre Filter"
    }
  },
  "title": "DatasetConfigSchema",
  "type": "object"
}
```

## ExoplanetDatasetConfigSchema

Configuration schema for ExoplanetGraphDataset.

### Parameters

**root** (Optional[string, null])
: Root directory for dataset files
 (default: `None`)

**transform** (Optional[string, null])
: Transform to apply to each sample
 (default: `None`)

**pre_transform** (Optional[string, null])
: Transform to apply before saving
 (default: `None`)

**pre_filter** (Optional[string, null])
: Filter to apply before saving
 (default: `None`)

**k_neighbors** (integer)
: Number of nearest neighbors for graph construction
 (default: `5`)
 (min: 1, max: 50)

**max_distance** (number)
: Maximum distance for connections (parsecs)
 (default: `100.0`)

### JSON Schema

```json
{
  "description": "Configuration schema for ExoplanetGraphDataset.",
  "properties": {
    "root": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Root directory for dataset files",
      "title": "Root"
    },
    "transform": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Transform to apply to each sample",
      "title": "Transform"
    },
    "pre_transform": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Transform to apply before saving",
      "title": "Pre Transform"
    },
    "pre_filter": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Filter to apply before saving",
      "title": "Pre Filter"
    },
    "k_neighbors": {
      "default": 5,
      "description": "Number of nearest neighbors for graph construction",
      "maximum": 50,
      "minimum": 1,
      "title": "K Neighbors",
      "type": "integer"
    },
    "max_distance": {
      "default": 100.0,
      "description": "Maximum distance for connections (parsecs)",
      "exclusiveMinimum": 0.0,
      "title": "Max Distance",
      "type": "number"
    }
  },
  "title": "ExoplanetDatasetConfigSchema",
  "type": "object"
}
```

## GaiaDatasetConfigSchema

Configuration schema for GaiaGraphDataset.

### Parameters

**root** (Optional[string, null])
: Root directory for dataset files
 (default: `None`)

**transform** (Optional[string, null])
: Transform to apply to each sample
 (default: `None`)

**pre_transform** (Optional[string, null])
: Transform to apply before saving
 (default: `None`)

**pre_filter** (Optional[string, null])
: Filter to apply before saving
 (default: `None`)

**magnitude_limit** (number)
: Magnitude limit for star selection
 (default: `12.0`)
 (min: 5.0, max: 20.0)

**k_neighbors** (integer)
: Number of nearest neighbors for graph construction
 (default: `8`)
 (min: 1, max: 50)

**max_distance** (number)
: Maximum distance for connections (kpc)
 (default: `1.0`)

### JSON Schema

```json
{
  "description": "Configuration schema for GaiaGraphDataset.",
  "properties": {
    "root": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Root directory for dataset files",
      "title": "Root"
    },
    "transform": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Transform to apply to each sample",
      "title": "Transform"
    },
    "pre_transform": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Transform to apply before saving",
      "title": "Pre Transform"
    },
    "pre_filter": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Filter to apply before saving",
      "title": "Pre Filter"
    },
    "magnitude_limit": {
      "default": 12.0,
      "description": "Magnitude limit for star selection",
      "maximum": 20.0,
      "minimum": 5.0,
      "title": "Magnitude Limit",
      "type": "number"
    },
    "k_neighbors": {
      "default": 8,
      "description": "Number of nearest neighbors for graph construction",
      "maximum": 50,
      "minimum": 1,
      "title": "K Neighbors",
      "type": "integer"
    },
    "max_distance": {
      "default": 1.0,
      "description": "Maximum distance for connections (kpc)",
      "exclusiveMinimum": 0.0,
      "title": "Max Distance",
      "type": "number"
    }
  },
  "title": "GaiaDatasetConfigSchema",
  "type": "object"
}
```

## NSADatasetConfigSchema

Configuration schema for NSAGraphDataset.

### Parameters

**root** (Optional[string, null])
: Root directory for dataset files
 (default: `None`)

**transform** (Optional[string, null])
: Transform to apply to each sample
 (default: `None`)

**pre_transform** (Optional[string, null])
: Transform to apply before saving
 (default: `None`)

**pre_filter** (Optional[string, null])
: Filter to apply before saving
 (default: `None`)

**max_galaxies** (integer)
: Maximum number of galaxies to include
 (default: `10000`)
 (min: 10, max: 1000000)

**k_neighbors** (integer)
: Number of nearest neighbors for graph construction
 (default: `8`)
 (min: 1, max: 50)

**distance_threshold** (number)
: Distance threshold for connections (Mpc)
 (default: `50.0`)

### JSON Schema

```json
{
  "description": "Configuration schema for NSAGraphDataset.",
  "properties": {
    "root": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Root directory for dataset files",
      "title": "Root"
    },
    "transform": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Transform to apply to each sample",
      "title": "Transform"
    },
    "pre_transform": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Transform to apply before saving",
      "title": "Pre Transform"
    },
    "pre_filter": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Filter to apply before saving",
      "title": "Pre Filter"
    },
    "max_galaxies": {
      "default": 10000,
      "description": "Maximum number of galaxies to include",
      "maximum": 1000000,
      "minimum": 10,
      "title": "Max Galaxies",
      "type": "integer"
    },
    "k_neighbors": {
      "default": 8,
      "description": "Number of nearest neighbors for graph construction",
      "maximum": 50,
      "minimum": 1,
      "title": "K Neighbors",
      "type": "integer"
    },
    "distance_threshold": {
      "default": 50.0,
      "description": "Distance threshold for connections (Mpc)",
      "exclusiveMinimum": 0.0,
      "title": "Distance Threshold",
      "type": "number"
    }
  },
  "title": "NSADatasetConfigSchema",
  "type": "object"
}
```

## ProcessingConfigSchema

Configuration schema for data processing.

### Parameters

**device** (string)
: Device for tensor operations (auto, cpu, cuda, mps)
 (default: `auto`)

**batch_size** (integer)
: Batch size for processing
 (default: `32`)
 (min: 1, max: 10000)

**max_samples** (Optional[object, null])
: Maximum samples per survey
 (default: `None`)

**surveys** (Optional[array, null])
: List of surveys to process
 (default: `None`)

### JSON Schema

```json
{
  "description": "Configuration schema for data processing.",
  "properties": {
    "device": {
      "default": "auto",
      "description": "Device for tensor operations (auto, cpu, cuda, mps)",
      "title": "Device",
      "type": "string"
    },
    "batch_size": {
      "default": 32,
      "description": "Batch size for processing",
      "maximum": 10000,
      "minimum": 1,
      "title": "Batch Size",
      "type": "integer"
    },
    "max_samples": {
      "anyOf": [
        {
          "additionalProperties": {
            "type": "integer"
          },
          "type": "object"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Maximum samples per survey",
      "title": "Max Samples"
    },
    "surveys": {
      "anyOf": [
        {
          "items": {
            "type": "string"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "List of surveys to process",
      "title": "Surveys"
    }
  },
  "title": "ProcessingConfigSchema",
  "type": "object"
}
```

## Functions

### Field(default: 'Any' = PydanticUndefined, *, default_factory: 'Callable[[], Any] | Callable[[dict[str, Any]], Any] | None' = PydanticUndefined, alias: 'str | None' = PydanticUndefined, alias_priority: 'int | None' = PydanticUndefined, validation_alias: 'str | AliasPath | AliasChoices | None' = PydanticUndefined, serialization_alias: 'str | None' = PydanticUndefined, title: 'str | None' = PydanticUndefined, field_title_generator: 'Callable[[str, FieldInfo], str] | None' = PydanticUndefined, description: 'str | None' = PydanticUndefined, examples: 'list[Any] | None' = PydanticUndefined, exclude: 'bool | None' = PydanticUndefined, discriminator: 'str | types.Discriminator | None' = PydanticUndefined, deprecated: 'Deprecated | str | bool | None' = PydanticUndefined, json_schema_extra: 'JsonDict | Callable[[JsonDict], None] | None' = PydanticUndefined, frozen: 'bool | None' = PydanticUndefined, validate_default: 'bool | None' = PydanticUndefined, repr: 'bool' = PydanticUndefined, init: 'bool | None' = PydanticUndefined, init_var: 'bool | None' = PydanticUndefined, kw_only: 'bool | None' = PydanticUndefined, pattern: 'str | typing.Pattern[str] | None' = PydanticUndefined, strict: 'bool | None' = PydanticUndefined, coerce_numbers_to_str: 'bool | None' = PydanticUndefined, gt: 'annotated_types.SupportsGt | None' = PydanticUndefined, ge: 'annotated_types.SupportsGe | None' = PydanticUndefined, lt: 'annotated_types.SupportsLt | None' = PydanticUndefined, le: 'annotated_types.SupportsLe | None' = PydanticUndefined, multiple_of: 'float | None' = PydanticUndefined, allow_inf_nan: 'bool | None' = PydanticUndefined, max_digits: 'int | None' = PydanticUndefined, decimal_places: 'int | None' = PydanticUndefined, min_length: 'int | None' = PydanticUndefined, max_length: 'int | None' = PydanticUndefined, union_mode: "Literal['smart', 'left_to_right']" = PydanticUndefined, fail_fast: 'bool | None' = PydanticUndefined, **extra: 'Unpack[_EmptyKwargs]') -> 'Any'

!!! abstract "Usage Documentation"
    [Fields](../concepts/fields.md)

Create a field for objects that can be configured.

Used to provide extra information about a field, either for the model schema or complex validation. Some arguments
apply only to number fields (`int`, `float`, `Decimal`) and some apply only to `str`.

Note:
    - Any `_Unset` objects will be replaced by the corresponding value defined in the `_DefaultValues` dictionary. If a key for the `_Unset` object is not found in the `_DefaultValues` dictionary, it will default to `None`

Args:
    default: Default value if the field is not set.
    default_factory: A callable to generate the default value. The callable can either take 0 arguments
        (in which case it is called as is) or a single argument containing the already validated data.
    alias: The name to use for the attribute when validating or serializing by alias.
        This is often used for things like converting between snake and camel case.
    alias_priority: Priority of the alias. This affects whether an alias generator is used.
    validation_alias: Like `alias`, but only affects validation, not serialization.
    serialization_alias: Like `alias`, but only affects serialization, not validation.
    title: Human-readable title.
    field_title_generator: A callable that takes a field name and returns title for it.
    description: Human-readable description.
    examples: Example values for this field.
    exclude: Whether to exclude the field from the model serialization.
    discriminator: Field name or Discriminator for discriminating the type in a tagged union.
    deprecated: A deprecation message, an instance of `warnings.deprecated` or the `typing_extensions.deprecated` backport,
        or a boolean. If `True`, a default deprecation message will be emitted when accessing the field.
    json_schema_extra: A dict or callable to provide extra JSON schema properties.
    frozen: Whether the field is frozen. If true, attempts to change the value on an instance will raise an error.
    validate_default: If `True`, apply validation to the default value every time you create an instance.
        Otherwise, for performance reasons, the default value of the field is trusted and not validated.
    repr: A boolean indicating whether to include the field in the `__repr__` output.
    init: Whether the field should be included in the constructor of the dataclass.
        (Only applies to dataclasses.)
    init_var: Whether the field should _only_ be included in the constructor of the dataclass.
        (Only applies to dataclasses.)
    kw_only: Whether the field should be a keyword-only argument in the constructor of the dataclass.
        (Only applies to dataclasses.)
    coerce_numbers_to_str: Whether to enable coercion of any `Number` type to `str` (not applicable in `strict` mode).
    strict: If `True`, strict validation is applied to the field.
        See [Strict Mode](../concepts/strict_mode.md) for details.
    gt: Greater than. If set, value must be greater than this. Only applicable to numbers.
    ge: Greater than or equal. If set, value must be greater than or equal to this. Only applicable to numbers.
    lt: Less than. If set, value must be less than this. Only applicable to numbers.
    le: Less than or equal. If set, value must be less than or equal to this. Only applicable to numbers.
    multiple_of: Value must be a multiple of this. Only applicable to numbers.
    min_length: Minimum length for iterables.
    max_length: Maximum length for iterables.
    pattern: Pattern for strings (a regular expression).
    allow_inf_nan: Allow `inf`, `-inf`, `nan`. Only applicable to float and [`Decimal`][decimal.Decimal] numbers.
    max_digits: Maximum number of allow digits for strings.
    decimal_places: Maximum number of decimal places allowed for numbers.
    union_mode: The strategy to apply when validating a union. Can be `smart` (the default), or `left_to_right`.
        See [Union Mode](../concepts/unions.md#union-modes) for details.
    fail_fast: If `True`, validation will stop on the first error. If `False`, all validation errors will be collected.
        This option can be applied only to iterable types (list, tuple, set, and frozenset).
    extra: (Deprecated) Extra fields that will be included in the JSON schema.

        !!! warning Deprecated
            The `extra` kwargs is deprecated. Use `json_schema_extra` instead.

Returns:
    A new [`FieldInfo`][pydantic.fields.FieldInfo]. The return annotation is `Any` so `Field` can be used on
        type-annotated fields without causing a type error.
