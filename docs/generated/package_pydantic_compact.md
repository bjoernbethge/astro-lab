# Pydantic Package Documentation

Auto-generated documentation for installed package `pydantic`

## Package Information

- **Version**: 2.11.7
- **Location**: D:\astro-lab\.venv\Lib\site-packages
- **Summary**: Data validation using Python type hints

## Submodules

### dataclasses
Module: `pydantic.dataclasses`

Provide an enhanced dataclass that performs validation.

## RootModel

!!! abstract "Usage Documentation"
    [`RootModel` and Custom Root Types](../concepts/models.md#rootmodel-and-custom-root-types)

A Pydantic `BaseModel` for the root object of the model.

Attributes:
    root: The root object of the model.
    __pydantic_root_model__: Whether the model is a RootModel.
    __pydantic_private__: Private fields in the model.
    __pydantic_extra__: Extra fields in the model.

### Usage

```python
from docs.auto.schemas.data_schemas import RootModel

config = RootModel()
```

## Pydantic Model Methods

## Functions

### Field(default: 'Any' = PydanticUndefined, *, default_factory: 'Callable[[], Any] | Callable[[dict[str, Any]], Any] | None' = PydanticUndefined, alias: 'str | None' = PydanticUndefined, alias_priority: 'int | None' = PydanticUndefined, validation_alias: 'str | AliasPath | AliasChoices | None' = PydanticUndefined, serialization_alias: 'str | None' = PydanticUndefined, title: 'str | None' = PydanticUndefined, field_title_generator: 'Callable[[str, FieldInfo], str] | None' = PydanticUndefined, description: 'str | None' = PydanticUndefined, examples: 'list[Any] | None' = PydanticUndefined, exclude: 'bool | None' = PydanticUndefined, discriminator: 'str | types.Discriminator | None' = PydanticUndefined, deprecated: 'Deprecated | str | bool | None' = PydanticUndefined, json_schema_extra: 'JsonDict | Callable[[JsonDict], None] | None' = PydanticUndefined, frozen: 'bool | None' = PydanticUndefined, validate_default: 'bool | None' = PydanticUndefined, repr: 'bool' = PydanticUndefined, init: 'bool | None' = PydanticUndefined, init_var: 'bool | None' = PydanticUndefined, kw_only: 'bool | None' = PydanticUndefined, pattern: 'str | typing.Pattern[str] | None' = PydanticUndefined, strict: 'bool | None' = PydanticUndefined, coerce_numbers_to_str: 'bool | None' = PydanticUndefined, gt: 'annotated_types.SupportsGt | None' = PydanticUndefined, ge: 'annotated_types.SupportsGe | None' = PydanticUndefined, lt: 'annotated_types.SupportsLt | None' = PydanticUndefined, le: 'annotated_types.SupportsLe | None' = PydanticUndefined, multiple_of: 'float | None' = PydanticUndefined, allow_inf_nan: 'bool | None' = PydanticUndefined, max_digits: 'int | None' = PydanticUndefined, decimal_places: 'int | None' = PydanticUndefined, min_length: 'int | None' = PydanticUndefined, max_length: 'int | None' = PydanticUndefined, union_mode: "Literal['smart', 'left_to_right']" = PydanticUndefined, fail_fast: 'bool | None' = PydanticUndefined, **extra: 'Unpack[_EmptyKwargs]') -> 'Any'
Module: `pydantic.fields`

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

### PrivateAttr(default: 'Any' = PydanticUndefined, *, default_factory: 'Callable[[], Any] | None' = None, init: 'Literal[False]' = False) -> 'Any'
Module: `pydantic.fields`

!!! abstract "Usage Documentation"
    [Private Model Attributes](../concepts/models.md#private-model-attributes)

Indicates that an attribute is intended for private use and not handled during normal validation/serialization.

Private attributes are not validated by Pydantic, so it's up to you to ensure they are used in a type-safe manner.

Private attributes are stored in `__private_attributes__` on the model.

Args:
    default: The attribute's default value. Defaults to Undefined.
    default_factory: Callable that will be
        called when a default value is needed for this attribute.
        If both `default` and `default_factory` are set, an error will be raised.
    init: Whether the attribute should be included in the constructor of the dataclass. Always `False`.

Returns:
    An instance of [`ModelPrivateAttr`][pydantic.fields.ModelPrivateAttr] class.

Raises:
    ValueError: If both `default` and `default_factory` are set.

### computed_field(func: 'PropertyT | None' = None, /, *, alias: 'str | None' = None, alias_priority: 'int | None' = None, title: 'str | None' = None, field_title_generator: 'typing.Callable[[str, ComputedFieldInfo], str] | None' = None, description: 'str | None' = None, deprecated: 'Deprecated | str | bool | None' = None, examples: 'list[Any] | None' = None, json_schema_extra: 'JsonDict | typing.Callable[[JsonDict], None] | None' = None, repr: 'bool | None' = None, return_type: 'Any' = PydanticUndefined) -> 'PropertyT | typing.Callable[[PropertyT], PropertyT]'
Module: `pydantic.fields`

!!! abstract "Usage Documentation"
    [The `computed_field` decorator](../concepts/fields.md#the-computed_field-decorator)

Decorator to include `property` and `cached_property` when serializing models or dataclasses.

This is useful for fields that are computed from other fields, or for fields that are expensive to compute and should be cached.

```python
from pydantic import BaseModel, computed_field

class Rectangle(BaseModel):
    width: int
    length: int

    @computed_field
    @property
    def area(self) -> int:
        return self.width * self.length

print(Rectangle(width=3, length=2).model_dump())
#> {'width': 3, 'length': 2, 'area': 6}
```

If applied to functions not yet decorated with `@property` or `@cached_property`, the function is
automatically wrapped with `property`. Although this is more concise, you will lose IntelliSense in your IDE,
and confuse static type checkers, thus explicit use of `@property` is recommended.

!!! warning "Mypy Warning"
    Even with the `@property` or `@cached_property` applied to your function before `@computed_field`,
    mypy may throw a `Decorated property not supported` error.
    See [mypy issue #1362](https://github.com/python/mypy/issues/1362), for more information.
    To avoid this error message, add `# type: ignore[prop-decorator]` to the `@computed_field` line.

    [pyright](https://github.com/microsoft/pyright) supports `@computed_field` without error.

```python
import random

from pydantic import BaseModel, computed_field

class Square(BaseModel):
    width: float

    @computed_field
    def area(self) -> float:  # converted to a `property` by `computed_field`
        return round(self.width**2, 2)

    @area.setter
    def area(self, new_area: float) -> None:
        self.width = new_area**0.5

    @computed_field(alias='the magic number', repr=False)
    def random_number(self) -> int:
        return random.randint(0, 1_000)

square = Square(width=1.3)

# `random_number` does not appear in representation
print(repr(square))
#> Square(width=1.3, area=1.69)

print(square.random_number)
#> 3

square.area = 4

print(square.model_dump_json(by_alias=True))
#> {"width":2.0,"area":4.0,"the magic number":3}
```

!!! warning "Overriding with `computed_field`"
    You can't override a field from a parent class with a `computed_field` in the child class.
    `mypy` complains about this behavior if allowed, and `dataclasses` doesn't allow this pattern either.
    See the example below:

```python
from pydantic import BaseModel, computed_field

class Parent(BaseModel):
    a: str

try:

    class Child(Parent):
        @computed_field
        @property
        def a(self) -> str:
            return 'new a'

except TypeError as e:
    print(e)
    '''
    Field 'a' of class 'Child' overrides symbol of same name in a parent class. This override with a computed_field is incompatible.
    '''
```

Private properties decorated with `@computed_field` have `repr=False` by default.

```python
from functools import cached_property

from pydantic import BaseModel, computed_field

class Model(BaseModel):
    foo: int

    @computed_field
    @cached_property
    def _private_cached_property(self) -> int:
        return -self.foo

    @computed_field
    @property
    def _private_property(self) -> int:
        return -self.foo

m = Model(foo=1)
print(repr(m))
#> Model(foo=1)
```

Args:
    func: the function to wrap.
    alias: alias to use when serializing this computed field, only used when `by_alias=True`
    alias_priority: priority of the alias. This affects whether an alias generator is used
    title: Title to use when including this computed field in JSON Schema
    field_title_generator: A callable that takes a field name and returns title for it.
    description: Description to use when including this computed field in JSON Schema, defaults to the function's
        docstring
    deprecated: A deprecation message (or an instance of `warnings.deprecated` or the `typing_extensions.deprecated` backport).
        to be emitted when accessing the field. Or a boolean. This will automatically be set if the property is decorated with the
        `deprecated` decorator.
    examples: Example values to use when including this computed field in JSON Schema
    json_schema_extra: A dict or callable to provide extra JSON schema properties.
    repr: whether to include this computed field in model repr.
        Default is `False` for private properties and `True` for public properties.
    return_type: optional return for serialization logic to expect when serializing to JSON, if included
        this must be correct, otherwise a `TypeError` is raised.
        If you don't include a return type Any is used, which does runtime introspection to handle arbitrary
        objects.

Returns:
    A proxy wrapper for the property.

### conbytes(*, min_length: 'int | None' = None, max_length: 'int | None' = None, strict: 'bool | None' = None) -> 'type[bytes]'
Module: `pydantic.types`

A wrapper around `bytes` that allows for additional constraints.

Args:
    min_length: The minimum length of the bytes.
    max_length: The maximum length of the bytes.
    strict: Whether to validate the bytes in strict mode.

Returns:
    The wrapped bytes type.

### condate(*, strict: 'bool | None' = None, gt: 'date | None' = None, ge: 'date | None' = None, lt: 'date | None' = None, le: 'date | None' = None) -> 'type[date]'
Module: `pydantic.types`

A wrapper for date that adds constraints.

Args:
    strict: Whether to validate the date value in strict mode. Defaults to `None`.
    gt: The value must be greater than this. Defaults to `None`.
    ge: The value must be greater than or equal to this. Defaults to `None`.
    lt: The value must be less than this. Defaults to `None`.
    le: The value must be less than or equal to this. Defaults to `None`.

Returns:
    A date type with the specified constraints.

### condecimal(*, strict: 'bool | None' = None, gt: 'int | Decimal | None' = None, ge: 'int | Decimal | None' = None, lt: 'int | Decimal | None' = None, le: 'int | Decimal | None' = None, multiple_of: 'int | Decimal | None' = None, max_digits: 'int | None' = None, decimal_places: 'int | None' = None, allow_inf_nan: 'bool | None' = None) -> 'type[Decimal]'
Module: `pydantic.types`

!!! warning "Discouraged"
    This function is **discouraged** in favor of using
    [`Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated) with
    [`Field`][pydantic.fields.Field] instead.

    This function will be **deprecated** in Pydantic 3.0.

    The reason is that `condecimal` returns a type, which doesn't play well with static analysis tools.

    === ":x: Don't do this"
        ```python
        from pydantic import BaseModel, condecimal

        class Foo(BaseModel):
            bar: condecimal(strict=True, allow_inf_nan=True)
        ```

    === ":white_check_mark: Do this"
        ```python
        from decimal import Decimal
        from typing import Annotated

        from pydantic import BaseModel, Field

        class Foo(BaseModel):
            bar: Annotated[Decimal, Field(strict=True, allow_inf_nan=True)]
        ```

A wrapper around Decimal that adds validation.

Args:
    strict: Whether to validate the value in strict mode. Defaults to `None`.
    gt: The value must be greater than this. Defaults to `None`.
    ge: The value must be greater than or equal to this. Defaults to `None`.
    lt: The value must be less than this. Defaults to `None`.
    le: The value must be less than or equal to this. Defaults to `None`.
    multiple_of: The value must be a multiple of this. Defaults to `None`.
    max_digits: The maximum number of digits. Defaults to `None`.
    decimal_places: The number of decimal places. Defaults to `None`.
    allow_inf_nan: Whether to allow infinity and NaN. Defaults to `None`.

```python
from decimal import Decimal

from pydantic import BaseModel, ValidationError, condecimal

class ConstrainedExample(BaseModel):
    constrained_decimal: condecimal(gt=Decimal('1.0'))

m = ConstrainedExample(constrained_decimal=Decimal('1.1'))
print(repr(m))
#> ConstrainedExample(constrained_decimal=Decimal('1.1'))

try:
    ConstrainedExample(constrained_decimal=Decimal('0.9'))
except ValidationError as e:
    print(e.errors())
    '''
    [
        {
            'type': 'greater_than',
            'loc': ('constrained_decimal',),
            'msg': 'Input should be greater than 1.0',
            'input': Decimal('0.9'),
            'ctx': {'gt': Decimal('1.0')},
            'url': 'https://errors.pydantic.dev/2/v/greater_than',
        }
    ]
    '''
```

### confloat(*, strict: 'bool | None' = None, gt: 'float | None' = None, ge: 'float | None' = None, lt: 'float | None' = None, le: 'float | None' = None, multiple_of: 'float | None' = None, allow_inf_nan: 'bool | None' = None) -> 'type[float]'
Module: `pydantic.types`

!!! warning "Discouraged"
    This function is **discouraged** in favor of using
    [`Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated) with
    [`Field`][pydantic.fields.Field] instead.

    This function will be **deprecated** in Pydantic 3.0.

    The reason is that `confloat` returns a type, which doesn't play well with static analysis tools.

    === ":x: Don't do this"
        ```python
        from pydantic import BaseModel, confloat

        class Foo(BaseModel):
            bar: confloat(strict=True, gt=0)
        ```

    === ":white_check_mark: Do this"
        ```python
        from typing import Annotated

        from pydantic import BaseModel, Field

        class Foo(BaseModel):
            bar: Annotated[float, Field(strict=True, gt=0)]
        ```

A wrapper around `float` that allows for additional constraints.

Args:
    strict: Whether to validate the float in strict mode.
    gt: The value must be greater than this.
    ge: The value must be greater than or equal to this.
    lt: The value must be less than this.
    le: The value must be less than or equal to this.
    multiple_of: The value must be a multiple of this.
    allow_inf_nan: Whether to allow `-inf`, `inf`, and `nan`.

Returns:
    The wrapped float type.

```python
from pydantic import BaseModel, ValidationError, confloat

class ConstrainedExample(BaseModel):
    constrained_float: confloat(gt=1.0)

m = ConstrainedExample(constrained_float=1.1)
print(repr(m))
#> ConstrainedExample(constrained_float=1.1)

try:
    ConstrainedExample(constrained_float=0.9)
except ValidationError as e:
    print(e.errors())
    '''
    [
        {
            'type': 'greater_than',
            'loc': ('constrained_float',),
            'msg': 'Input should be greater than 1',
            'input': 0.9,
            'ctx': {'gt': 1.0},
            'url': 'https://errors.pydantic.dev/2/v/greater_than',
        }
    ]
    '''
```

### confrozenset(item_type: 'type[HashableItemType]', *, min_length: 'int | None' = None, max_length: 'int | None' = None) -> 'type[frozenset[HashableItemType]]'
Module: `pydantic.types`

A wrapper around `typing.FrozenSet` that allows for additional constraints.

Args:
    item_type: The type of the items in the frozenset.
    min_length: The minimum length of the frozenset.
    max_length: The maximum length of the frozenset.

Returns:
    The wrapped frozenset type.

### conint(*, strict: 'bool | None' = None, gt: 'int | None' = None, ge: 'int | None' = None, lt: 'int | None' = None, le: 'int | None' = None, multiple_of: 'int | None' = None) -> 'type[int]'
Module: `pydantic.types`

!!! warning "Discouraged"
    This function is **discouraged** in favor of using
    [`Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated) with
    [`Field`][pydantic.fields.Field] instead.

    This function will be **deprecated** in Pydantic 3.0.

    The reason is that `conint` returns a type, which doesn't play well with static analysis tools.

    === ":x: Don't do this"
        ```python
        from pydantic import BaseModel, conint

        class Foo(BaseModel):
            bar: conint(strict=True, gt=0)
        ```

    === ":white_check_mark: Do this"
        ```python
        from typing import Annotated

        from pydantic import BaseModel, Field

        class Foo(BaseModel):
            bar: Annotated[int, Field(strict=True, gt=0)]
        ```

A wrapper around `int` that allows for additional constraints.

Args:
    strict: Whether to validate the integer in strict mode. Defaults to `None`.
    gt: The value must be greater than this.
    ge: The value must be greater than or equal to this.
    lt: The value must be less than this.
    le: The value must be less than or equal to this.
    multiple_of: The value must be a multiple of this.

Returns:
    The wrapped integer type.

```python
from pydantic import BaseModel, ValidationError, conint

class ConstrainedExample(BaseModel):
    constrained_int: conint(gt=1)

m = ConstrainedExample(constrained_int=2)
print(repr(m))
#> ConstrainedExample(constrained_int=2)

try:
    ConstrainedExample(constrained_int=0)
except ValidationError as e:
    print(e.errors())
    '''
    [
        {
            'type': 'greater_than',
            'loc': ('constrained_int',),
            'msg': 'Input should be greater than 1',
            'input': 0,
            'ctx': {'gt': 1},
            'url': 'https://errors.pydantic.dev/2/v/greater_than',
        }
    ]
    '''
```

### conlist(item_type: 'type[AnyItemType]', *, min_length: 'int | None' = None, max_length: 'int | None' = None, unique_items: 'bool | None' = None) -> 'type[list[AnyItemType]]'
Module: `pydantic.types`

A wrapper around [`list`][] that adds validation.

Args:
    item_type: The type of the items in the list.
    min_length: The minimum length of the list. Defaults to None.
    max_length: The maximum length of the list. Defaults to None.
    unique_items: Whether the items in the list must be unique. Defaults to None.
        !!! warning Deprecated
            The `unique_items` parameter is deprecated, use `Set` instead.
            See [this issue](https://github.com/pydantic/pydantic-core/issues/296) for more details.

Returns:
    The wrapped list type.

### conset(item_type: 'type[HashableItemType]', *, min_length: 'int | None' = None, max_length: 'int | None' = None) -> 'type[set[HashableItemType]]'
Module: `pydantic.types`

A wrapper around `typing.Set` that allows for additional constraints.

Args:
    item_type: The type of the items in the set.
    min_length: The minimum length of the set.
    max_length: The maximum length of the set.

Returns:
    The wrapped set type.

### constr(*, strip_whitespace: 'bool | None' = None, to_upper: 'bool | None' = None, to_lower: 'bool | None' = None, strict: 'bool | None' = None, min_length: 'int | None' = None, max_length: 'int | None' = None, pattern: 'str | Pattern[str] | None' = None) -> 'type[str]'
Module: `pydantic.types`

!!! warning "Discouraged"
    This function is **discouraged** in favor of using
    [`Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated) with
    [`StringConstraints`][pydantic.types.StringConstraints] instead.

    This function will be **deprecated** in Pydantic 3.0.

    The reason is that `constr` returns a type, which doesn't play well with static analysis tools.

    === ":x: Don't do this"
        ```python
        from pydantic import BaseModel, constr

        class Foo(BaseModel):
            bar: constr(strip_whitespace=True, to_upper=True, pattern=r'^[A-Z]+$')
        ```

    === ":white_check_mark: Do this"
        ```python
        from typing import Annotated

        from pydantic import BaseModel, StringConstraints

        class Foo(BaseModel):
            bar: Annotated[
                str,
                StringConstraints(
                    strip_whitespace=True, to_upper=True, pattern=r'^[A-Z]+$'
                ),
            ]
        ```

A wrapper around `str` that allows for additional constraints.

```python
from pydantic import BaseModel, constr

class Foo(BaseModel):
    bar: constr(strip_whitespace=True, to_upper=True)

foo = Foo(bar='  hello  ')
print(foo)
#> bar='HELLO'
```

Args:
    strip_whitespace: Whether to remove leading and trailing whitespace.
    to_upper: Whether to turn all characters to uppercase.
    to_lower: Whether to turn all characters to lowercase.
    strict: Whether to validate the string in strict mode.
    min_length: The minimum length of the string.
    max_length: The maximum length of the string.
    pattern: A regex pattern to validate the string against.

Returns:
    The wrapped string type.

### create_model(model_name: 'str', /, *, __config__: 'ConfigDict | None' = None, __doc__: 'str | None' = None, __base__: 'type[ModelT] | tuple[type[ModelT], ...] | None' = None, __module__: 'str | None' = None, __validators__: 'dict[str, Callable[..., Any]] | None' = None, __cls_kwargs__: 'dict[str, Any] | None' = None, **field_definitions: 'Any | tuple[str, Any]') -> 'type[ModelT]'
Module: `pydantic.main`

!!! abstract "Usage Documentation"
    [Dynamic Model Creation](../concepts/models.md#dynamic-model-creation)

Dynamically creates and returns a new Pydantic model, in other words, `create_model` dynamically creates a
subclass of [`BaseModel`][pydantic.BaseModel].

Args:
    model_name: The name of the newly created model.
    __config__: The configuration of the new model.
    __doc__: The docstring of the new model.
    __base__: The base class or classes for the new model.
    __module__: The name of the module that the model belongs to;
        if `None`, the value is taken from `sys._getframe(1)`
    __validators__: A dictionary of methods that validate fields. The keys are the names of the validation methods to
        be added to the model, and the values are the validation methods themselves. You can read more about functional
        validators [here](https://docs.pydantic.dev/2.9/concepts/validators/#field-validators).
    __cls_kwargs__: A dictionary of keyword arguments for class creation, such as `metaclass`.
    **field_definitions: Field definitions of the new model. Either:

        - a single element, representing the type annotation of the field.
        - a two-tuple, the first element being the type and the second element the assigned value
          (either a default or the [`Field()`][pydantic.Field] function).

Returns:
    The new [model][pydantic.BaseModel].

Raises:
    PydanticUserError: If `__base__` and `__config__` are both passed.

### field_serializer(*fields: 'str', mode: "Literal['plain', 'wrap']" = 'plain', return_type: 'Any' = PydanticUndefined, when_used: 'WhenUsed' = 'always', check_fields: 'bool | None' = None) -> 'Callable[[_FieldWrapSerializerT], _FieldWrapSerializerT] | Callable[[_FieldPlainSerializerT], _FieldPlainSerializerT]'
Module: `pydantic.functional_serializers`

Decorator that enables custom field serialization.

In the below example, a field of type `set` is used to mitigate duplication. A `field_serializer` is used to serialize the data as a sorted list.

```python
from typing import Set

from pydantic import BaseModel, field_serializer

class StudentModel(BaseModel):
    name: str = 'Jane'
    courses: Set[str]

    @field_serializer('courses', when_used='json')
    def serialize_courses_in_order(self, courses: Set[str]):
        return sorted(courses)

student = StudentModel(courses={'Math', 'Chemistry', 'English'})
print(student.model_dump_json())
#> {"name":"Jane","courses":["Chemistry","English","Math"]}
```

See [Custom serializers](../concepts/serialization.md#custom-serializers) for more information.

Four signatures are supported:

- `(self, value: Any, info: FieldSerializationInfo)`
- `(self, value: Any, nxt: SerializerFunctionWrapHandler, info: FieldSerializationInfo)`
- `(value: Any, info: SerializationInfo)`
- `(value: Any, nxt: SerializerFunctionWrapHandler, info: SerializationInfo)`

Args:
    fields: Which field(s) the method should be called on.
    mode: The serialization mode.

        - `plain` means the function will be called instead of the default serialization logic,
        - `wrap` means the function will be called with an argument to optionally call the
           default serialization logic.
    return_type: Optional return type for the function, if omitted it will be inferred from the type annotation.
    when_used: Determines the serializer will be used for serialization.
    check_fields: Whether to check that the fields actually exist on the model.

Returns:
    The decorator function.

### field_validator(field: 'str', /, *fields: 'str', mode: 'FieldValidatorModes' = 'after', check_fields: 'bool | None' = None, json_schema_input_type: 'Any' = PydanticUndefined) -> 'Callable[[Any], Any]'
Module: `pydantic.functional_validators`

!!! abstract "Usage Documentation"
    [field validators](../concepts/validators.md#field-validators)

Decorate methods on the class indicating that they should be used to validate fields.

Example usage:
```python
from typing import Any

from pydantic import (
    BaseModel,
    ValidationError,
    field_validator,
)

class Model(BaseModel):
    a: str

    @field_validator('a')
    @classmethod
    def ensure_foobar(cls, v: Any):
        if 'foobar' not in v:
            raise ValueError('"foobar" not found in a')
        return v

print(repr(Model(a='this is foobar good')))
#> Model(a='this is foobar good')

try:
    Model(a='snap')
except ValidationError as exc_info:
    print(exc_info)
    '''
    1 validation error for Model
    a
      Value error, "foobar" not found in a [type=value_error, input_value='snap', input_type=str]
    '''
```

For more in depth examples, see [Field Validators](../concepts/validators.md#field-validators).

Args:
    field: The first field the `field_validator` should be called on; this is separate
        from `fields` to ensure an error is raised if you don't pass at least one.
    *fields: Additional field(s) the `field_validator` should be called on.
    mode: Specifies whether to validate the fields before or after validation.
    check_fields: Whether to check that the fields actually exist on the model.
    json_schema_input_type: The input type of the function. This is only used to generate
        the appropriate JSON Schema (in validation mode) and can only specified
        when `mode` is either `'before'`, `'plain'` or `'wrap'`.

Returns:
    A decorator that can be used to decorate a function to be used as a field_validator.

Raises:
    PydanticUserError:
        - If `@field_validator` is used bare (with no fields).
        - If the args passed to `@field_validator` as fields are not strings.
        - If `@field_validator` applied to instance methods.

### model_serializer(f: '_ModelPlainSerializerT | _ModelWrapSerializerT | None' = None, /, *, mode: "Literal['plain', 'wrap']" = 'plain', when_used: 'WhenUsed' = 'always', return_type: 'Any' = PydanticUndefined) -> '_ModelPlainSerializerT | Callable[[_ModelWrapSerializerT], _ModelWrapSerializerT] | Callable[[_ModelPlainSerializerT], _ModelPlainSerializerT]'
Module: `pydantic.functional_serializers`

Decorator that enables custom model serialization.

This is useful when a model need to be serialized in a customized manner, allowing for flexibility beyond just specific fields.

An example would be to serialize temperature to the same temperature scale, such as degrees Celsius.

```python
from typing import Literal

from pydantic import BaseModel, model_serializer

class TemperatureModel(BaseModel):
    unit: Literal['C', 'F']
    value: int

    @model_serializer()
    def serialize_model(self):
        if self.unit == 'F':
            return {'unit': 'C', 'value': int((self.value - 32) / 1.8)}
        return {'unit': self.unit, 'value': self.value}

temperature = TemperatureModel(unit='F', value=212)
print(temperature.model_dump())
#> {'unit': 'C', 'value': 100}
```

Two signatures are supported for `mode='plain'`, which is the default:

- `(self)`
- `(self, info: SerializationInfo)`

And two other signatures for `mode='wrap'`:

- `(self, nxt: SerializerFunctionWrapHandler)`
- `(self, nxt: SerializerFunctionWrapHandler, info: SerializationInfo)`

    See [Custom serializers](../concepts/serialization.md#custom-serializers) for more information.

Args:
    f: The function to be decorated.
    mode: The serialization mode.

        - `'plain'` means the function will be called instead of the default serialization logic
        - `'wrap'` means the function will be called with an argument to optionally call the default
            serialization logic.
    when_used: Determines when this serializer should be used.
    return_type: The return type for the function. If omitted it will be inferred from the type annotation.

Returns:
    The decorator function.

### model_validator(*, mode: "Literal['wrap', 'before', 'after']") -> 'Any'
Module: `pydantic.functional_validators`

!!! abstract "Usage Documentation"
    [Model Validators](../concepts/validators.md#model-validators)

Decorate model methods for validation purposes.

Example usage:
```python
from typing_extensions import Self

from pydantic import BaseModel, ValidationError, model_validator

class Square(BaseModel):
    width: float
    height: float

    @model_validator(mode='after')
    def verify_square(self) -> Self:
        if self.width != self.height:
            raise ValueError('width and height do not match')
        return self

s = Square(width=1, height=1)
print(repr(s))
#> Square(width=1.0, height=1.0)

try:
    Square(width=1, height=2)
except ValidationError as e:
    print(e)
    '''
    1 validation error for Square
      Value error, width and height do not match [type=value_error, input_value={'width': 1, 'height': 2}, input_type=dict]
    '''
```

For more in depth examples, see [Model Validators](../concepts/validators.md#model-validators).

Args:
    mode: A required string literal that specifies the validation mode.
        It can be one of the following: 'wrap', 'before', or 'after'.

Returns:
    A decorator that can be used to decorate a function to be used as a model validator.

### parse_obj_as(type_: 'type[T]', obj: 'Any', type_name: 'NameFactory | None' = None) -> 'T'
Module: `pydantic.deprecated.tools`

### root_validator(*__args, pre: 'bool' = False, skip_on_failure: 'bool' = False, allow_reuse: 'bool' = False) -> 'Any'
Module: `pydantic.deprecated.class_validators`

Decorate methods on a model indicating that they should be used to validate (and perhaps
modify) data either before or after standard model parsing/validation is performed.

Args:
    pre (bool, optional): Whether this validator should be called before the standard
        validators (else after). Defaults to False.
    skip_on_failure (bool, optional): Whether to stop validation and return as soon as a
        failure is encountered. Defaults to False.
    allow_reuse (bool, optional): Whether to track and raise an error if another validator
        refers to the decorated function. Defaults to False.

Returns:
    Any: A decorator that can be used to decorate a function to be used as a root_validator.

### schema_json_of(type_: 'Any', *, title: 'NameFactory | None' = None, by_alias: 'bool' = True, ref_template: 'str' = '#/$defs/{model}', schema_generator: 'type[GenerateJsonSchema]' = <class 'pydantic.json_schema.GenerateJsonSchema'>, **dumps_kwargs: 'Any') -> 'str'
Module: `pydantic.deprecated.tools`

Generate a JSON schema (as JSON) for the passed model or dynamically generated one.

### schema_of(type_: 'Any', *, title: 'NameFactory | None' = None, by_alias: 'bool' = True, ref_template: 'str' = '#/$defs/{model}', schema_generator: 'type[GenerateJsonSchema]' = <class 'pydantic.json_schema.GenerateJsonSchema'>) -> 'dict[str, Any]'
Module: `pydantic.deprecated.tools`

Generate a JSON schema (as dict) for the passed model or dynamically generated one.

### validate_call(func: 'AnyCallableT | None' = None, /, *, config: 'ConfigDict | None' = None, validate_return: 'bool' = False) -> 'AnyCallableT | Callable[[AnyCallableT], AnyCallableT]'
Module: `pydantic.validate_call_decorator`

!!! abstract "Usage Documentation"
    [Validation Decorator](../concepts/validation_decorator.md)

Returns a decorated wrapper around the function that validates the arguments and, optionally, the return value.

Usage may be either as a plain decorator `@validate_call` or with arguments `@validate_call(...)`.

Args:
    func: The function to be decorated.
    config: The configuration dictionary.
    validate_return: Whether to validate the return value.

Returns:
    The decorated function.

### validate_email(value: 'str') -> 'tuple[str, str]'
Module: `pydantic.networks`

Email address validation using [email-validator](https://pypi.org/project/email-validator/).

Returns:
    A tuple containing the local part of the email (or the name for "pretty" email addresses)
        and the normalized email.

Raises:
    PydanticCustomError: If the email is invalid.

Note:
    Note that:

    * Raw IP address (literal) domain parts are not allowed.
    * `"John Doe <local_part@domain.com>"` style "pretty" email addresses are processed.
    * Spaces are striped from the beginning and end of addresses, but no error is raised.

### validator(__field: 'str', *fields: 'str', pre: 'bool' = False, each_item: 'bool' = False, always: 'bool' = False, check_fields: 'bool | None' = None, allow_reuse: 'bool' = False) -> 'Callable[[_V1ValidatorType], _V1ValidatorType]'
Module: `pydantic.deprecated.class_validators`

Decorate methods on the class indicating that they should be used to validate fields.

Args:
    __field (str): The first field the validator should be called on; this is separate
        from `fields` to ensure an error is raised if you don't pass at least one.
    *fields (str): Additional field(s) the validator should be called on.
    pre (bool, optional): Whether this validator should be called before the standard
        validators (else after). Defaults to False.
    each_item (bool, optional): For complex objects (sets, lists etc.) whether to validate
        individual elements rather than the whole object. Defaults to False.
    always (bool, optional): Whether this method and other validators should be called even if
        the value is missing. Defaults to False.
    check_fields (bool | None, optional): Whether to check that the fields actually exist on the model.
        Defaults to None.
    allow_reuse (bool, optional): Whether to track and raise an error if another validator refers to
        the decorated function. Defaults to False.

Returns:
    Callable: A decorator that can be used to decorate a
        function to be used as a validator.

### with_config(config: 'ConfigDict | None' = None, /, **kwargs: 'Any') -> 'Callable[[_TypeT], _TypeT]'
Module: `pydantic.config`

!!! abstract "Usage Documentation"
    [Configuration with other types](../concepts/config.md#configuration-on-other-supported-types)

A convenience decorator to set a [Pydantic configuration](config.md) on a `TypedDict` or a `dataclass` from the standard library.

Although the configuration can be set using the `__pydantic_config__` attribute, it does not play well with type checkers,
especially with `TypedDict`.

!!! example "Usage"

    ```python
    from typing_extensions import TypedDict

    from pydantic import ConfigDict, TypeAdapter, with_config

    @with_config(ConfigDict(str_to_lower=True))
    class TD(TypedDict):
        x: str

    ta = TypeAdapter(TD)

    print(ta.validate_python({'x': 'ABC'}))
    #> {'x': 'abc'}
    ```

## Classes

### AfterValidator
Module: `pydantic.functional_validators`

!!! abstract "Usage Documentation"
    [field *after* validators](../concepts/validators.md#field-after-validator)

A metadata class that indicates that a validation should be applied **after** the inner validation logic.

Attributes:
    func: The validator function.

Example:
    ```python
    from typing import Annotated

    from pydantic import AfterValidator, BaseModel, ValidationError

    MyInt = Annotated[int, AfterValidator(lambda v: v + 1)]

    class Model(BaseModel):
        a: MyInt

    print(Model(a=1).a)
    #> 2

    try:
        Model(a='a')
    except ValidationError as e:
        print(e.json(indent=2))
        '''
        [
          {
            "type": "int_parsing",
            "loc": [
              "a"
            ],
            "msg": "Input should be a valid integer, unable to parse string as an integer",
            "input": "a",
            "url": "https://errors.pydantic.dev/2/v/int_parsing"
          }
        ]
        '''
    ```

### AliasChoices
Module: `pydantic.aliases`

!!! abstract "Usage Documentation"
    [`AliasPath` and `AliasChoices`](../concepts/alias.md#aliaspath-and-aliaschoices)

A data class used by `validation_alias` as a convenience to create aliases.

Attributes:
    choices: A list containing a string or `AliasPath`.

#### Methods

**`convert_to_aliases(self) -> 'list[list[str | int]]'`**

Converts arguments to a list of lists containing string or integer aliases.

Returns:
The list of aliases.

### AliasGenerator
Module: `pydantic.aliases`

!!! abstract "Usage Documentation"
    [Using an `AliasGenerator`](../concepts/alias.md#using-an-aliasgenerator)

A data class used by `alias_generator` as a convenience to create various aliases.

Attributes:
    alias: A callable that takes a field name and returns an alias for it.
    validation_alias: A callable that takes a field name and returns a validation alias for it.
    serialization_alias: A callable that takes a field name and returns a serialization alias for it.

#### Methods

**`generate_aliases(self, field_name: 'str') -> 'tuple[str | None, str | AliasPath | AliasChoices | None, str | None]'`**

Generate `alias`, `validation_alias`, and `serialization_alias` for a field.

Returns:
A tuple of three aliases - validation, alias, and serialization.

### AliasPath
Module: `pydantic.aliases`

!!! abstract "Usage Documentation"
    [`AliasPath` and `AliasChoices`](../concepts/alias.md#aliaspath-and-aliaschoices)

A data class used by `validation_alias` as a convenience to create aliases.

Attributes:
    path: A list of string or integer aliases.

#### Methods

**`convert_to_aliases(self) -> 'list[str | int]'`**

Converts arguments to a list of string or integer aliases.

Returns:
The list of aliases.

**`search_dict_for_path(self, d: 'dict') -> 'Any'`**

Searches a dictionary for the path specified by the alias.

Returns:
The value at the specified path, or `PydanticUndefined` if the path is not found.

### AllowInfNan
Module: `pydantic.types`

A field metadata class to indicate that a field should allow `-inf`, `inf`, and `nan`.

Use this class as an annotation via [`Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated), as seen below.

Attributes:
    allow_inf_nan: Whether to allow `-inf`, `inf`, and `nan`. Defaults to `True`.

Example:
    ```python
    from typing import Annotated

    from pydantic.types import AllowInfNan

    LaxFloat = Annotated[float, AllowInfNan()]
    ```

### AmqpDsn
Module: `pydantic.networks`

A type that will accept any AMQP DSN.

* User info required
* TLD not required
* Host not required

### AnyHttpUrl
Module: `pydantic.networks`

A type that will accept any http or https URL.

* TLD not required
* Host not required

### AnyUrl
Module: `pydantic.networks`

Base type for all URLs.

* Any scheme allowed
* Top-level domain (TLD) not required
* Host not required

Assuming an input URL of `http://samuel:pass@example.com:8000/the/path/?query=here#fragment=is;this=bit`,
the types export the following properties:

- `scheme`: the URL scheme (`http`), always set.
- `host`: the URL host (`example.com`).
- `username`: optional username if included (`samuel`).
- `password`: optional password if included (`pass`).
- `port`: optional port (`8000`).
- `path`: optional path (`/the/path/`).
- `query`: optional URL query (for example, `GET` arguments or "search string", such as `query=here`).
- `fragment`: optional fragment (`fragment=is;this=bit`).

### AnyWebsocketUrl
Module: `pydantic.networks`

A type that will accept any ws or wss URL.

* TLD not required
* Host not required

### AwareDatetime
Module: `pydantic.types`

A datetime that requires timezone info.

### Base64Encoder
Module: `pydantic.types`

Standard (non-URL-safe) Base64 encoder.

### BaseConfig
Module: `pydantic.deprecated.config`

This class is only retained for backwards compatibility.

!!! Warning "Deprecated"
    BaseConfig is deprecated. Use the [`pydantic.ConfigDict`][pydantic.ConfigDict] instead.

### BeforeValidator
Module: `pydantic.functional_validators`

!!! abstract "Usage Documentation"
    [field *before* validators](../concepts/validators.md#field-before-validator)

A metadata class that indicates that a validation should be applied **before** the inner validation logic.

Attributes:
    func: The validator function.
    json_schema_input_type: The input type of the function. This is only used to generate the appropriate
        JSON Schema (in validation mode).

Example:
    ```python
    from typing import Annotated

    from pydantic import BaseModel, BeforeValidator

    MyInt = Annotated[int, BeforeValidator(lambda v: v + 1)]

    class Model(BaseModel):
        a: MyInt

    print(Model(a=1).a)
    #> 2

    try:
        Model(a='a')
    except TypeError as e:
        print(e)
        #> can only concatenate str (not "int") to str
    ```

### ByteSize
Module: `pydantic.types`

Converts a string representing a number of bytes with units (such as `'1KB'` or `'11.5MiB'`) into an integer.

You can use the `ByteSize` data type to (case-insensitively) convert a string representation of a number of bytes into
an integer, and also to print out human-readable strings representing a number of bytes.

In conformance with [IEC 80000-13 Standard](https://en.wikipedia.org/wiki/ISO/IEC_80000) we interpret `'1KB'` to mean 1000 bytes,
and `'1KiB'` to mean 1024 bytes. In general, including a middle `'i'` will cause the unit to be interpreted as a power of 2,
rather than a power of 10 (so, for example, `'1 MB'` is treated as `1_000_000` bytes, whereas `'1 MiB'` is treated as `1_048_576` bytes).

!!! info
    Note that `1b` will be parsed as "1 byte" and not "1 bit".

```python
from pydantic import BaseModel, ByteSize

class MyModel(BaseModel):
    size: ByteSize

print(MyModel(size=52000).size)
#> 52000
print(MyModel(size='3000 KiB').size)
#> 3072000

m = MyModel(size='50 PB')
print(m.size.human_readable())
#> 44.4PiB
print(m.size.human_readable(decimal=True))
#> 50.0PB
print(m.size.human_readable(separator=' '))
#> 44.4 PiB

print(m.size.to('TiB'))
#> 45474.73508864641
```

#### Methods

**`human_readable(self, decimal: 'bool' = False, separator: 'str' = '') -> 'str'`**

Converts a byte size to a human readable string.

Args:
decimal: If True, use decimal units (e.g. 1000 bytes per KB). If False, use binary units
(e.g. 1024 bytes per KiB).
separator: A string used to split the value and unit. Defaults to an empty string ('').

Returns:
A human readable string representation of the byte size.

**`to(self, unit: 'str') -> 'float'`**

Converts a byte size to another unit, including both byte and bit units.

Args:
unit: The unit to convert to. Must be one of the following: B, KB, MB, GB, TB, PB, EB,
KiB, MiB, GiB, TiB, PiB, EiB (byte units) and
bit, kbit, mbit, gbit, tbit, pbit, ebit,
kibit, mibit, gibit, tibit, pibit, eibit (bit units).

Returns:
The byte size in the new unit.

### ClickHouseDsn
Module: `pydantic.networks`

A type that will accept any ClickHouse DSN.

* User info required
* TLD not required
* Host not required

### CockroachDsn
Module: `pydantic.networks`

A type that will accept any Cockroach DSN.

* User info required
* TLD not required
* Host required

### ConfigDict
Module: `pydantic.config`

A TypedDict for configuring Pydantic behaviour.

### Discriminator
Module: `pydantic.types`

!!! abstract "Usage Documentation"
    [Discriminated Unions with `Callable` `Discriminator`](../concepts/unions.md#discriminated-unions-with-callable-discriminator)

Provides a way to use a custom callable as the way to extract the value of a union discriminator.

This allows you to get validation behavior like you'd get from `Field(discriminator=<field_name>)`,
but without needing to have a single shared field across all the union choices. This also makes it
possible to handle unions of models and primitive types with discriminated-union-style validation errors.
Finally, this allows you to use a custom callable as the way to identify which member of a union a value
belongs to, while still seeing all the performance benefits of a discriminated union.

Consider this example, which is much more performant with the use of `Discriminator` and thus a `TaggedUnion`
than it would be as a normal `Union`.

```python
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Discriminator, Tag

class Pie(BaseModel):
    time_to_cook: int
    num_ingredients: int

class ApplePie(Pie):
    fruit: Literal['apple'] = 'apple'

class PumpkinPie(Pie):
    filling: Literal['pumpkin'] = 'pumpkin'

def get_discriminator_value(v: Any) -> str:
    if isinstance(v, dict):
        return v.get('fruit', v.get('filling'))
    return getattr(v, 'fruit', getattr(v, 'filling', None))

class ThanksgivingDinner(BaseModel):
    dessert: Annotated[
        Union[
            Annotated[ApplePie, Tag('apple')],
            Annotated[PumpkinPie, Tag('pumpkin')],
        ],
        Discriminator(get_discriminator_value),
    ]

apple_variation = ThanksgivingDinner.model_validate(
    {'dessert': {'fruit': 'apple', 'time_to_cook': 60, 'num_ingredients': 8}}
)
print(repr(apple_variation))
'''
ThanksgivingDinner(dessert=ApplePie(time_to_cook=60, num_ingredients=8, fruit='apple'))
'''

pumpkin_variation = ThanksgivingDinner.model_validate(
    {
        'dessert': {
            'filling': 'pumpkin',
            'time_to_cook': 40,
            'num_ingredients': 6,
        }
    }
)
print(repr(pumpkin_variation))
'''
ThanksgivingDinner(dessert=PumpkinPie(time_to_cook=40, num_ingredients=6, filling='pumpkin'))
'''
```

See the [Discriminated Unions] concepts docs for more details on how to use `Discriminator`s.

[Discriminated Unions]: ../concepts/unions.md#discriminated-unions

### EmailStr
Module: `pydantic.networks`

Info:
    To use this type, you need to install the optional
    [`email-validator`](https://github.com/JoshData/python-email-validator) package:

    ```bash
    pip install email-validator
    ```

Validate email addresses.

```python
from pydantic import BaseModel, EmailStr

class Model(BaseModel):
    email: EmailStr

print(Model(email='contact@mail.com'))
#> email='contact@mail.com'
```

### EncodedBytes
Module: `pydantic.types`

A bytes type that is encoded and decoded using the specified encoder.

`EncodedBytes` needs an encoder that implements `EncoderProtocol` to operate.

```python
from typing import Annotated

from pydantic import BaseModel, EncodedBytes, EncoderProtocol, ValidationError

class MyEncoder(EncoderProtocol):
    @classmethod
    def decode(cls, data: bytes) -> bytes:
        if data == b'**undecodable**':
            raise ValueError('Cannot decode data')
        return data[13:]

    @classmethod
    def encode(cls, value: bytes) -> bytes:
        return b'**encoded**: ' + value

    @classmethod
    def get_json_format(cls) -> str:
        return 'my-encoder'

MyEncodedBytes = Annotated[bytes, EncodedBytes(encoder=MyEncoder)]

class Model(BaseModel):
    my_encoded_bytes: MyEncodedBytes

# Initialize the model with encoded data
m = Model(my_encoded_bytes=b'**encoded**: some bytes')

# Access decoded value
print(m.my_encoded_bytes)
#> b'some bytes'

# Serialize into the encoded form
print(m.model_dump())
#> {'my_encoded_bytes': b'**encoded**: some bytes'}

# Validate encoded data
try:
    Model(my_encoded_bytes=b'**undecodable**')
except ValidationError as e:
    print(e)
    '''
    1 validation error for Model
    my_encoded_bytes
      Value error, Cannot decode data [type=value_error, input_value=b'**undecodable**', input_type=bytes]
    '''
```

#### Methods

**`decode(self, data: 'bytes', _: 'core_schema.ValidationInfo') -> 'bytes'`**

Decode the data using the specified encoder.

Args:
data: The data to decode.

Returns:
The decoded data.

**`encode(self, value: 'bytes') -> 'bytes'`**

Encode the data using the specified encoder.

Args:
value: The data to encode.

Returns:
The encoded data.

### EncodedStr
Module: `pydantic.types`

A str type that is encoded and decoded using the specified encoder.

`EncodedStr` needs an encoder that implements `EncoderProtocol` to operate.

```python
from typing import Annotated

from pydantic import BaseModel, EncodedStr, EncoderProtocol, ValidationError

class MyEncoder(EncoderProtocol):
    @classmethod
    def decode(cls, data: bytes) -> bytes:
        if data == b'**undecodable**':
            raise ValueError('Cannot decode data')
        return data[13:]

    @classmethod
    def encode(cls, value: bytes) -> bytes:
        return b'**encoded**: ' + value

    @classmethod
    def get_json_format(cls) -> str:
        return 'my-encoder'

MyEncodedStr = Annotated[str, EncodedStr(encoder=MyEncoder)]

class Model(BaseModel):
    my_encoded_str: MyEncodedStr

# Initialize the model with encoded data
m = Model(my_encoded_str='**encoded**: some str')

# Access decoded value
print(m.my_encoded_str)
#> some str

# Serialize into the encoded form
print(m.model_dump())
#> {'my_encoded_str': '**encoded**: some str'}

# Validate encoded data
try:
    Model(my_encoded_str='**undecodable**')
except ValidationError as e:
    print(e)
    '''
    1 validation error for Model
    my_encoded_str
      Value error, Cannot decode data [type=value_error, input_value='**undecodable**', input_type=str]
    '''
```

#### Methods

**`decode_str(self, data: 'str', _: 'core_schema.ValidationInfo') -> 'str'`**

Decode the data using the specified encoder.

Args:
data: The data to decode.

Returns:
The decoded data.

**`encode_str(self, value: 'str') -> 'str'`**

Encode the data using the specified encoder.

Args:
value: The data to encode.

Returns:
The encoded data.

### EncoderProtocol
Module: `pydantic.types`

Protocol for encoding and decoding data to and from bytes.

### Extra
Module: `pydantic.deprecated.config`

### FailFast
Module: `pydantic.types`

A `FailFast` annotation can be used to specify that validation should stop at the first error.

This can be useful when you want to validate a large amount of data and you only need to know if it's valid or not.

You might want to enable this setting if you want to validate your data faster (basically, if you use this,
validation will be more performant with the caveat that you get less information).

```python
from typing import Annotated

from pydantic import BaseModel, FailFast, ValidationError

class Model(BaseModel):
    x: Annotated[list[int], FailFast()]

# This will raise a single error for the first invalid value and stop validation
try:
    obj = Model(x=[1, 2, 'a', 4, 5, 'b', 7, 8, 9, 'c'])
except ValidationError as e:
    print(e)
    '''
    1 validation error for Model
    x.2
      Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='a', input_type=str]
    '''
```

### FieldSerializationInfo
Module: `pydantic_core.core_schema`

Base class for protocol classes.

Protocol classes are defined as::

    class Proto(Protocol):
        def meth(self) -> int:
            ...

Such classes are primarily used with static type checkers that recognize
structural subtyping (static duck-typing).

For example::

    class C:
        def meth(self) -> int:
            return 0

    def func(x: Proto) -> int:
        return x.meth()

    func(C())  # Passes static type check

See PEP 544 for details. Protocol classes decorated with
@typing.runtime_checkable act as simple-minded runtime protocols that check
only the presence of given attributes, ignoring their type signatures.
Protocol classes can be generic, they are defined as::

    class GenProto(Protocol[T]):
        def meth(self) -> T:
            ...

### FileUrl
Module: `pydantic.networks`

A type that will accept any file URL.

* Host not required

### FtpUrl
Module: `pydantic.networks`

A type that will accept ftp URL.

* TLD not required
* Host not required

### FutureDate
Module: `pydantic.types`

A date in the future.

### FutureDatetime
Module: `pydantic.types`

A datetime that must be in the future.

### GetCoreSchemaHandler
Module: `pydantic.annotated_handlers`

Handler to call into the next CoreSchema schema generation function.

#### Methods

**`generate_schema(self, source_type: 'Any', /) -> 'core_schema.CoreSchema'`**

Generate a schema unrelated to the current context.
Use this function if e.g. you are handling schema generation for a sequence
and want to generate a schema for its items.
Otherwise, you may end up doing something like applying a `min_length` constraint
that was intended for the sequence itself to its items!

Args:
source_type: The input type.

Returns:
CoreSchema: The `pydantic-core` CoreSchema generated.

**`resolve_ref_schema(self, maybe_ref_schema: 'core_schema.CoreSchema', /) -> 'core_schema.CoreSchema'`**

Get the real schema for a `definition-ref` schema.
If the schema given is not a `definition-ref` schema, it will be returned as is.
This means you don't have to check before calling this function.

Args:
maybe_ref_schema: A `CoreSchema`, `ref`-based or not.

Raises:
LookupError: If the `ref` is not found.

Returns:
A concrete `CoreSchema`.

### GetJsonSchemaHandler
Module: `pydantic.annotated_handlers`

Handler to call into the next JSON schema generation function.

Attributes:
    mode: Json schema mode, can be `validation` or `serialization`.

#### Methods

**`resolve_ref_schema(self, maybe_ref_json_schema: 'JsonSchemaValue', /) -> 'JsonSchemaValue'`**

Get the real schema for a `{"$ref": ...}` schema.
If the schema given is not a `$ref` schema, it will be returned as is.
This means you don't have to check before calling this function.

Args:
maybe_ref_json_schema: A JsonSchemaValue which may be a `$ref` schema.

Raises:
LookupError: If the ref is not found.

Returns:
JsonSchemaValue: A JsonSchemaValue that has no `$ref`.

### GetPydanticSchema
Module: `pydantic.types`

!!! abstract "Usage Documentation"
    [Using `GetPydanticSchema` to Reduce Boilerplate](../concepts/types.md#using-getpydanticschema-to-reduce-boilerplate)

A convenience class for creating an annotation that provides pydantic custom type hooks.

This class is intended to eliminate the need to create a custom "marker" which defines the
 `__get_pydantic_core_schema__` and `__get_pydantic_json_schema__` custom hook methods.

For example, to have a field treated by type checkers as `int`, but by pydantic as `Any`, you can do:
```python
from typing import Annotated, Any

from pydantic import BaseModel, GetPydanticSchema

HandleAsAny = GetPydanticSchema(lambda _s, h: h(Any))

class Model(BaseModel):
    x: Annotated[int, HandleAsAny]  # pydantic sees `x: Any`

print(repr(Model(x='abc').x))
#> 'abc'
```

### HttpUrl
Module: `pydantic.networks`

A type that will accept any http or https URL.

* TLD not required
* Host not required
* Max length 2083

```python
from pydantic import BaseModel, HttpUrl, ValidationError

class MyModel(BaseModel):
    url: HttpUrl

m = MyModel(url='http://www.example.com')  # (1)!
print(m.url)
#> http://www.example.com/

try:
    MyModel(url='ftp://invalid.url')
except ValidationError as e:
    print(e)
    '''
    1 validation error for MyModel
    url
      URL scheme should be 'http' or 'https' [type=url_scheme, input_value='ftp://invalid.url', input_type=str]
    '''

try:
    MyModel(url='not a url')
except ValidationError as e:
    print(e)
    '''
    1 validation error for MyModel
    url
      Input should be a valid URL, relative URL without a base [type=url_parsing, input_value='not a url', input_type=str]
    '''
```

1. Note: mypy would prefer `m = MyModel(url=HttpUrl('http://www.example.com'))`, but Pydantic will convert the string to an HttpUrl instance anyway.

"International domains" (e.g. a URL where the host or TLD includes non-ascii characters) will be encoded via
[punycode](https://en.wikipedia.org/wiki/Punycode) (see
[this article](https://www.xudongz.com/blog/2017/idn-phishing/) for a good description of why this is important):

```python
from pydantic import BaseModel, HttpUrl

class MyModel(BaseModel):
    url: HttpUrl

m1 = MyModel(url='http://puny£code.com')
print(m1.url)
#> http://xn--punycode-eja.com/
m2 = MyModel(url='https://www.аррӏе.com/')
print(m2.url)
#> https://www.xn--80ak6aa92e.com/
m3 = MyModel(url='https://www.example.珠宝/')
print(m3.url)
#> https://www.example.xn--pbt977c/
```


!!! warning "Underscores in Hostnames"
    In Pydantic, underscores are allowed in all parts of a domain except the TLD.
    Technically this might be wrong - in theory the hostname cannot have underscores, but subdomains can.

    To explain this; consider the following two cases:

    - `exam_ple.co.uk`: the hostname is `exam_ple`, which should not be allowed since it contains an underscore.
    - `foo_bar.example.com` the hostname is `example`, which should be allowed since the underscore is in the subdomain.

    Without having an exhaustive list of TLDs, it would be impossible to differentiate between these two. Therefore
    underscores are allowed, but you can always do further validation in a validator if desired.

    Also, Chrome, Firefox, and Safari all currently accept `http://exam_ple.com` as a URL, so we're in good
    (or at least big) company.

### IPvAnyAddress
Module: `pydantic.networks`

Validate an IPv4 or IPv6 address.

```python
from pydantic import BaseModel
from pydantic.networks import IPvAnyAddress

class IpModel(BaseModel):
    ip: IPvAnyAddress

print(IpModel(ip='127.0.0.1'))
#> ip=IPv4Address('127.0.0.1')

try:
    IpModel(ip='http://www.example.com')
except ValueError as e:
    print(e.errors())
    '''
    [
        {
            'type': 'ip_any_address',
            'loc': ('ip',),
            'msg': 'value is not a valid IPv4 or IPv6 address',
            'input': 'http://www.example.com',
        }
    ]
    '''
```

### IPvAnyInterface
Module: `pydantic.networks`

Validate an IPv4 or IPv6 interface.

### IPvAnyNetwork
Module: `pydantic.networks`

Validate an IPv4 or IPv6 network.

### ImportString
Module: `pydantic.types`

A type that can be used to import a Python object from a string.

`ImportString` expects a string and loads the Python object importable at that dotted path.
Attributes of modules may be separated from the module by `:` or `.`, e.g. if `'math:cos'` is provided,
the resulting field value would be the function `cos`. If a `.` is used and both an attribute and submodule
are present at the same path, the module will be preferred.

On model instantiation, pointers will be evaluated and imported. There is
some nuance to this behavior, demonstrated in the examples below.

```python
import math

from pydantic import BaseModel, Field, ImportString, ValidationError

class ImportThings(BaseModel):
    obj: ImportString

# A string value will cause an automatic import
my_cos = ImportThings(obj='math.cos')

# You can use the imported function as you would expect
cos_of_0 = my_cos.obj(0)
assert cos_of_0 == 1

# A string whose value cannot be imported will raise an error
try:
    ImportThings(obj='foo.bar')
except ValidationError as e:
    print(e)
    '''
    1 validation error for ImportThings
    obj
      Invalid python path: No module named 'foo.bar' [type=import_error, input_value='foo.bar', input_type=str]
    '''

# Actual python objects can be assigned as well
my_cos = ImportThings(obj=math.cos)
my_cos_2 = ImportThings(obj='math.cos')
my_cos_3 = ImportThings(obj='math:cos')
assert my_cos == my_cos_2 == my_cos_3

# You can set default field value either as Python object:
class ImportThingsDefaultPyObj(BaseModel):
    obj: ImportString = math.cos

# or as a string value (but only if used with `validate_default=True`)
class ImportThingsDefaultString(BaseModel):
    obj: ImportString = Field(default='math.cos', validate_default=True)

my_cos_default1 = ImportThingsDefaultPyObj()
my_cos_default2 = ImportThingsDefaultString()
assert my_cos_default1.obj == my_cos_default2.obj == math.cos

# note: this will not work!
class ImportThingsMissingValidateDefault(BaseModel):
    obj: ImportString = 'math.cos'

my_cos_default3 = ImportThingsMissingValidateDefault()
assert my_cos_default3.obj == 'math.cos'  # just string, not evaluated
```

Serializing an `ImportString` type to json is also possible.

```python
from pydantic import BaseModel, ImportString

class ImportThings(BaseModel):
    obj: ImportString

# Create an instance
m = ImportThings(obj='math.cos')
print(m)
#> obj=<built-in function cos>
print(m.model_dump_json())
#> {"obj":"math.cos"}
```

### InstanceOf
Module: `pydantic.functional_validators`

Generic type for annotating a type that is an instance of a given class.

Example:
    ```python
    from pydantic import BaseModel, InstanceOf

    class Foo:
        ...

    class Bar(BaseModel):
        foo: InstanceOf[Foo]

    Bar(foo=Foo())
    try:
        Bar(foo=42)
    except ValidationError as e:
        print(e)
        """
        [
        │   {
        │   │   'type': 'is_instance_of',
        │   │   'loc': ('foo',),
        │   │   'msg': 'Input should be an instance of Foo',
        │   │   'input': 42,
        │   │   'ctx': {'class': 'Foo'},
        │   │   'url': 'https://errors.pydantic.dev/0.38.0/v/is_instance_of'
        │   }
        ]
        """
    ```

### Json
Module: `pydantic.types`

A special type wrapper which loads JSON before parsing.

You can use the `Json` data type to make Pydantic first load a raw JSON string before
validating the loaded data into the parametrized type:

```python
from typing import Any

from pydantic import BaseModel, Json, ValidationError

class AnyJsonModel(BaseModel):
    json_obj: Json[Any]

class ConstrainedJsonModel(BaseModel):
    json_obj: Json[list[int]]

print(AnyJsonModel(json_obj='{"b": 1}'))
#> json_obj={'b': 1}
print(ConstrainedJsonModel(json_obj='[1, 2, 3]'))
#> json_obj=[1, 2, 3]

try:
    ConstrainedJsonModel(json_obj=12)
except ValidationError as e:
    print(e)
    '''
    1 validation error for ConstrainedJsonModel
    json_obj
      JSON input should be string, bytes or bytearray [type=json_type, input_value=12, input_type=int]
    '''

try:
    ConstrainedJsonModel(json_obj='[a, b]')
except ValidationError as e:
    print(e)
    '''
    1 validation error for ConstrainedJsonModel
    json_obj
      Invalid JSON: expected value at line 1 column 2 [type=json_invalid, input_value='[a, b]', input_type=str]
    '''

try:
    ConstrainedJsonModel(json_obj='["a", "b"]')
except ValidationError as e:
    print(e)
    '''
    2 validation errors for ConstrainedJsonModel
    json_obj.0
      Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='a', input_type=str]
    json_obj.1
      Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='b', input_type=str]
    '''
```

When you dump the model using `model_dump` or `model_dump_json`, the dumped value will be the result of validation,
not the original JSON string. However, you can use the argument `round_trip=True` to get the original JSON string back:

```python
from pydantic import BaseModel, Json

class ConstrainedJsonModel(BaseModel):
    json_obj: Json[list[int]]

print(ConstrainedJsonModel(json_obj='[1, 2, 3]').model_dump_json())
#> {"json_obj":[1,2,3]}
print(
    ConstrainedJsonModel(json_obj='[1, 2, 3]').model_dump_json(round_trip=True)
)
#> {"json_obj":"[1,2,3]"}
```

### KafkaDsn
Module: `pydantic.networks`

A type that will accept any Kafka DSN.

* User info required
* TLD not required
* Host not required

### MariaDBDsn
Module: `pydantic.networks`

A type that will accept any MariaDB DSN.

* User info required
* TLD not required
* Host not required

### ModelWrapValidatorHandler
Module: `pydantic.functional_validators`

`@model_validator` decorated function handler argument type. This is used when `mode='wrap'`.

### MongoDsn
Module: `pydantic.networks`

A type that will accept any MongoDB DSN.

* User info not required
* Database name not required
* Port not required
* User info may be passed without user part (e.g., `mongodb://mongodb0.example.com:27017`).

### MySQLDsn
Module: `pydantic.networks`

A type that will accept any MySQL DSN.

* User info required
* TLD not required
* Host not required

### NaiveDatetime
Module: `pydantic.types`

A datetime that doesn't require timezone info.

### NameEmail
Module: `pydantic.networks`

Info:
    To use this type, you need to install the optional
    [`email-validator`](https://github.com/JoshData/python-email-validator) package:

    ```bash
    pip install email-validator
    ```

Validate a name and email address combination, as specified by
[RFC 5322](https://datatracker.ietf.org/doc/html/rfc5322#section-3.4).

The `NameEmail` has two properties: `name` and `email`.
In case the `name` is not provided, it's inferred from the email address.

```python
from pydantic import BaseModel, NameEmail

class User(BaseModel):
    email: NameEmail

user = User(email='Fred Bloggs <fred.bloggs@example.com>')
print(user.email)
#> Fred Bloggs <fred.bloggs@example.com>
print(user.email.name)
#> Fred Bloggs

user = User(email='fred.bloggs@example.com')
print(user.email)
#> fred.bloggs <fred.bloggs@example.com>
print(user.email.name)
#> fred.bloggs
```

### NatsDsn
Module: `pydantic.networks`

A type that will accept any NATS DSN.

NATS is a connective technology built for the ever increasingly hyper-connected world.
It is a single technology that enables applications to securely communicate across
any combination of cloud vendors, on-premise, edge, web and mobile, and devices.
More: https://nats.io

### PastDate
Module: `pydantic.types`

A date in the past.

### PastDatetime
Module: `pydantic.types`

A datetime that must be in the past.

### PaymentCardNumber
Module: `pydantic.types`

Based on: https://en.wikipedia.org/wiki/Payment_card_number.

#### Methods

**`validate_brand(card_number: 'str') -> 'PaymentCardBrand'`**

Validate length based on BIN for major brands:
https://en.wikipedia.org/wiki/Payment_card_number#Issuer_identification_number_(IIN).

### PlainSerializer
Module: `pydantic.functional_serializers`

Plain serializers use a function to modify the output of serialization.

This is particularly helpful when you want to customize the serialization for annotated types.
Consider an input of `list`, which will be serialized into a space-delimited string.

```python
from typing import Annotated

from pydantic import BaseModel, PlainSerializer

CustomStr = Annotated[
    list, PlainSerializer(lambda x: ' '.join(x), return_type=str)
]

class StudentModel(BaseModel):
    courses: CustomStr

student = StudentModel(courses=['Math', 'Chemistry', 'English'])
print(student.model_dump())
#> {'courses': 'Math Chemistry English'}
```

Attributes:
    func: The serializer function.
    return_type: The return type for the function. If omitted it will be inferred from the type annotation.
    when_used: Determines when this serializer should be used. Accepts a string with values `'always'`,
        `'unless-none'`, `'json'`, and `'json-unless-none'`. Defaults to 'always'.

### PlainValidator
Module: `pydantic.functional_validators`

!!! abstract "Usage Documentation"
    [field *plain* validators](../concepts/validators.md#field-plain-validator)

A metadata class that indicates that a validation should be applied **instead** of the inner validation logic.

!!! note
    Before v2.9, `PlainValidator` wasn't always compatible with JSON Schema generation for `mode='validation'`.
    You can now use the `json_schema_input_type` argument to specify the input type of the function
    to be used in the JSON schema when `mode='validation'` (the default). See the example below for more details.

Attributes:
    func: The validator function.
    json_schema_input_type: The input type of the function. This is only used to generate the appropriate
        JSON Schema (in validation mode). If not provided, will default to `Any`.

Example:
    ```python
    from typing import Annotated, Union

    from pydantic import BaseModel, PlainValidator

    MyInt = Annotated[
        int,
        PlainValidator(
            lambda v: int(v) + 1, json_schema_input_type=Union[str, int]  # (1)!
        ),
    ]

    class Model(BaseModel):
        a: MyInt

    print(Model(a='1').a)
    #> 2

    print(Model(a=1).a)
    #> 2
    ```

    1. In this example, we've specified the `json_schema_input_type` as `Union[str, int]` which indicates to the JSON schema
    generator that in validation mode, the input type for the `a` field can be either a `str` or an `int`.

### PostgresDsn
Module: `pydantic.networks`

A type that will accept any Postgres DSN.

* User info required
* TLD not required
* Host required
* Supports multiple hosts

If further validation is required, these properties can be used by validators to enforce specific behaviour:

```python
from pydantic import (
    BaseModel,
    HttpUrl,
    PostgresDsn,
    ValidationError,
    field_validator,
)

class MyModel(BaseModel):
    url: HttpUrl

m = MyModel(url='http://www.example.com')

# the repr() method for a url will display all properties of the url
print(repr(m.url))
#> HttpUrl('http://www.example.com/')
print(m.url.scheme)
#> http
print(m.url.host)
#> www.example.com
print(m.url.port)
#> 80

class MyDatabaseModel(BaseModel):
    db: PostgresDsn

    @field_validator('db')
    def check_db_name(cls, v):
        assert v.path and len(v.path) > 1, 'database must be provided'
        return v

m = MyDatabaseModel(db='postgres://user:pass@localhost:5432/foobar')
print(m.db)
#> postgres://user:pass@localhost:5432/foobar

try:
    MyDatabaseModel(db='postgres://user:pass@localhost:5432')
except ValidationError as e:
    print(e)
    '''
    1 validation error for MyDatabaseModel
    db
      Assertion failed, database must be provided
    assert (None)
     +  where None = PostgresDsn('postgres://user:pass@localhost:5432').path [type=assertion_error, input_value='postgres://user:pass@localhost:5432', input_type=str]
    '''
```

### PydanticDeprecatedSince20
Module: `pydantic.warnings`

A specific `PydanticDeprecationWarning` subclass defining functionality deprecated since Pydantic 2.0.

### PydanticDeprecatedSince210
Module: `pydantic.warnings`

A specific `PydanticDeprecationWarning` subclass defining functionality deprecated since Pydantic 2.10.

### PydanticDeprecatedSince211
Module: `pydantic.warnings`

A specific `PydanticDeprecationWarning` subclass defining functionality deprecated since Pydantic 2.11.

### PydanticDeprecatedSince26
Module: `pydantic.warnings`

A specific `PydanticDeprecationWarning` subclass defining functionality deprecated since Pydantic 2.6.

### PydanticDeprecatedSince29
Module: `pydantic.warnings`

A specific `PydanticDeprecationWarning` subclass defining functionality deprecated since Pydantic 2.9.

### PydanticDeprecationWarning
Module: `pydantic.warnings`

A Pydantic specific deprecation warning.

This warning is raised when using deprecated functionality in Pydantic. It provides information on when the
deprecation was introduced and the expected version in which the corresponding functionality will be removed.

Attributes:
    message: Description of the warning.
    since: Pydantic version in what the deprecation was introduced.
    expected_removal: Pydantic version in what the corresponding functionality expected to be removed.

### PydanticExperimentalWarning
Module: `pydantic.warnings`

A Pydantic specific experimental functionality warning.

This warning is raised when using experimental functionality in Pydantic.
It is raised to warn users that the functionality may change or be removed in future versions of Pydantic.

### PydanticForbiddenQualifier
Module: `pydantic.errors`

An error raised if a forbidden type qualifier is found in a type annotation.

### PydanticImportError
Module: `pydantic.errors`

An error raised when an import fails due to module changes between V1 and V2.

Attributes:
    message: Description of the error.

### PydanticInvalidForJsonSchema
Module: `pydantic.errors`

An error raised during failures to generate a JSON schema for some `CoreSchema`.

Attributes:
    message: Description of the error.

### PydanticSchemaGenerationError
Module: `pydantic.errors`

An error raised during failures to generate a `CoreSchema` for some type.

Attributes:
    message: Description of the error.

### PydanticUndefinedAnnotation
Module: `pydantic.errors`

A subclass of `NameError` raised when handling undefined annotations during `CoreSchema` generation.

Attributes:
    name: Name of the error.
    message: Description of the error.

### PydanticUserError
Module: `pydantic.errors`

An error raised due to incorrect use of Pydantic.

### RedisDsn
Module: `pydantic.networks`

A type that will accept any Redis DSN.

* User info required
* TLD not required
* Host required (e.g., `rediss://:pass@localhost`)

### Secret
Module: `pydantic.types`

A generic base class used for defining a field with sensitive information that you do not want to be visible in logging or tracebacks.

You may either directly parametrize `Secret` with a type, or subclass from `Secret` with a parametrized type. The benefit of subclassing
is that you can define a custom `_display` method, which will be used for `repr()` and `str()` methods. The examples below demonstrate both
ways of using `Secret` to create a new secret type.

1. Directly parametrizing `Secret` with a type:

```python
from pydantic import BaseModel, Secret

SecretBool = Secret[bool]

class Model(BaseModel):
    secret_bool: SecretBool

m = Model(secret_bool=True)
print(m.model_dump())
#> {'secret_bool': Secret('**********')}

print(m.model_dump_json())
#> {"secret_bool":"**********"}

print(m.secret_bool.get_secret_value())
#> True
```

2. Subclassing from parametrized `Secret`:

```python
from datetime import date

from pydantic import BaseModel, Secret

class SecretDate(Secret[date]):
    def _display(self) -> str:
        return '****/**/**'

class Model(BaseModel):
    secret_date: SecretDate

m = Model(secret_date=date(2022, 1, 1))
print(m.model_dump())
#> {'secret_date': SecretDate('****/**/**')}

print(m.model_dump_json())
#> {"secret_date":"****/**/**"}

print(m.secret_date.get_secret_value())
#> 2022-01-01
```

The value returned by the `_display` method will be used for `repr()` and `str()`.

You can enforce constraints on the underlying type through annotations:
For example:

```python
from typing import Annotated

from pydantic import BaseModel, Field, Secret, ValidationError

SecretPosInt = Secret[Annotated[int, Field(gt=0, strict=True)]]

class Model(BaseModel):
    sensitive_int: SecretPosInt

m = Model(sensitive_int=42)
print(m.model_dump())
#> {'sensitive_int': Secret('**********')}

try:
    m = Model(sensitive_int=-42)  # (1)!
except ValidationError as exc_info:
    print(exc_info.errors(include_url=False, include_input=False))
    '''
    [
        {
            'type': 'greater_than',
            'loc': ('sensitive_int',),
            'msg': 'Input should be greater than 0',
            'ctx': {'gt': 0},
        }
    ]
    '''

try:
    m = Model(sensitive_int='42')  # (2)!
except ValidationError as exc_info:
    print(exc_info.errors(include_url=False, include_input=False))
    '''
    [
        {
            'type': 'int_type',
            'loc': ('sensitive_int',),
            'msg': 'Input should be a valid integer',
        }
    ]
    '''
```

1. The input value is not greater than 0, so it raises a validation error.
2. The input value is not an integer, so it raises a validation error because the `SecretPosInt` type has strict mode enabled.

### SecretBytes
Module: `pydantic.types`

A bytes used for storing sensitive information that you do not want to be visible in logging or tracebacks.

It displays `b'**********'` instead of the string value on `repr()` and `str()` calls.
When the secret value is nonempty, it is displayed as `b'**********'` instead of the underlying value in
calls to `repr()` and `str()`. If the value _is_ empty, it is displayed as `b''`.

```python
from pydantic import BaseModel, SecretBytes

class User(BaseModel):
    username: str
    password: SecretBytes

user = User(username='scolvin', password=b'password1')
#> username='scolvin' password=SecretBytes(b'**********')
print(user.password.get_secret_value())
#> b'password1'
print((SecretBytes(b'password'), SecretBytes(b'')))
#> (SecretBytes(b'**********'), SecretBytes(b''))
```

### SecretStr
Module: `pydantic.types`

A string used for storing sensitive information that you do not want to be visible in logging or tracebacks.

When the secret value is nonempty, it is displayed as `'**********'` instead of the underlying value in
calls to `repr()` and `str()`. If the value _is_ empty, it is displayed as `''`.

```python
from pydantic import BaseModel, SecretStr

class User(BaseModel):
    username: str
    password: SecretStr

user = User(username='scolvin', password='password1')

print(user)
#> username='scolvin' password=SecretStr('**********')
print(user.password.get_secret_value())
#> password1
print((SecretStr('password'), SecretStr('')))
#> (SecretStr('**********'), SecretStr(''))
```

As seen above, by default, [`SecretStr`][pydantic.types.SecretStr] (and [`SecretBytes`][pydantic.types.SecretBytes])
will be serialized as `**********` when serializing to json.

You can use the [`field_serializer`][pydantic.functional_serializers.field_serializer] to dump the
secret as plain-text when serializing to json.

```python
from pydantic import BaseModel, SecretBytes, SecretStr, field_serializer

class Model(BaseModel):
    password: SecretStr
    password_bytes: SecretBytes

    @field_serializer('password', 'password_bytes', when_used='json')
    def dump_secret(self, v):
        return v.get_secret_value()

model = Model(password='IAmSensitive', password_bytes=b'IAmSensitiveBytes')
print(model)
#> password=SecretStr('**********') password_bytes=SecretBytes(b'**********')
print(model.password)
#> **********
print(model.model_dump())
'''
{
    'password': SecretStr('**********'),
    'password_bytes': SecretBytes(b'**********'),
}
'''
print(model.model_dump_json())
#> {"password":"IAmSensitive","password_bytes":"IAmSensitiveBytes"}
```

### SerializationInfo
Module: `pydantic_core.core_schema`

Base class for protocol classes.

Protocol classes are defined as::

    class Proto(Protocol):
        def meth(self) -> int:
            ...

Such classes are primarily used with static type checkers that recognize
structural subtyping (static duck-typing).

For example::

    class C:
        def meth(self) -> int:
            return 0

    def func(x: Proto) -> int:
        return x.meth()

    func(C())  # Passes static type check

See PEP 544 for details. Protocol classes decorated with
@typing.runtime_checkable act as simple-minded runtime protocols that check
only the presence of given attributes, ignoring their type signatures.
Protocol classes can be generic, they are defined as::

    class GenProto(Protocol[T]):
        def meth(self) -> T:
            ...

#### Methods

**`mode_is_json(self) -> 'bool'`**

*No documentation available.*

### SerializeAsAny
Module: `pydantic.functional_serializers`

SerializeAsAny()

### SerializerFunctionWrapHandler
Module: `pydantic_core.core_schema`

Base class for protocol classes.

Protocol classes are defined as::

    class Proto(Protocol):
        def meth(self) -> int:
            ...

Such classes are primarily used with static type checkers that recognize
structural subtyping (static duck-typing).

For example::

    class C:
        def meth(self) -> int:
            return 0

    def func(x: Proto) -> int:
        return x.meth()

    func(C())  # Passes static type check

See PEP 544 for details. Protocol classes decorated with
@typing.runtime_checkable act as simple-minded runtime protocols that check
only the presence of given attributes, ignoring their type signatures.
Protocol classes can be generic, they are defined as::

    class GenProto(Protocol[T]):
        def meth(self) -> T:
            ...

### SkipValidation
Module: `pydantic.functional_validators`

If this is applied as an annotation (e.g., via `x: Annotated[int, SkipValidation]`), validation will be
    skipped. You can also use `SkipValidation[int]` as a shorthand for `Annotated[int, SkipValidation]`.

This can be useful if you want to use a type annotation for documentation/IDE/type-checking purposes,
and know that it is safe to skip validation for one or more of the fields.

Because this converts the validation schema to `any_schema`, subsequent annotation-applied transformations
may not have the expected effects. Therefore, when used, this annotation should generally be the final
annotation applied to a type.

### SnowflakeDsn
Module: `pydantic.networks`

A type that will accept any Snowflake DSN.

* User info required
* TLD not required
* Host required

### Strict
Module: `pydantic.types`

!!! abstract "Usage Documentation"
    [Strict Mode with `Annotated` `Strict`](../concepts/strict_mode.md#strict-mode-with-annotated-strict)

A field metadata class to indicate that a field should be validated in strict mode.
Use this class as an annotation via [`Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated), as seen below.

Attributes:
    strict: Whether to validate the field in strict mode.

Example:
    ```python
    from typing import Annotated

    from pydantic.types import Strict

    StrictBool = Annotated[bool, Strict()]
    ```

### StringConstraints
Module: `pydantic.types`

!!! abstract "Usage Documentation"
    [`StringConstraints`](../concepts/fields.md#string-constraints)

A field metadata class to apply constraints to `str` types.
Use this class as an annotation via [`Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated), as seen below.

Attributes:
    strip_whitespace: Whether to remove leading and trailing whitespace.
    to_upper: Whether to convert the string to uppercase.
    to_lower: Whether to convert the string to lowercase.
    strict: Whether to validate the string in strict mode.
    min_length: The minimum length of the string.
    max_length: The maximum length of the string.
    pattern: A regex pattern that the string must match.

Example:
    ```python
    from typing import Annotated

    from pydantic.types import StringConstraints

    ConstrainedStr = Annotated[str, StringConstraints(min_length=1, max_length=10)]
    ```

### Tag
Module: `pydantic.types`

Provides a way to specify the expected tag to use for a case of a (callable) discriminated union.

Also provides a way to label a union case in error messages.

When using a callable `Discriminator`, attach a `Tag` to each case in the `Union` to specify the tag that
should be used to identify that case. For example, in the below example, the `Tag` is used to specify that
if `get_discriminator_value` returns `'apple'`, the input should be validated as an `ApplePie`, and if it
returns `'pumpkin'`, the input should be validated as a `PumpkinPie`.

The primary role of the `Tag` here is to map the return value from the callable `Discriminator` function to
the appropriate member of the `Union` in question.

```python
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Discriminator, Tag

class Pie(BaseModel):
    time_to_cook: int
    num_ingredients: int

class ApplePie(Pie):
    fruit: Literal['apple'] = 'apple'

class PumpkinPie(Pie):
    filling: Literal['pumpkin'] = 'pumpkin'

def get_discriminator_value(v: Any) -> str:
    if isinstance(v, dict):
        return v.get('fruit', v.get('filling'))
    return getattr(v, 'fruit', getattr(v, 'filling', None))

class ThanksgivingDinner(BaseModel):
    dessert: Annotated[
        Union[
            Annotated[ApplePie, Tag('apple')],
            Annotated[PumpkinPie, Tag('pumpkin')],
        ],
        Discriminator(get_discriminator_value),
    ]

apple_variation = ThanksgivingDinner.model_validate(
    {'dessert': {'fruit': 'apple', 'time_to_cook': 60, 'num_ingredients': 8}}
)
print(repr(apple_variation))
'''
ThanksgivingDinner(dessert=ApplePie(time_to_cook=60, num_ingredients=8, fruit='apple'))
'''

pumpkin_variation = ThanksgivingDinner.model_validate(
    {
        'dessert': {
            'filling': 'pumpkin',
            'time_to_cook': 40,
            'num_ingredients': 6,
        }
    }
)
print(repr(pumpkin_variation))
'''
ThanksgivingDinner(dessert=PumpkinPie(time_to_cook=40, num_ingredients=6, filling='pumpkin'))
'''
```

!!! note
    You must specify a `Tag` for every case in a `Tag` that is associated with a
    callable `Discriminator`. Failing to do so will result in a `PydanticUserError` with code
    [`callable-discriminator-no-tag`](../errors/usage_errors.md#callable-discriminator-no-tag).

See the [Discriminated Unions] concepts docs for more details on how to use `Tag`s.

[Discriminated Unions]: ../concepts/unions.md#discriminated-unions

### TypeAdapter
Module: `pydantic.type_adapter`

!!! abstract "Usage Documentation"
    [`TypeAdapter`](../concepts/type_adapter.md)

Type adapters provide a flexible way to perform validation and serialization based on a Python type.

A `TypeAdapter` instance exposes some of the functionality from `BaseModel` instance methods
for types that do not have such methods (such as dataclasses, primitive types, and more).

**Note:** `TypeAdapter` instances are not types, and cannot be used as type annotations for fields.

Args:
    type: The type associated with the `TypeAdapter`.
    config: Configuration for the `TypeAdapter`, should be a dictionary conforming to
        [`ConfigDict`][pydantic.config.ConfigDict].

        !!! note
            You cannot provide a configuration when instantiating a `TypeAdapter` if the type you're using
            has its own config that cannot be overridden (ex: `BaseModel`, `TypedDict`, and `dataclass`). A
            [`type-adapter-config-unused`](../errors/usage_errors.md#type-adapter-config-unused) error will
            be raised in this case.
    _parent_depth: Depth at which to search for the [parent frame][frame-objects]. This frame is used when
        resolving forward annotations during schema building, by looking for the globals and locals of this
        frame. Defaults to 2, which will result in the frame where the `TypeAdapter` was instantiated.

        !!! note
            This parameter is named with an underscore to suggest its private nature and discourage use.
            It may be deprecated in a minor version, so we only recommend using it if you're comfortable
            with potential change in behavior/support. It's default value is 2 because internally,
            the `TypeAdapter` class makes another call to fetch the frame.
    module: The module that passes to plugin if provided.

Attributes:
    core_schema: The core schema for the type.
    validator: The schema validator for the type.
    serializer: The schema serializer for the type.
    pydantic_complete: Whether the core schema for the type is successfully built.

??? tip "Compatibility with `mypy`"
    Depending on the type used, `mypy` might raise an error when instantiating a `TypeAdapter`. As a workaround, you can explicitly
    annotate your variable:

    ```py
    from typing import Union

    from pydantic import TypeAdapter

    ta: TypeAdapter[Union[str, int]] = TypeAdapter(Union[str, int])  # type: ignore[arg-type]
    ```

??? info "Namespace management nuances and implementation details"

    Here, we collect some notes on namespace management, and subtle differences from `BaseModel`:

    `BaseModel` uses its own `__module__` to find out where it was defined
    and then looks for symbols to resolve forward references in those globals.
    On the other hand, `TypeAdapter` can be initialized with arbitrary objects,
    which may not be types and thus do not have a `__module__` available.
    So instead we look at the globals in our parent stack frame.

    It is expected that the `ns_resolver` passed to this function will have the correct
    namespace for the type we're adapting. See the source code for `TypeAdapter.__init__`
    and `TypeAdapter.rebuild` for various ways to construct this namespace.

    This works for the case where this function is called in a module that
    has the target of forward references in its scope, but
    does not always work for more complex cases.

    For example, take the following:

    ```python {title="a.py"}
    IntList = list[int]
    OuterDict = dict[str, 'IntList']
    ```

    ```python {test="skip" title="b.py"}
    from a import OuterDict

    from pydantic import TypeAdapter

    IntList = int  # replaces the symbol the forward reference is looking for
    v = TypeAdapter(OuterDict)
    v({'x': 1})  # should fail but doesn't
    ```

    If `OuterDict` were a `BaseModel`, this would work because it would resolve
    the forward reference within the `a.py` namespace.
    But `TypeAdapter(OuterDict)` can't determine what module `OuterDict` came from.

    In other words, the assumption that _all_ forward references exist in the
    module we are being called from is not technically always true.
    Although most of the time it is and it works fine for recursive models and such,
    `BaseModel`'s behavior isn't perfect either and _can_ break in similar ways,
    so there is no right or wrong between the two.

    But at the very least this behavior is _subtly_ different from `BaseModel`'s.

#### Methods

**`rebuild(self, *, force: 'bool' = False, raise_errors: 'bool' = True, _parent_namespace_depth: 'int' = 2, _types_namespace: '_namespace_utils.MappingNamespace | None' = None) -> 'bool | None'`**

Try to rebuild the pydantic-core schema for the adapter's type.

This may be necessary when one of the annotations is a ForwardRef which could not be resolved during
the initial attempt to build the schema, and automatic rebuilding fails.

Args:
force: Whether to force the rebuilding of the type adapter's schema, defaults to `False`.
raise_errors: Whether to raise errors, defaults to `True`.
_parent_namespace_depth: Depth at which to search for the [parent frame][frame-objects]. This
frame is used when resolving forward annotations during schema rebuilding, by looking for
the locals of this frame. Defaults to 2, which will result in the frame where the method
was called.
_types_namespace: An explicit types namespace to use, instead of using the local namespace
from the parent frame. Defaults to `None`.

Returns:
Returns `None` if the schema is already "complete" and rebuilding was not required.
If rebuilding _was_ required, returns `True` if rebuilding was successful, otherwise `False`.

**`validate_python(self, object: 'Any', /, *, strict: 'bool | None' = None, from_attributes: 'bool | None' = None, context: 'dict[str, Any] | None' = None, experimental_allow_partial: "bool | Literal['off', 'on', 'trailing-strings']" = False, by_alias: 'bool | None' = None, by_name: 'bool | None' = None) -> 'T'`**

Validate a Python object against the model.

Args:
object: The Python object to validate against the model.
strict: Whether to strictly check types.
from_attributes: Whether to extract data from object attributes.
context: Additional context to pass to the validator.
experimental_allow_partial: **Experimental** whether to enable
[partial validation](../concepts/experimental.md#partial-validation), e.g. to process streams.
* False / 'off': Default behavior, no partial validation.
* True / 'on': Enable partial validation.
* 'trailing-strings': Enable partial validation and allow trailing strings in the input.
by_alias: Whether to use the field's alias when validating against the provided input data.
by_name: Whether to use the field's name when validating against the provided input data.

!!! note
When using `TypeAdapter` with a Pydantic `dataclass`, the use of the `from_attributes`
argument is not supported.

Returns:
The validated object.

**`validate_json(self, data: 'str | bytes | bytearray', /, *, strict: 'bool | None' = None, context: 'dict[str, Any] | None' = None, experimental_allow_partial: "bool | Literal['off', 'on', 'trailing-strings']" = False, by_alias: 'bool | None' = None, by_name: 'bool | None' = None) -> 'T'`**

!!! abstract "Usage Documentation"
[JSON Parsing](../concepts/json.md#json-parsing)

Validate a JSON string or bytes against the model.

Args:
data: The JSON data to validate against the model.
strict: Whether to strictly check types.
context: Additional context to use during validation.
experimental_allow_partial: **Experimental** whether to enable
[partial validation](../concepts/experimental.md#partial-validation), e.g. to process streams.
* False / 'off': Default behavior, no partial validation.
* True / 'on': Enable partial validation.
* 'trailing-strings': Enable partial validation and allow trailing strings in the input.
by_alias: Whether to use the field's alias when validating against the provided input data.
by_name: Whether to use the field's name when validating against the provided input data.

Returns:
The validated object.

**`validate_strings(self, obj: 'Any', /, *, strict: 'bool | None' = None, context: 'dict[str, Any] | None' = None, experimental_allow_partial: "bool | Literal['off', 'on', 'trailing-strings']" = False, by_alias: 'bool | None' = None, by_name: 'bool | None' = None) -> 'T'`**

Validate object contains string data against the model.

Args:
obj: The object contains string data to validate.
strict: Whether to strictly check types.
context: Additional context to use during validation.
experimental_allow_partial: **Experimental** whether to enable
[partial validation](../concepts/experimental.md#partial-validation), e.g. to process streams.
* False / 'off': Default behavior, no partial validation.
* True / 'on': Enable partial validation.
* 'trailing-strings': Enable partial validation and allow trailing strings in the input.
by_alias: Whether to use the field's alias when validating against the provided input data.
by_name: Whether to use the field's name when validating against the provided input data.

Returns:
The validated object.

**`get_default_value(self, *, strict: 'bool | None' = None, context: 'dict[str, Any] | None' = None) -> 'Some[T] | None'`**

Get the default value for the wrapped type.

Args:
strict: Whether to strictly check types.
context: Additional context to pass to the validator.

Returns:
The default value wrapped in a `Some` if there is one or None if not.

**`dump_python(self, instance: 'T', /, *, mode: "Literal['json', 'python']" = 'python', include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, by_alias: 'bool | None' = None, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, round_trip: 'bool' = False, warnings: "bool | Literal['none', 'warn', 'error']" = True, fallback: 'Callable[[Any], Any] | None' = None, serialize_as_any: 'bool' = False, context: 'dict[str, Any] | None' = None) -> 'Any'`**

Dump an instance of the adapted type to a Python object.

Args:
instance: The Python object to serialize.
mode: The output format.
include: Fields to include in the output.
exclude: Fields to exclude from the output.
by_alias: Whether to use alias names for field names.
exclude_unset: Whether to exclude unset fields.
exclude_defaults: Whether to exclude fields with default values.
exclude_none: Whether to exclude fields with None values.
round_trip: Whether to output the serialized data in a way that is compatible with deserialization.
warnings: How to handle serialization errors. False/"none" ignores them, True/"warn" logs errors,
"error" raises a [`PydanticSerializationError`][pydantic_core.PydanticSerializationError].
fallback: A function to call when an unknown value is encountered. If not provided,
a [`PydanticSerializationError`][pydantic_core.PydanticSerializationError] error is raised.
serialize_as_any: Whether to serialize fields with duck-typing serialization behavior.
context: Additional context to pass to the serializer.

Returns:
The serialized object.

**`dump_json(self, instance: 'T', /, *, indent: 'int | None' = None, include: 'IncEx | None' = None, exclude: 'IncEx | None' = None, by_alias: 'bool | None' = None, exclude_unset: 'bool' = False, exclude_defaults: 'bool' = False, exclude_none: 'bool' = False, round_trip: 'bool' = False, warnings: "bool | Literal['none', 'warn', 'error']" = True, fallback: 'Callable[[Any], Any] | None' = None, serialize_as_any: 'bool' = False, context: 'dict[str, Any] | None' = None) -> 'bytes'`**

!!! abstract "Usage Documentation"
[JSON Serialization](../concepts/json.md#json-serialization)

Serialize an instance of the adapted type to JSON.

Args:
instance: The instance to be serialized.
indent: Number of spaces for JSON indentation.
include: Fields to include.
exclude: Fields to exclude.
by_alias: Whether to use alias names for field names.
exclude_unset: Whether to exclude unset fields.
exclude_defaults: Whether to exclude fields with default values.
exclude_none: Whether to exclude fields with a value of `None`.
round_trip: Whether to serialize and deserialize the instance to ensure round-tripping.
warnings: How to handle serialization errors. False/"none" ignores them, True/"warn" logs errors,
"error" raises a [`PydanticSerializationError`][pydantic_core.PydanticSerializationError].
fallback: A function to call when an unknown value is encountered. If not provided,
a [`PydanticSerializationError`][pydantic_core.PydanticSerializationError] error is raised.
serialize_as_any: Whether to serialize fields with duck-typing serialization behavior.
context: Additional context to pass to the serializer.

Returns:
The JSON representation of the given instance as bytes.

**`json_schema(self, *, by_alias: 'bool' = True, ref_template: 'str' = '#/$defs/{model}', schema_generator: 'type[GenerateJsonSchema]' = <class 'pydantic.json_schema.GenerateJsonSchema'>, mode: 'JsonSchemaMode' = 'validation') -> 'dict[str, Any]'`**

Generate a JSON schema for the adapted type.

Args:
by_alias: Whether to use alias names for field names.
ref_template: The format string used for generating $ref strings.
schema_generator: The generator class used for creating the schema.
mode: The mode to use for schema generation.

Returns:
The JSON schema for the model as a dictionary.

**`json_schemas(inputs: 'Iterable[tuple[JsonSchemaKeyT, JsonSchemaMode, TypeAdapter[Any]]]', /, *, by_alias: 'bool' = True, title: 'str | None' = None, description: 'str | None' = None, ref_template: 'str' = '#/$defs/{model}', schema_generator: 'type[GenerateJsonSchema]' = <class 'pydantic.json_schema.GenerateJsonSchema'>) -> 'tuple[dict[tuple[JsonSchemaKeyT, JsonSchemaMode], JsonSchemaValue], JsonSchemaValue]'`**

Generate a JSON schema including definitions from multiple type adapters.

Args:
inputs: Inputs to schema generation. The first two items will form the keys of the (first)
output mapping; the type adapters will provide the core schemas that get converted into
definitions in the output JSON schema.
by_alias: Whether to use alias names.
title: The title for the schema.
description: The description for the schema.
ref_template: The format string used for generating $ref strings.
schema_generator: The generator class used for creating the schema.

Returns:
A tuple where:

- The first element is a dictionary whose keys are tuples of JSON schema key type and JSON mode, and
whose values are the JSON schema corresponding to that pair of inputs. (These schemas may have
JsonRef references to definitions that are defined in the second returned element.)
- The second element is a JSON schema containing all definitions referenced in the first returned
element, along with the optional title and description keys.

### UrlConstraints
Module: `pydantic.networks`

Url constraints.

Attributes:
    max_length: The maximum length of the url. Defaults to `None`.
    allowed_schemes: The allowed schemes. Defaults to `None`.
    host_required: Whether the host is required. Defaults to `None`.
    default_host: The default host. Defaults to `None`.
    default_port: The default port. Defaults to `None`.
    default_path: The default path. Defaults to `None`.

### ValidationError
Module: `pydantic_core._pydantic_core`

#### Methods

**`from_exception_data(cls, /, title, line_errors, input_type='python', hide_input=False)`**

*No documentation available.*

**`error_count(self, /)`**

*No documentation available.*

**`errors(self, /, *, include_url=True, include_context=True, include_input=True)`**

*No documentation available.*

**`json(self, /, *, indent=None, include_url=True, include_context=True, include_input=True)`**

*No documentation available.*

### ValidationInfo
Module: `pydantic_core.core_schema`

Argument passed to validation functions.

### ValidatorFunctionWrapHandler
Module: `pydantic_core.core_schema`

Base class for protocol classes.

Protocol classes are defined as::

    class Proto(Protocol):
        def meth(self) -> int:
            ...

Such classes are primarily used with static type checkers that recognize
structural subtyping (static duck-typing).

For example::

    class C:
        def meth(self) -> int:
            return 0

    def func(x: Proto) -> int:
        return x.meth()

    func(C())  # Passes static type check

See PEP 544 for details. Protocol classes decorated with
@typing.runtime_checkable act as simple-minded runtime protocols that check
only the presence of given attributes, ignoring their type signatures.
Protocol classes can be generic, they are defined as::

    class GenProto(Protocol[T]):
        def meth(self) -> T:
            ...

### WebsocketUrl
Module: `pydantic.networks`

A type that will accept any ws or wss URL.

* TLD not required
* Host not required
* Max length 2083

### WithJsonSchema
Module: `pydantic.json_schema`

!!! abstract "Usage Documentation"
    [`WithJsonSchema` Annotation](../concepts/json_schema.md#withjsonschema-annotation)

Add this as an annotation on a field to override the (base) JSON schema that would be generated for that field.
This provides a way to set a JSON schema for types that would otherwise raise errors when producing a JSON schema,
such as Callable, or types that have an is-instance core schema, without needing to go so far as creating a
custom subclass of pydantic.json_schema.GenerateJsonSchema.
Note that any _modifications_ to the schema that would normally be made (such as setting the title for model fields)
will still be performed.

If `mode` is set this will only apply to that schema generation mode, allowing you
to set different json schemas for validation and serialization.

### WrapSerializer
Module: `pydantic.functional_serializers`

Wrap serializers receive the raw inputs along with a handler function that applies the standard serialization
logic, and can modify the resulting value before returning it as the final output of serialization.

For example, here's a scenario in which a wrap serializer transforms timezones to UTC **and** utilizes the existing `datetime` serialization logic.

```python
from datetime import datetime, timezone
from typing import Annotated, Any

from pydantic import BaseModel, WrapSerializer

class EventDatetime(BaseModel):
    start: datetime
    end: datetime

def convert_to_utc(value: Any, handler, info) -> dict[str, datetime]:
    # Note that `handler` can actually help serialize the `value` for
    # further custom serialization in case it's a subclass.
    partial_result = handler(value, info)
    if info.mode == 'json':
        return {
            k: datetime.fromisoformat(v).astimezone(timezone.utc)
            for k, v in partial_result.items()
        }
    return {k: v.astimezone(timezone.utc) for k, v in partial_result.items()}

UTCEventDatetime = Annotated[EventDatetime, WrapSerializer(convert_to_utc)]

class EventModel(BaseModel):
    event_datetime: UTCEventDatetime

dt = EventDatetime(
    start='2024-01-01T07:00:00-08:00', end='2024-01-03T20:00:00+06:00'
)
event = EventModel(event_datetime=dt)
print(event.model_dump())
'''
{
    'event_datetime': {
        'start': datetime.datetime(
            2024, 1, 1, 15, 0, tzinfo=datetime.timezone.utc
        ),
        'end': datetime.datetime(
            2024, 1, 3, 14, 0, tzinfo=datetime.timezone.utc
        ),
    }
}
'''

print(event.model_dump_json())
'''
{"event_datetime":{"start":"2024-01-01T15:00:00Z","end":"2024-01-03T14:00:00Z"}}
'''
```

Attributes:
    func: The serializer function to be wrapped.
    return_type: The return type for the function. If omitted it will be inferred from the type annotation.
    when_used: Determines when this serializer should be used. Accepts a string with values `'always'`,
        `'unless-none'`, `'json'`, and `'json-unless-none'`. Defaults to 'always'.

### WrapValidator
Module: `pydantic.functional_validators`

!!! abstract "Usage Documentation"
    [field *wrap* validators](../concepts/validators.md#field-wrap-validator)

A metadata class that indicates that a validation should be applied **around** the inner validation logic.

Attributes:
    func: The validator function.
    json_schema_input_type: The input type of the function. This is only used to generate the appropriate
        JSON Schema (in validation mode).

```python
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ValidationError, WrapValidator

def validate_timestamp(v, handler):
    if v == 'now':
        # we don't want to bother with further validation, just return the new value
        return datetime.now()
    try:
        return handler(v)
    except ValidationError:
        # validation failed, in this case we want to return a default value
        return datetime(2000, 1, 1)

MyTimestamp = Annotated[datetime, WrapValidator(validate_timestamp)]

class Model(BaseModel):
    a: MyTimestamp

print(Model(a='now').a)
#> 2032-01-02 03:04:05.000006
print(Model(a='invalid').a)
#> 2000-01-01 00:00:00
```
