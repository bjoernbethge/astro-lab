# fs

Part of `torch_geometric.io`
Module: `torch_geometric.io.fs`

## Functions (17)

### `cp(path1: str, path2: str, extract: bool = False, log: bool = True, use_cache: bool = True, clear_cache: bool = True) -> None`

### `exists(path: str) -> bool`

### `get_fs(path: str) -> fsspec.spec.AbstractFileSystem`

Get filesystem backend given a path URI to the resource.

Here are some common example paths and dispatch result:

* :obj:`"/home/file"` ->
  :class:`fsspec.implementations.local.LocalFileSystem`
* :obj:`"memory://home/file"` ->
  :class:`fsspec.implementations.memory.MemoryFileSystem`
* :obj:`"https://home/file"` ->
  :class:`fsspec.implementations.http.HTTPFileSystem`
* :obj:`"gs://home/file"` -> :class:`gcsfs.GCSFileSystem`
* :obj:`"s3://home/file"` -> :class:`s3fs.S3FileSystem`

A full list of supported backend implementations of :class:`fsspec` can be
found `here <https://github.com/fsspec/filesystem_spec/blob/master/fsspec/
registry.py#L62>`_.

The backend dispatch logic can be updated with custom backends following
`this tutorial <https://filesystem-spec.readthedocs.io/en/latest/
developer.html#implementing-a-backend>`_.

Args:
    path (str): The URI to the filesystem location, *e.g.*,
        :obj:`"gs://home/me/file"`, :obj:`"s3://..."`.

### `glob(path: str) -> List[str]`

### `isdir(path: str) -> bool`

### `isdisk(path: str) -> bool`

### `isfile(path: str) -> bool`

### `islocal(path: str) -> bool`

### `ls(path: str, detail: bool = False) -> Union[List[str], List[Dict[str, Any]]]`

### `makedirs(path: str, exist_ok: bool = True) -> None`

### `mv(path1: str, path2: str) -> None`

### `normpath(path: str) -> str`

### `overload(func)`

Decorator for overloaded functions/methods.

In a stub file, place two or more stub definitions for the same
function in a row, each decorated with @overload.

For example::

    @overload
    def utf8(value: None) -> None: ...
    @overload
    def utf8(value: bytes) -> bytes: ...
    @overload
    def utf8(value: str) -> bytes: ...

In a non-stub file (i.e. a regular .py file), do the same but
follow it with an implementation.  The implementation should *not*
be decorated with @overload::

    @overload
    def utf8(value: None) -> None: ...
    @overload
    def utf8(value: bytes) -> bytes: ...
    @overload
    def utf8(value: str) -> bytes: ...
    def utf8(value):
        ...  # implementation goes here

The overloads for a function can be retrieved at runtime using the
get_overloads() function.

### `rm(path: str, recursive: bool = True) -> None`

### `torch_load(path: str, map_location: Any = None) -> Any`

### `torch_save(data: Any, path: str) -> None`

### `uuid4()`

Generate a random UUID.

## Classes (1)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.
