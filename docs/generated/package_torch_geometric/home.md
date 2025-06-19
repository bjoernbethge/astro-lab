# home Submodule

Part of the `torch_geometric` package
Module: `torch_geometric.home`

## Functions (2)

### `get_home_dir() -> str`

Get the cache directory used for storing all :pyg:`PyG`-related data.

If :meth:`set_home_dir` is not called, the path is given by the environment
variable :obj:`$PYG_HOME` which defaults to :obj:`"~/.cache/pyg"`.

### `set_home_dir(path: str) -> None`

Set the cache directory used for storing all :pyg:`PyG`-related data.

Args:
    path (str): The path to a local folder.

## Important Data Types (1)

### `Optional`
**Type**: `<class 'typing._SpecialForm'>`

Optional[X] is equivalent to Union[X, None].

*(callable)*
