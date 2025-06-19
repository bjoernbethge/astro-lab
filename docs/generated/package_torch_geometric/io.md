# io Submodule

Part of the `torch_geometric` package
Module: `torch_geometric.io`

## Functions (12)

### `parse_npz(f: Dict[str, Any], to_undirected: bool = True) -> torch_geometric.data.data.Data`

### `parse_sdf(src: str) -> torch_geometric.data.data.Data`

### `parse_txt_array(src: List[str], sep: Optional[str] = None, start: int = 0, end: Optional[int] = None, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor`

### `read_npz(path: str, to_undirected: bool = True) -> torch_geometric.data.data.Data`

### `read_obj(in_file: str) -> Optional[torch_geometric.data.data.Data]`

### `read_off(path: str) -> torch_geometric.data.data.Data`

Reads an OFF (Object File Format) file, returning both the position of
nodes and their connectivity in a :class:`torch_geometric.data.Data`
object.

Args:
    path (str): The path to the file.

### `read_planetoid_data(folder: str, prefix: str) -> torch_geometric.data.data.Data`

### `read_ply(path: str) -> torch_geometric.data.data.Data`

### `read_sdf(path: str) -> torch_geometric.data.data.Data`

### `read_tu_data(folder: str, prefix: str) -> Tuple[torch_geometric.data.data.Data, Dict[str, torch.Tensor], Dict[str, int]]`

### `read_txt_array(path: str, sep: Optional[str] = None, start: int = 0, end: Optional[int] = None, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor`

### `write_off(data: torch_geometric.data.data.Data, path: str) -> None`

Writes a :class:`torch_geometric.data.Data` object to an OFF (Object
File Format) file.

Args:
    data (:class:`torch_geometric.data.Data`): The data object.
    path (str): The path to the file.

## Nested Submodules (9)

Each nested submodule is documented in a separate file:

### [fs](./io/fs.md)
Module: `torch_geometric.io.fs`

*Contains: 17 functions, 1 classes*

### [npz](./io/npz.md)
Module: `torch_geometric.io.npz`

*Contains: 4 functions, 2 classes*

### [obj](./io/obj.md)
Module: `torch_geometric.io.obj`

*Contains: 2 functions, 1 classes*

### [off](./io/off.md)
Module: `torch_geometric.io.off`

*Contains: 5 functions, 2 classes*

### [planetoid](./io/planetoid.md)
Module: `torch_geometric.io.planetoid`

*Contains: 8 functions, 3 classes*

### [ply](./io/ply.md)
Module: `torch_geometric.io.ply`

*Contains: 1 functions, 1 classes*

### [sdf](./io/sdf.md)
Module: `torch_geometric.io.sdf`

*Contains: 5 functions, 1 classes*

### [tu](./io/tu.md)
Module: `torch_geometric.io.tu`

*Contains: 9 functions, 2 classes*

### [txt_array](./io/txt_array.md)
Module: `torch_geometric.io.txt_array`

*Contains: 2 functions, 1 classes*
