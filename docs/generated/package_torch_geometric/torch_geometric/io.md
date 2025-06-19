# io

Part of `torch_geometric.torch_geometric`
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
