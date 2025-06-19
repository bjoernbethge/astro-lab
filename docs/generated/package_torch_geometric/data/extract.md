# extract

Part of `torch_geometric.data`
Module: `torch_geometric.data.extract`

## Functions (5)

### `extract_bz2(path: str, folder: str, log: bool = True) -> None`

Extracts a bz2 archive to a specific folder.

Args:
    path (str): The path to the tar archive.
    folder (str): The folder.
    log (bool, optional): If :obj:`False`, will not print anything to the
        console. (default: :obj:`True`)

### `extract_gz(path: str, folder: str, log: bool = True) -> None`

Extracts a gz archive to a specific folder.

Args:
    path (str): The path to the tar archive.
    folder (str): The folder.
    log (bool, optional): If :obj:`False`, will not print anything to the
        console. (default: :obj:`True`)

### `extract_tar(path: str, folder: str, mode: str = 'r:gz', log: bool = True) -> None`

Extracts a tar archive to a specific folder.

Args:
    path (str): The path to the tar archive.
    folder (str): The folder.
    mode (str, optional): The compression mode. (default: :obj:`"r:gz"`)
    log (bool, optional): If :obj:`False`, will not print anything to the
        console. (default: :obj:`True`)

### `extract_zip(path: str, folder: str, log: bool = True) -> None`

Extracts a zip archive to a specific folder.

Args:
    path (str): The path to the tar archive.
    folder (str): The folder.
    log (bool, optional): If :obj:`False`, will not print anything to the
        console. (default: :obj:`True`)

### `maybe_log(path: str, log: bool = True) -> None`
