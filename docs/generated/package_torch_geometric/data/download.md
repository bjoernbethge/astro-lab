# download

Part of `torch_geometric.data`
Module: `torch_geometric.data.download`

## Functions (2)

### `download_google_url(id: str, folder: str, filename: str, log: bool = True)`

Downloads the content of a Google Drive ID to a specific folder.

### `download_url(url: str, folder: str, log: bool = True, filename: Optional[str] = None)`

Downloads the content of an URL to a specific folder.

Args:
    url (str): The URL.
    folder (str): The folder.
    log (bool, optional): If :obj:`False`, will not print anything to the
        console. (default: :obj:`True`)
    filename (str, optional): The filename of the downloaded file. If set
        to :obj:`None`, will correspond to the filename given by the URL.
        (default: :obj:`None`)
