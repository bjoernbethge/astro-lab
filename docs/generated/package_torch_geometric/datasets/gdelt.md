# gdelt

Part of `torch_geometric.datasets`
Module: `torch_geometric.datasets.gdelt`

## Functions (2)

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

### `read_txt_array(path: str, sep: Optional[str] = None, start: int = 0, end: Optional[int] = None, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor`

## Classes (3)

### `EventDataset`

Dataset base class for creating graph datasets which easily fit
into CPU memory.
See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
create_dataset.html#creating-in-memory-datasets>`__ for the accompanying
tutorial.

Args:
    root (str, optional): Root directory where the dataset should be saved.
        (optional: :obj:`None`)
    transform (callable, optional): A function/transform that takes in a
        :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object and returns a
        transformed version.
        The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        a :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object and returns a
        transformed version.
        The data object will be transformed before being saved to disk.
        (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in a
        :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object and returns a
        boolean value, indicating whether the data object should be
        included in the final dataset. (default: :obj:`None`)
    log (bool, optional): Whether to print any console output while
        downloading and processing the dataset. (default: :obj:`True`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`process_events(self) -> torch.Tensor`**

### `GDELT`

The Global Database of Events, Language, and Tone (GDELT) dataset used
in the, *e.g.*, `"Recurrent Event Network for Reasoning over Temporal
Knowledge Graphs" <https://arxiv.org/abs/1904.05530>`_ paper, consisting of
events collected from 1/1/2018 to 1/31/2018 (15 minutes time granularity).

Args:
    root (str): Root directory where the dataset should be saved.
    split (str, optional): If :obj:`"train"`, loads the training dataset.
        If :obj:`"val"`, loads the validation dataset.
        If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process_events(self) -> torch.Tensor`**

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
