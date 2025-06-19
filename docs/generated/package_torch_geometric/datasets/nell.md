# nell

Part of `torch_geometric.datasets`
Module: `torch_geometric.datasets.nell`

## Functions (3)

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

### `extract_tar(path: str, folder: str, mode: str = 'r:gz', log: bool = True) -> None`

Extracts a tar archive to a specific folder.

Args:
    path (str): The path to the tar archive.
    folder (str): The folder.
    mode (str, optional): The compression mode. (default: :obj:`"r:gz"`)
    log (bool, optional): If :obj:`False`, will not print anything to the
        console. (default: :obj:`True`)

### `read_planetoid_data(folder: str, prefix: str) -> torch_geometric.data.data.Data`

## Classes (2)

### `InMemoryDataset`

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

- **`len(self) -> int`**
  Returns the number of data objects stored in the dataset.

- **`get(self, idx: int) -> torch_geometric.data.data.BaseData`**
  Gets the data object at index :obj:`idx`.

- **`load(self, path: str, data_cls: Type[torch_geometric.data.data.BaseData] = <class 'torch_geometric.data.data.Data'>) -> None`**
  Loads the dataset from the file path :obj:`path`.

### `NELL`

The NELL dataset, a knowledge graph from the
`"Toward an Architecture for Never-Ending Language Learning"
<https://www.cs.cmu.edu/~acarlson/papers/carlson-aaai10.pdf>`_ paper.
The dataset is processed as in the
`"Revisiting Semi-Supervised Learning with Graph Embeddings"
<https://arxiv.org/abs/1603.08861>`_ paper.

.. note::

    Entity nodes are described by sparse feature vectors of type
    :class:`torch.sparse_csr_tensor`.

Args:
    root (str): Root directory where the dataset should be saved.
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10
    :header-rows: 1

    * - #nodes
      - #edges
      - #features
      - #classes
    * - 65,755
      - 251,550
      - 61,278
      - 186

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.
