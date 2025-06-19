# citation_full

Part of `torch_geometric.datasets`
Module: `torch_geometric.datasets.citation_full`

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

### `read_npz(path: str, to_undirected: bool = True) -> torch_geometric.data.data.Data`

## Classes (3)

### `CitationFull`

The full citation network datasets from the
`"Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via
Ranking" <https://arxiv.org/abs/1707.03815>`_ paper.
Nodes represent documents and edges represent citation links.
Datasets include :obj:`"Cora"`, :obj:`"Cora_ML"`, :obj:`"CiteSeer"`,
:obj:`"DBLP"`, :obj:`"PubMed"`.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"Cora"`, :obj:`"Cora_ML"`
        :obj:`"CiteSeer"`, :obj:`"DBLP"`, :obj:`"PubMed"`).
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    to_undirected (bool, optional): Whether the original graph is
        converted to an undirected one. (default: :obj:`True`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - Name
      - #nodes
      - #edges
      - #features
      - #classes
    * - Cora
      - 19,793
      - 126,842
      - 8,710
      - 70
    * - Cora_ML
      - 2,995
      - 16,316
      - 2,879
      - 7
    * - CiteSeer
      - 4,230
      - 10,674
      - 602
      - 6
    * - DBLP
      - 17,716
      - 105,734
      - 1,639
      - 4
    * - PubMed
      - 19,717
      - 88,648
      - 500
      - 3

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `CoraFull`

Alias for :class:`~torch_geometric.datasets.CitationFull` with
:obj:`name="Cora"`.

**STATS:**

.. list-table::
    :widths: 10 10 10 10
    :header-rows: 1

    * - #nodes
      - #edges
      - #features
      - #classes
    * - 19,793
      - 126,842
      - 8,710
      - 70

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

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
