# molecule_net

Part of `torch_geometric.datasets`
Module: `torch_geometric.datasets.molecule_net`

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

### `extract_gz(path: str, folder: str, log: bool = True) -> None`

Extracts a gz archive to a specific folder.

Args:
    path (str): The path to the tar archive.
    folder (str): The folder.
    log (bool, optional): If :obj:`False`, will not print anything to the
        console. (default: :obj:`True`)

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

### `MoleculeNet`

The `MoleculeNet <http://moleculenet.org/datasets-1>`_ benchmark
collection  from the `"MoleculeNet: A Benchmark for Molecular Machine
Learning" <https://arxiv.org/abs/1703.00564>`_ paper, containing datasets
from physical chemistry, biophysics and physiology.
All datasets come with the additional node and edge features introduced by
the :ogb:`null`
`Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`_.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"ESOL"`, :obj:`"FreeSolv"`,
        :obj:`"Lipo"`, :obj:`"PCBA"`, :obj:`"MUV"`, :obj:`"HIV"`,
        :obj:`"BACE"`, :obj:`"BBBP"`, :obj:`"Tox21"`, :obj:`"ToxCast"`,
        :obj:`"SIDER"`, :obj:`"ClinTox"`).
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
    from_smiles (callable, optional): A custom function that takes a SMILES
        string and outputs a :obj:`~torch_geometric.data.Data` object.
        If not set, defaults to :meth:`~torch_geometric.utils.from_smiles`.
        (default: :obj:`None`)

**STATS:**

.. list-table::
    :widths: 20 10 10 10 10 10
    :header-rows: 1

    * - Name
      - #graphs
      - #nodes
      - #edges
      - #features
      - #classes
    * - ESOL
      - 1,128
      - ~13.3
      - ~27.4
      - 9
      - 1
    * - FreeSolv
      - 642
      - ~8.7
      - ~16.8
      - 9
      - 1
    * - Lipophilicity
      - 4,200
      - ~27.0
      - ~59.0
      - 9
      - 1
    * - PCBA
      - 437,929
      - ~26.0
      - ~56.2
      - 9
      - 128
    * - MUV
      - 93,087
      - ~24.2
      - ~52.6
      - 9
      - 17
    * - HIV
      - 41,127
      - ~25.5
      - ~54.9
      - 9
      - 1
    * - BACE
      - 1513
      - ~34.1
      - ~73.7
      - 9
      - 1
    * - BBBP
      - 2,050
      - ~23.9
      - ~51.6
      - 9
      - 1
    * - Tox21
      - 7,831
      - ~18.6
      - ~38.6
      - 9
      - 12
    * - ToxCast
      - 8,597
      - ~18.7
      - ~38.4
      - 9
      - 617
    * - SIDER
      - 1,427
      - ~33.6
      - ~70.7
      - 9
      - 27
    * - ClinTox
      - 1,484
      - ~26.1
      - ~55.5
      - 9
      - 2

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.
