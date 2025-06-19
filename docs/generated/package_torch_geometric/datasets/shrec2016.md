# shrec2016

Part of `torch_geometric.datasets`
Module: `torch_geometric.datasets.shrec2016`

## Functions (4)

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

### `extract_zip(path: str, folder: str, log: bool = True) -> None`

Extracts a zip archive to a specific folder.

Args:
    path (str): The path to the tar archive.
    folder (str): The folder.
    log (bool, optional): If :obj:`False`, will not print anything to the
        console. (default: :obj:`True`)

### `read_off(path: str) -> torch_geometric.data.data.Data`

Reads an OFF (Object File Format) file, returning both the position of
nodes and their connectivity in a :class:`torch_geometric.data.Data`
object.

Args:
    path (str): The path to the file.

### `read_txt_array(path: str, sep: Optional[str] = None, start: int = 0, end: Optional[int] = None, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor`

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

### `SHREC2016`

The SHREC 2016 partial matching dataset from the `"SHREC'16: Partial
Matching of Deformable Shapes"
<http://www.dais.unive.it/~shrec2016/shrec16-partial.pdf>`_ paper.
The reference shape can be referenced via :obj:`dataset.ref`.

.. note::

    Data objects hold mesh faces instead of edge indices.
    To convert the mesh to a graph, use the
    :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
    To convert the mesh to a point cloud, use the
    :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
    sample a fixed number of points on the mesh faces according to their
    face area.

Args:
    root (str): Root directory where the dataset should be saved.
    partiality (str): The partiality of the dataset (one of :obj:`"Holes"`,
        :obj:`"Cuts"`).
    category (str): The category of the dataset (one of
        :obj:`"Cat"`, :obj:`"Centaur"`, :obj:`"David"`, :obj:`"Dog"`,
        :obj:`"Horse"`, :obj:`"Michael"`, :obj:`"Victoria"`,
        :obj:`"Wolf"`).
    train (bool, optional): If :obj:`True`, loads the training dataset,
        otherwise the test dataset. (default: :obj:`True`)
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

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.
