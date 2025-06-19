# cornell

Part of `torch_geometric.datasets`
Module: `torch_geometric.datasets.cornell`

## Functions (1)

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

## Classes (3)

### `CornellTemporalHyperGraphDataset`

A collection of temporal higher-order network datasets from the
`"Simplicial Closure and higher-order link prediction"
<https://arxiv.org/abs/1802.06916>`_ paper.
Each of the datasets is a timestamped sequence of simplices, where a
simplex is a set of :math:`k` nodes.

See the original `datasets page
<https://www.cs.cornell.edu/~arb/data/>`_ for more details about
individual datasets.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset.
    split (str, optional): If :obj:`"train"`, loads the training dataset.
        If :obj:`"val"`, loads the validation dataset.
        If :obj:`"test"`, loads the test dataset.
        (default: :obj:`"train"`)
    setting (str, optional): If :obj:`"transductive"`, loads the dataset
        for transductive training.
        If :obj:`"inductive"`, loads the dataset for inductive training.
        (default: :obj:`"transductive"`)
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

### `HyperGraphData`

A data object describing a hypergraph.

The data object can hold node-level, link-level and graph-level attributes.
This object differs from a standard :obj:`~torch_geometric.data.Data`
object by having hyperedges, i.e. edges that connect more
than two nodes. For example, in the hypergraph scenario
:math:`\mathcal{G} = (\mathcal{V}, \mathcal{E})` with
:math:`\mathcal{V} = \{ 0, 1, 2, 3, 4 \}` and
:math:`\mathcal{E} = \{ \{ 0, 1, 2 \}, \{ 1, 2, 3, 4 \} \}`, the
hyperedge index :obj:`edge_index` is represented as:

.. code-block:: python

    # hyper graph with two hyperedges
    # connecting 3 and 4 nodes, respectively
    edge_index = torch.tensor([
        [0, 1, 2, 1, 2, 3, 4],
        [0, 0, 0, 1, 1, 1, 1],
    ])

Args:
    x (torch.Tensor, optional): Node feature matrix with shape
        :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
    edge_index (LongTensor, optional): Hyperedge tensor
        with shape :obj:`[2, num_edges*num_nodes_per_edge]`.
        Where `edge_index[1]` denotes the hyperedge index and
        `edge_index[0]` denotes the node indicies that are connected
        by the hyperedge. (default: :obj:`None`)
        (default: :obj:`None`)
    edge_attr (torch.Tensor, optional): Edge feature matrix with shape
        :obj:`[num_edges, num_edge_features]`.
        (default: :obj:`None`)
    y (torch.Tensor, optional): Graph-level or node-level ground-truth
        labels with arbitrary shape. (default: :obj:`None`)
    pos (torch.Tensor, optional): Node position matrix with shape
        :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
    **kwargs (optional): Additional attributes.

#### Methods

- **`is_edge_attr(self, key: str) -> bool`**
  Returns :obj:`True` if the object at key :obj:`key` denotes an

- **`subgraph(self, subset: torch.Tensor) -> 'HyperGraphData'`**
  Returns the induced subgraph given by the node indices

- **`edge_subgraph(self, subset: torch.Tensor) -> Self`**
  Returns the induced subgraph given by the edge indices

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
