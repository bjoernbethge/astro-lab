# sbm_dataset

Part of `torch_geometric.datasets`
Module: `torch_geometric.datasets.sbm_dataset`

## Functions (1)

### `stochastic_blockmodel_graph(block_sizes: Union[List[int], torch.Tensor], edge_probs: Union[List[List[float]], torch.Tensor], directed: bool = False) -> torch.Tensor`

Returns the :obj:`edge_index` of a stochastic blockmodel graph.

Args:
    block_sizes ([int] or LongTensor): The sizes of blocks.
    edge_probs ([[float]] or FloatTensor): The density of edges going
        from each block to each other block. Must be symmetric if the
        graph is undirected.
    directed (bool, optional): If set to :obj:`True`, will return a
        directed graph. (default: :obj:`False`)

Examples:
    >>> block_sizes = [2, 2, 4]
    >>> edge_probs = [[0.25, 0.05, 0.02],
    ...               [0.05, 0.35, 0.07],
    ...               [0.02, 0.07, 0.40]]
    >>> stochastic_blockmodel_graph(block_sizes, edge_probs,
    ...                             directed=False)
    tensor([[2, 4, 4, 5, 5, 6, 7, 7],
            [5, 6, 7, 2, 7, 4, 4, 5]])

    >>> stochastic_blockmodel_graph(block_sizes, edge_probs,
    ...                             directed=True)
    tensor([[0, 2, 3, 4, 4, 5, 5],
            [3, 4, 1, 5, 6, 6, 7]])

## Classes (6)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `Data`

A data object describing a homogeneous graph.
The data object can hold node-level, link-level and graph-level attributes.
In general, :class:`~torch_geometric.data.Data` tries to mimic the
behavior of a regular :python:`Python` dictionary.
In addition, it provides useful functionality for analyzing graph
structures, and provides basic PyTorch tensor functionalities.
See `here <https://pytorch-geometric.readthedocs.io/en/latest/get_started/
introduction.html#data-handling-of-graphs>`__ for the accompanying
tutorial.

.. code-block:: python

    from torch_geometric.data import Data

    data = Data(x=x, edge_index=edge_index, ...)

    # Add additional arguments to `data`:
    data.train_idx = torch.tensor([...], dtype=torch.long)
    data.test_mask = torch.tensor([...], dtype=torch.bool)

    # Analyzing the graph structure:
    data.num_nodes
    >>> 23

    data.is_directed()
    >>> False

    # PyTorch tensor functionality:
    data = data.pin_memory()
    data = data.to('cuda:0', non_blocking=True)

Args:
    x (torch.Tensor, optional): Node feature matrix with shape
        :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
    edge_index (LongTensor, optional): Graph connectivity in COO format
        with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
    edge_attr (torch.Tensor, optional): Edge feature matrix with shape
        :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
    y (torch.Tensor, optional): Graph-level or node-level ground-truth
        labels with arbitrary shape. (default: :obj:`None`)
    pos (torch.Tensor, optional): Node position matrix with shape
        :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
    time (torch.Tensor, optional): The timestamps for each event with shape
        :obj:`[num_edges]` or :obj:`[num_nodes]`. (default: :obj:`None`)
    **kwargs (optional): Additional attributes.

#### Methods

- **`stores_as(self, data: Self)`**

- **`to_dict(self) -> Dict[str, Any]`**
  Returns a dictionary of stored key/value pairs.

- **`to_namedtuple(self) -> <function NamedTuple at 0x000001FE17E66F20>`**
  Returns a :obj:`NamedTuple` of stored key/value pairs.

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

### `RandomPartitionGraphDataset`

The random partition graph dataset from the `"How to Find Your
Friendly Neighborhood: Graph Attention Design with Self-Supervision"
<https://openreview.net/forum?id=Wi5KUNlqWty>`_ paper.
This is a synthetic graph of communities controlled by the node homophily
and the average degree, and each community is considered as a class.
The node features are sampled from normal distributions where the centers
of clusters are vertices of a hypercube, as computed by the
:meth:`sklearn.datasets.make_classification` method.

Args:
    root (str): Root directory where the dataset should be saved.
    num_classes (int): The number of classes.
    num_nodes_per_class (int): The number of nodes per class.
    node_homophily_ratio (float): The degree of node homophily.
    average_degree (float): The average degree of the graph.
    num_graphs (int, optional): The number of graphs. (default: :obj:`1`)
    num_channels (int, optional): The number of node features. If given
        as :obj:`None`, node features are not generated.
        (default: :obj:`None`)
    is_undirected (bool, optional): Whether the graph to generate is
        undirected. (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes
        in an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    **kwargs (optional): The keyword arguments that are passed down
        to :meth:`sklearn.datasets.make_classification` method in
        drawing node features.

#### Methods

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `StochasticBlockModelDataset`

A synthetic graph dataset generated by the stochastic block model.
The node features of each block are sampled from normal distributions where
the centers of clusters are vertices of a hypercube, as computed by the
:meth:`sklearn.datasets.make_classification` method.

Args:
    root (str): Root directory where the dataset should be saved.
    block_sizes ([int] or LongTensor): The sizes of blocks.
    edge_probs ([[float]] or FloatTensor): The density of edges going from
        each block to each other block. Must be symmetric if the graph is
        undirected.
    num_graphs (int, optional): The number of graphs. (default: :obj:`1`)
    num_channels (int, optional): The number of node features. If given
        as :obj:`None`, node features are not generated.
        (default: :obj:`None`)
    is_undirected (bool, optional): Whether the graph to generate is
        undirected. (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes
        in an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed
        before being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)
    **kwargs (optional): The keyword arguments that are passed down to the
        :meth:`sklearn.datasets.make_classification` method for drawing
        node features.

#### Methods

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
