# node_property_split

Part of `torch_geometric.transforms`
Module: `torch_geometric.transforms.node_property_split`

## Functions (2)

### `functional_transform(name: str) -> Callable`

### `to_networkx(data: Union[ForwardRef('torch_geometric.data.Data'), ForwardRef('torch_geometric.data.HeteroData')], node_attrs: Optional[Iterable[str]] = None, edge_attrs: Optional[Iterable[str]] = None, graph_attrs: Optional[Iterable[str]] = None, to_undirected: Union[bool, str, NoneType] = False, to_multi: bool = False, remove_self_loops: bool = False) -> Any`

Converts a :class:`torch_geometric.data.Data` instance to a
:obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
a directed :obj:`networkx.DiGraph` otherwise.

Args:
    data (torch_geometric.data.Data or torch_geometric.data.HeteroData): A
        homogeneous or heterogeneous data object.
    node_attrs (iterable of str, optional): The node attributes to be
        copied. (default: :obj:`None`)
    edge_attrs (iterable of str, optional): The edge attributes to be
        copied. (default: :obj:`None`)
    graph_attrs (iterable of str, optional): The graph attributes to be
        copied. (default: :obj:`None`)
    to_undirected (bool or str, optional): If set to :obj:`True`, will
        return a :class:`networkx.Graph` instead of a
        :class:`networkx.DiGraph`.
        By default, will include all edges and make them undirected.
        If set to :obj:`"upper"`, the undirected graph will only correspond
        to the upper triangle of the input adjacency matrix.
        If set to :obj:`"lower"`, the undirected graph will only correspond
        to the lower triangle of the input adjacency matrix.
        Only applicable in case the :obj:`data` object holds a homogeneous
        graph. (default: :obj:`False`)
    to_multi (bool, optional): if set to :obj:`True`, will return a
        :class:`networkx.MultiGraph` or a :class:`networkx:MultiDiGraph`
        (depending on the :obj:`to_undirected` option), which will not drop
        duplicated edges that may exist in :obj:`data`.
        (default: :obj:`False`)
    remove_self_loops (bool, optional): If set to :obj:`True`, will not
        include self-loops in the resulting graph. (default: :obj:`False`)

Examples:
    >>> edge_index = torch.tensor([
    ...     [0, 1, 1, 2, 2, 3],
    ...     [1, 0, 2, 1, 3, 2],
    ... ])
    >>> data = Data(edge_index=edge_index, num_nodes=4)
    >>> to_networkx(data)
    <networkx.classes.digraph.DiGraph at 0x2713fdb40d0>

## Classes (5)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `BaseTransform`

An abstract base class for writing transforms.

Transforms are a general way to modify and customize
:class:`~torch_geometric.data.Data` or
:class:`~torch_geometric.data.HeteroData` objects, either by implicitly
passing them as an argument to a :class:`~torch_geometric.data.Dataset`, or
by applying them explicitly to individual
:class:`~torch_geometric.data.Data` or
:class:`~torch_geometric.data.HeteroData` objects:

.. code-block:: python

    import torch_geometric.transforms as T
    from torch_geometric.datasets import TUDataset

    transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])

    dataset = TUDataset(path, name='MUTAG', transform=transform)
    data = dataset[0]  # Implicitly transform data on every access.

    data = TUDataset(path, name='MUTAG')[0]
    data = transform(data)  # Explicitly transform data.

#### Methods

- **`forward(self, data: Any) -> Any`**

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

### `NodePropertySplit`

Creates a node-level split with distributional shift based on a given
node property, as proposed in the `"Evaluating Robustness and Uncertainty
of Graph Models Under Structural Distributional Shifts"
<https://arxiv.org/abs/2302.13875>`__ paper
(functional name: :obj:`node_property_split`).

It splits the nodes in a given graph into five non-intersecting parts
based on their structural properties.
This can be used for transductive node prediction tasks with distributional
shifts.
It considers the in-distribution (ID) and out-of-distribution (OOD) subsets
of nodes.
The ID subset includes training, validation and testing parts, while
the OOD subset includes validation and testing parts.
As a result, it creates five associated node mask vectors for each graph,
three which are for the ID nodes (:obj:`id_train_mask`,
:obj:`id_val_mask`, :obj:`id_test_mask`), and two which are for the OOD
nodes (:obj:`ood_val_mask`, :obj:`ood_test_mask`).

This class implements three particular strategies for inducing
distributional shifts in a graph — based on **popularity**, **locality**
or **density**.

Args:
    property_name (str): The name of the node property to be used
        (:obj:`"popularity"`, :obj:`"locality"`, :obj:`"density"`).
    ratios ([float]): A list of five ratio values for ID training,
        ID validation, ID test, OOD validation and OOD test parts.
        The values must sum to :obj:`1.0`.
    ascending (bool, optional): Whether to sort nodes in ascending order
        of the node property, so that nodes with greater values of the
        property are considered to be OOD (default: :obj:`True`)

.. code-block:: python

    from torch_geometric.transforms import NodePropertySplit
    from torch_geometric.datasets.graph_generator import ERGraph

    data = ERGraph(num_nodes=1000, edge_prob=0.4)()

    property_name = 'popularity'
    ratios = [0.3, 0.1, 0.1, 0.3, 0.2]
    transform = NodePropertySplit(property_name, ratios)

    data = transform(data)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
