# cluster

Part of `torch_geometric.loader`
Module: `torch_geometric.loader.cluster`

## Functions (8)

### `dataclass(cls=None, /, *, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False, weakref_slot=False)`

Add dunder methods based on the fields defined in the class.

Examines PEP 526 __annotations__ to determine fields.

If init is true, an __init__() method is added to the class. If repr
is true, a __repr__() method is added. If order is true, rich
comparison dunder methods are added. If unsafe_hash is true, a
__hash__() method is added. If frozen is true, fields may not be
assigned to after instance creation. If match_args is true, the
__match_args__ tuple is added. If kw_only is true, then by default
all fields are keyword-only. If slots is true, a new class with a
__slots__ attribute is returned.

### `index2ptr(index: torch.Tensor, size: Optional[int] = None) -> torch.Tensor`

### `index_sort(inputs: torch.Tensor, max_value: Optional[int] = None, stable: bool = False) -> Tuple[torch.Tensor, torch.Tensor]`

Sorts the elements of the :obj:`inputs` tensor in ascending order.
It is expected that :obj:`inputs` is one-dimensional and that it only
contains positive integer values. If :obj:`max_value` is given, it can
be used by the underlying algorithm for better performance.

Args:
    inputs (torch.Tensor): A vector with positive integer values.
    max_value (int, optional): The maximum value stored inside
        :obj:`inputs`. This value can be an estimation, but needs to be
        greater than or equal to the real maximum.
        (default: :obj:`None`)
    stable (bool, optional): Makes the sorting routine stable, which
        guarantees that the order of equivalent elements is preserved.
        (default: :obj:`False`)

### `map_index(src: torch.Tensor, index: torch.Tensor, max_index: Union[int, torch.Tensor, NoneType] = None, inclusive: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]`

Maps indices in :obj:`src` to the positional value of their
corresponding occurrence in :obj:`index`.
Indices must be strictly positive.

Args:
    src (torch.Tensor): The source tensor to map.
    index (torch.Tensor): The index tensor that denotes the new mapping.
    max_index (int, optional): The maximum index value.
        (default :obj:`None`)
    inclusive (bool, optional): If set to :obj:`True`, it is assumed that
        every entry in :obj:`src` has a valid entry in :obj:`index`.
        Can speed-up computation. (default: :obj:`False`)

:rtype: (:class:`torch.Tensor`, :class:`torch.BoolTensor`)

Examples:
    >>> src = torch.tensor([2, 0, 1, 0, 3])
    >>> index = torch.tensor([3, 2, 0, 1])

    >>> map_index(src, index)
    (tensor([1, 2, 3, 2, 0]), tensor([True, True, True, True, True]))

    >>> src = torch.tensor([2, 0, 1, 0, 3])
    >>> index = torch.tensor([3, 2, 0])

    >>> map_index(src, index)
    (tensor([1, 2, 2, 0]), tensor([True, True, False, True, True]))

.. note::

    If inputs are on GPU and :obj:`cudf` is available, consider using RMM
    for significant speed boosts.
    Proceed with caution as RMM may conflict with other allocators or
    fragments.

    .. code-block:: python

        import rmm
        rmm.reinitialize(pool_allocator=True)
        torch.cuda.memory.change_current_allocator(rmm.rmm_torch_allocator)

### `narrow(src: Union[torch.Tensor, List[Any]], dim: int, start: int, length: int) -> Union[torch.Tensor, List[Any]]`

Narrows the input tensor or input list to the specified range.

Args:
    src (torch.Tensor or list): The input tensor or list.
    dim (int): The dimension along which to narrow.
    start (int): The starting dimension.
    length (int): The distance to the ending dimension.

### `ptr2index(ptr: torch.Tensor, output_size: Optional[int] = None) -> torch.Tensor`

### `select(src: Union[torch.Tensor, List[Any], torch_geometric.typing.TensorFrame], index_or_mask: torch.Tensor, dim: int) -> Union[torch.Tensor, List[Any]]`

Selects the input tensor or input list according to a given index or
mask vector.

Args:
    src (torch.Tensor or list): The input tensor or list.
    index_or_mask (torch.Tensor): The index or mask vector.
    dim (int): The dimension along which to select.

### `sort_edge_index(edge_index: torch.Tensor, edge_attr: Union[torch.Tensor, NoneType, List[torch.Tensor], str] = '???', num_nodes: Optional[int] = None, sort_by_row: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, List[torch.Tensor]]]`

Row-wise sorts :obj:`edge_index`.

Args:
    edge_index (torch.Tensor): The edge indices.
    edge_attr (torch.Tensor or List[torch.Tensor], optional): Edge weights
        or multi-dimensional edge features.
        If given as a list, will re-shuffle and remove duplicates for all
        its entries. (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    sort_by_row (bool, optional): If set to :obj:`False`, will sort
        :obj:`edge_index` column-wise/by destination node.
        (default: :obj:`True`)

:rtype: :class:`LongTensor` if :attr:`edge_attr` is not passed, else
    (:class:`LongTensor`, :obj:`Optional[Tensor]` or :obj:`List[Tensor]]`)

.. warning::

    From :pyg:`PyG >= 2.3.0` onwards, this function will always return a
    tuple whenever :obj:`edge_attr` is passed as an argument (even in case
    it is set to :obj:`None`).

Examples:
    >>> edge_index = torch.tensor([[2, 1, 1, 0],
                            [1, 2, 0, 1]])
    >>> edge_attr = torch.tensor([[1], [2], [3], [4]])
    >>> sort_edge_index(edge_index)
    tensor([[0, 1, 1, 2],
            [1, 0, 2, 1]])

    >>> sort_edge_index(edge_index, edge_attr)
    (tensor([[0, 1, 1, 2],
            [1, 0, 2, 1]]),
    tensor([[4],
            [3],
            [2],
            [1]]))

## Classes (6)

### `ClusterData`

Clusters/partitions a graph data object into multiple subgraphs, as
motivated by the `"Cluster-GCN: An Efficient Algorithm for Training Deep
and Large Graph Convolutional Networks"
<https://arxiv.org/abs/1905.07953>`_ paper.

.. note::
    The underlying METIS algorithm requires undirected graphs as input.

Args:
    data (torch_geometric.data.Data): The graph data object.
    num_parts (int): The number of partitions.
    recursive (bool, optional): If set to :obj:`True`, will use multilevel
        recursive bisection instead of multilevel k-way partitioning.
        (default: :obj:`False`)
    save_dir (str, optional): If set, will save the partitioned data to the
        :obj:`save_dir` directory for faster re-use. (default: :obj:`None`)
    filename (str, optional): Name of the stored partitioned file.
        (default: :obj:`None`)
    log (bool, optional): If set to :obj:`False`, will not log any
        progress. (default: :obj:`True`)
    keep_inter_cluster_edges (bool, optional): If set to :obj:`True`,
        will keep inter-cluster edge connections. (default: :obj:`False`)
    sparse_format (str, optional): The sparse format to use for computing
        partitions. (default: :obj:`"csr"`)

### `ClusterLoader`

The data loader scheme from the `"Cluster-GCN: An Efficient Algorithm
for Training Deep and Large Graph Convolutional Networks"
<https://arxiv.org/abs/1905.07953>`_ paper which merges partioned subgraphs
and their between-cluster links from a large-scale graph data object to
form a mini-batch.

.. note::

    Use :class:`~torch_geometric.loader.ClusterData` and
    :class:`~torch_geometric.loader.ClusterLoader` in conjunction to
    form mini-batches of clusters.
    For an example of using Cluster-GCN, see
    `examples/cluster_gcn_reddit.py <https://github.com/pyg-team/
    pytorch_geometric/blob/master/examples/cluster_gcn_reddit.py>`_ or
    `examples/cluster_gcn_ppi.py <https://github.com/pyg-team/
    pytorch_geometric/blob/master/examples/cluster_gcn_ppi.py>`_.

Args:
    cluster_data (torch_geometric.loader.ClusterData): The already
        partioned data object.
    **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
        :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.

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

### `Partition`

Partition(indptr: torch.Tensor, index: torch.Tensor, partptr: torch.Tensor, node_perm: torch.Tensor, edge_perm: torch.Tensor, sparse_format: Literal['csr', 'csc'])

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.

### `pyg_lib`

The base class of the class hierarchy.

When called, it accepts no arguments and returns a new featureless
instance that has no instance attributes and cannot be given any.
