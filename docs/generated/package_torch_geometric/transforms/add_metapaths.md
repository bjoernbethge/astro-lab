# add_metapaths

Part of `torch_geometric.transforms`
Module: `torch_geometric.transforms.add_metapaths`

## Functions (5)

### `cast(typ, val)`

Cast a value to a type.

This returns the value unchanged.  To the type checker this
signals that the return value has the designated type, but at
runtime we intentionally don't check anything (we want this
to be as fast as possible).

### `coalesce(edge_index: torch.Tensor, edge_attr: Union[torch.Tensor, NoneType, List[torch.Tensor], str] = '???', num_nodes: Optional[int] = None, reduce: str = 'sum', is_sorted: bool = False, sort_by_row: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, List[torch.Tensor]]]`

Row-wise sorts :obj:`edge_index` and removes its duplicated entries.
Duplicate entries in :obj:`edge_attr` are merged by scattering them
together according to the given :obj:`reduce` option.

Args:
    edge_index (torch.Tensor): The edge indices.
    edge_attr (torch.Tensor or List[torch.Tensor], optional): Edge weights
        or multi-dimensional edge features.
        If given as a list, will re-shuffle and remove duplicates for all
        its entries. (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    reduce (str, optional): The reduce operation to use for merging edge
        features (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
        :obj:`"mul"`, :obj:`"any"`). (default: :obj:`"sum"`)
    is_sorted (bool, optional): If set to :obj:`True`, will expect
        :obj:`edge_index` to be already sorted row-wise.
    sort_by_row (bool, optional): If set to :obj:`False`, will sort
        :obj:`edge_index` column-wise.

:rtype: :class:`LongTensor` if :attr:`edge_attr` is not passed, else
    (:class:`LongTensor`, :obj:`Optional[Tensor]` or :obj:`List[Tensor]]`)

.. warning::

    From :pyg:`PyG >= 2.3.0` onwards, this function will always return a
    tuple whenever :obj:`edge_attr` is passed as an argument (even in case
    it is set to :obj:`None`).

Example:
    >>> edge_index = torch.tensor([[1, 1, 2, 3],
    ...                            [3, 3, 1, 2]])
    >>> edge_attr = torch.tensor([1., 1., 1., 1.])
    >>> coalesce(edge_index)
    tensor([[1, 2, 3],
            [3, 1, 2]])

    >>> # Sort `edge_index` column-wise
    >>> coalesce(edge_index, sort_by_row=False)
    tensor([[2, 3, 1],
            [1, 2, 3]])

    >>> coalesce(edge_index, edge_attr)
    (tensor([[1, 2, 3],
            [3, 1, 2]]),
    tensor([2., 1., 1.]))

    >>> # Use 'mean' operation to merge edge features
    >>> coalesce(edge_index, edge_attr, reduce='mean')
    (tensor([[1, 2, 3],
            [3, 1, 2]]),
    tensor([1., 1., 1.]))

### `degree(index: torch.Tensor, num_nodes: Optional[int] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor`

Computes the (unweighted) degree of a given one-dimensional index
tensor.

Args:
    index (LongTensor): Index tensor.
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    dtype (:obj:`torch.dtype`, optional): The desired data type of the
        returned tensor.

:rtype: :class:`Tensor`

Example:
    >>> row = torch.tensor([0, 1, 0, 2, 0])
    >>> degree(row, dtype=torch.long)
    tensor([3, 1, 1])

### `functional_transform(name: str) -> Callable`

### `postprocess(data: torch_geometric.data.hetero_data.HeteroData, edge_types: List[Tuple[str, str, str]], drop_orig_edge_types: bool, keep_same_node_type: bool, drop_unconnected_node_types: bool) -> None`

## Classes (6)

### `AddMetaPaths`

Adds additional edge types to a
:class:`~torch_geometric.data.HeteroData` object between the source node
type and the destination node type of a given :obj:`metapath`, as described
in the `"Heterogenous Graph Attention Networks"
<https://arxiv.org/abs/1903.07293>`_ paper
(functional name: :obj:`add_metapaths`).

Meta-path based neighbors can exploit different aspects of structure
information in heterogeneous graphs.
Formally, a metapath is a path of the form

.. math::

    \mathcal{V}_1 \xrightarrow{R_1} \mathcal{V}_2 \xrightarrow{R_2} \ldots
    \xrightarrow{R_{\ell-1}} \mathcal{V}_{\ell}

in which :math:`\mathcal{V}_i` represents node types, and :math:`R_j`
represents the edge type connecting two node types.
The added edge type is given by the sequential multiplication  of
adjacency matrices along the metapath, and is added to the
:class:`~torch_geometric.data.HeteroData` object as edge type
:obj:`(src_node_type, "metapath_*", dst_node_type)`, where
:obj:`src_node_type` and :obj:`dst_node_type` denote :math:`\mathcal{V}_1`
and :math:`\mathcal{V}_{\ell}`, respectively.

In addition, a :obj:`metapath_dict` object is added to the
:class:`~torch_geometric.data.HeteroData` object which maps the
metapath-based edge type to its original metapath.

.. code-block:: python

    from torch_geometric.datasets import DBLP
    from torch_geometric.data import HeteroData
    from torch_geometric.transforms import AddMetaPaths

    data = DBLP(root)[0]
    # 4 node types: "paper", "author", "conference", and "term"
    # 6 edge types: ("paper","author"), ("author", "paper"),
    #               ("paper, "term"), ("paper", "conference"),
    #               ("term, "paper"), ("conference", "paper")

    # Add two metapaths:
    # 1. From "paper" to "paper" through "conference"
    # 2. From "author" to "conference" through "paper"
    metapaths = [[("paper", "conference"), ("conference", "paper")],
                 [("author", "paper"), ("paper", "conference")]]
    data = AddMetaPaths(metapaths)(data)

    print(data.edge_types)
    >>> [("author", "to", "paper"), ("paper", "to", "author"),
         ("paper", "to", "term"), ("paper", "to", "conference"),
         ("term", "to", "paper"), ("conference", "to", "paper"),
         ("paper", "metapath_0", "paper"),
         ("author", "metapath_1", "conference")]

    print(data.metapath_dict)
    >>> {("paper", "metapath_0", "paper"): [("paper", "conference"),
                                            ("conference", "paper")],
         ("author", "metapath_1", "conference"): [("author", "paper"),
                                                  ("paper", "conference")]}

Args:
    metapaths (List[List[Tuple[str, str, str]]]): The metapaths described
        by a list of lists of
        :obj:`(src_node_type, rel_type, dst_node_type)` tuples.
    drop_orig_edge_types (bool, optional): If set to :obj:`True`, existing
        edge types will be dropped. (default: :obj:`False`)
    keep_same_node_type (bool, optional): If set to :obj:`True`, existing
        edge types between the same node type are not dropped even in case
        :obj:`drop_orig_edge_types` is set to :obj:`True`.
        (default: :obj:`False`)
    drop_unconnected_node_types (bool, optional): If set to :obj:`True`,
        will drop node types not connected by any edge type.
        (default: :obj:`False`)
    max_sample (int, optional): If set, will sample at maximum
        :obj:`max_sample` neighbors within metapaths. Useful in order to
        tackle very dense metapath edges. (default: :obj:`None`)
    weighted (bool, optional): If set to :obj:`True`, computes weights for
        each metapath edge and stores them in :obj:`edge_weight`. The
        weight of each metapath edge is computed as the number of metapaths
        from the start to the end of the metapath edge.
        (default :obj:`False`)

#### Methods

- **`forward(self, data: torch_geometric.data.hetero_data.HeteroData) -> torch_geometric.data.hetero_data.HeteroData`**

### `AddRandomMetaPaths`

Adds additional edge types similar to :class:`AddMetaPaths`.
The key difference is that the added edge type is given by
multiple random walks along the metapath.
One might want to increase the number of random walks
via :obj:`walks_per_node` to achieve competitive performance with
:class:`AddMetaPaths`.

Args:
    metapaths (List[List[Tuple[str, str, str]]]): The metapaths described
        by a list of lists of
        :obj:`(src_node_type, rel_type, dst_node_type)` tuples.
    drop_orig_edge_types (bool, optional): If set to :obj:`True`, existing
        edge types will be dropped. (default: :obj:`False`)
    keep_same_node_type (bool, optional): If set to :obj:`True`, existing
        edge types between the same node type are not dropped even in case
        :obj:`drop_orig_edge_types` is set to :obj:`True`.
        (default: :obj:`False`)
    drop_unconnected_node_types (bool, optional): If set to :obj:`True`,
        will drop node types not connected by any edge type.
        (default: :obj:`False`)
    walks_per_node (int, List[int], optional): The number of random walks
        for each starting node in a metapath. (default: :obj:`1`)
    sample_ratio (float, optional): The ratio of source nodes to start
        random walks from. (default: :obj:`1.0`)

#### Methods

- **`forward(self, data: torch_geometric.data.hetero_data.HeteroData) -> torch_geometric.data.hetero_data.HeteroData`**

- **`sample(edge_index: torch_geometric.edge_index.EdgeIndex, subset: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`**
  Sample neighbors from :obj:`edge_index` for each node in

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

### `EdgeIndex`

A COO :obj:`edge_index` tensor with additional (meta)data attached.

:class:`EdgeIndex` is a :pytorch:`null` :class:`torch.Tensor`, that holds
an :obj:`edge_index` representation of shape :obj:`[2, num_edges]`.
Edges are given as pairwise source and destination node indices in sparse
COO format.

While :class:`EdgeIndex` sub-classes a general :pytorch:`null`
:class:`torch.Tensor`, it can hold additional (meta)data, *i.e.*:

* :obj:`sparse_size`: The underlying sparse matrix size
* :obj:`sort_order`: The sort order (if present), either by row or column.
* :obj:`is_undirected`: Whether edges are bidirectional.

Additionally, :class:`EdgeIndex` caches data for fast CSR or CSC conversion
in case its representation is sorted, such as its :obj:`rowptr` or
:obj:`colptr`, or the permutation vector for going from CSR to CSC or vice
versa.
Caches are filled based on demand (*e.g.*, when calling
:meth:`EdgeIndex.sort_by`), or when explicitly requested via
:meth:`EdgeIndex.fill_cache_`, and are maintained and adjusted over its
lifespan (*e.g.*, when calling :meth:`EdgeIndex.flip`).

This representation ensures for optimal computation in GNN message passing
schemes, while preserving the ease-of-use of regular COO-based :pyg:`PyG`
workflows.

.. code-block:: python

    from torch_geometric import EdgeIndex

    edge_index = EdgeIndex(
        [[0, 1, 1, 2],
         [1, 0, 2, 1]]
        sparse_size=(3, 3),
        sort_order='row',
        is_undirected=True,
        device='cpu',
    )
    >>> EdgeIndex([[0, 1, 1, 2],
    ...            [1, 0, 2, 1]])
    assert edge_index.is_sorted_by_row
    assert edge_index.is_undirected

    # Flipping order:
    edge_index = edge_index.flip(0)
    >>> EdgeIndex([[1, 0, 2, 1],
    ...            [0, 1, 1, 2]])
    assert edge_index.is_sorted_by_col
    assert edge_index.is_undirected

    # Filtering:
    mask = torch.tensor([True, True, True, False])
    edge_index = edge_index[:, mask]
    >>> EdgeIndex([[1, 0, 2],
    ...            [0, 1, 1]])
    assert edge_index.is_sorted_by_col
    assert not edge_index.is_undirected

    # Sparse-Dense Matrix Multiplication:
    out = edge_index.flip(0) @ torch.randn(3, 16)
    assert out.size() == (3, 16)

#### Methods

- **`validate(self) -> 'EdgeIndex'`**
  Validates the :class:`EdgeIndex` representation.

- **`sparse_size(self, dim: Optional[int] = None) -> Union[Tuple[Optional[int], Optional[int]], int, NoneType]`**
  The size of the underlying sparse matrix.

- **`get_sparse_size(self, dim: Optional[int] = None) -> Union[torch.Size, int]`**
  The size of the underlying sparse matrix.

### `HeteroData`

A data object describing a heterogeneous graph, holding multiple node
and/or edge types in disjunct storage objects.
Storage objects can hold either node-level, link-level or graph-level
attributes.
In general, :class:`~torch_geometric.data.HeteroData` tries to mimic the
behavior of a regular **nested** :python:`Python` dictionary.
In addition, it provides useful functionality for analyzing graph
structures, and provides basic PyTorch tensor functionalities.

.. code-block::

    from torch_geometric.data import HeteroData

    data = HeteroData()

    # Create two node types "paper" and "author" holding a feature matrix:
    data['paper'].x = torch.randn(num_papers, num_paper_features)
    data['author'].x = torch.randn(num_authors, num_authors_features)

    # Create an edge type "(author, writes, paper)" and building the
    # graph connectivity:
    data['author', 'writes', 'paper'].edge_index = ...  # [2, num_edges]

    data['paper'].num_nodes
    >>> 23

    data['author', 'writes', 'paper'].num_edges
    >>> 52

    # PyTorch tensor functionality:
    data = data.pin_memory()
    data = data.to('cuda:0', non_blocking=True)

Note that there exists multiple ways to create a heterogeneous graph data,
*e.g.*:

* To initialize a node of type :obj:`"paper"` holding a node feature
  matrix :obj:`x_paper` named :obj:`x`:

  .. code-block:: python

    from torch_geometric.data import HeteroData

    # (1) Assign attributes after initialization,
    data = HeteroData()
    data['paper'].x = x_paper

    # or (2) pass them as keyword arguments during initialization,
    data = HeteroData(paper={ 'x': x_paper })

    # or (3) pass them as dictionaries during initialization,
    data = HeteroData({'paper': { 'x': x_paper }})

* To initialize an edge from source node type :obj:`"author"` to
  destination node type :obj:`"paper"` with relation type :obj:`"writes"`
  holding a graph connectivity matrix :obj:`edge_index_author_paper` named
  :obj:`edge_index`:

  .. code-block:: python

    # (1) Assign attributes after initialization,
    data = HeteroData()
    data['author', 'writes', 'paper'].edge_index = edge_index_author_paper

    # or (2) pass them as keyword arguments during initialization,
    data = HeteroData(author__writes__paper={
        'edge_index': edge_index_author_paper
    })

    # or (3) pass them as dictionaries during initialization,
    data = HeteroData({
        ('author', 'writes', 'paper'):
        { 'edge_index': edge_index_author_paper }
    })

#### Methods

- **`stores_as(self, data: Self)`**

- **`node_items(self) -> List[Tuple[str, torch_geometric.data.storage.NodeStorage]]`**
  Returns a list of node type and node storage pairs.

- **`edge_items(self) -> List[Tuple[Tuple[str, str, str], torch_geometric.data.storage.EdgeStorage]]`**
  Returns a list of edge type and edge storage pairs.

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
