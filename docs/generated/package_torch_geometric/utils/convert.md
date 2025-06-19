# convert

Part of `torch_geometric.utils`
Module: `torch_geometric.utils.convert`

## Functions (14)

### `from_cugraph(g: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]`

Converts a :obj:`cugraph` graph object into :obj:`edge_index` and
optional :obj:`edge_weight` tensors.

Args:
    g (cugraph.Graph): A :obj:`cugraph` graph object.

### `from_dgl(g: Any) -> Union[ForwardRef('torch_geometric.data.Data'), ForwardRef('torch_geometric.data.HeteroData')]`

Converts a :obj:`dgl` graph object to a
:class:`torch_geometric.data.Data` or
:class:`torch_geometric.data.HeteroData` instance.

Args:
    g (dgl.DGLGraph): The :obj:`dgl` graph object.

Example:
    >>> g = dgl.graph(([0, 0, 1, 5], [1, 2, 2, 0]))
    >>> g.ndata['x'] = torch.randn(g.num_nodes(), 3)
    >>> g.edata['edge_attr'] = torch.randn(g.num_edges(), 2)
    >>> data = from_dgl(g)
    >>> data
    Data(x=[6, 3], edge_attr=[4, 2], edge_index=[2, 4])

    >>> g = dgl.heterograph({
    >>> g = dgl.heterograph({
    ...     ('author', 'writes', 'paper'): ([0, 1, 1, 2, 3, 3, 4],
    ...                                     [0, 0, 1, 1, 1, 2, 2])})
    >>> g.nodes['author'].data['x'] = torch.randn(5, 3)
    >>> g.nodes['paper'].data['x'] = torch.randn(5, 3)
    >>> data = from_dgl(g)
    >>> data
    HeteroData(
    author={ x=[5, 3] },
    paper={ x=[3, 3] },
    (author, writes, paper)={ edge_index=[2, 7] }
    )

### `from_dlpack(ext_tensor: Any) -> 'torch.Tensor'`

from_dlpack(ext_tensor) -> Tensor

Converts a tensor from an external library into a ``torch.Tensor``.

The returned PyTorch tensor will share the memory with the input tensor
(which may have come from another library). Note that in-place operations
will therefore also affect the data of the input tensor. This may lead to
unexpected issues (e.g., other libraries may have read-only flags or
immutable data structures), so the user should only do this if they know
for sure that this is fine.

Args:
    ext_tensor (object with ``__dlpack__`` attribute, or a DLPack capsule):
        The tensor or DLPack capsule to convert.

        If ``ext_tensor`` is a tensor (or ndarray) object, it must support
        the ``__dlpack__`` protocol (i.e., have a ``ext_tensor.__dlpack__``
        method). Otherwise ``ext_tensor`` may be a DLPack capsule, which is
        an opaque ``PyCapsule`` instance, typically produced by a
        ``to_dlpack`` function or method.

Examples::

    >>> import torch.utils.dlpack
    >>> t = torch.arange(4)

    # Convert a tensor directly (supported in PyTorch >= 1.10)
    >>> t2 = torch.from_dlpack(t)
    >>> t2[:2] = -1  # show that memory is shared
    >>> t2
    tensor([-1, -1,  2,  3])
    >>> t
    tensor([-1, -1,  2,  3])

    # The old-style DLPack usage, with an intermediate capsule object
    >>> capsule = torch.utils.dlpack.to_dlpack(t)
    >>> capsule
    <capsule object "dltensor" at ...>
    >>> t3 = torch.from_dlpack(capsule)
    >>> t3
    tensor([-1, -1,  2,  3])
    >>> t3[0] = -9  # now we're sharing memory between 3 tensors
    >>> t3
    tensor([-9, -1,  2,  3])
    >>> t2
    tensor([-9, -1,  2,  3])
    >>> t
    tensor([-9, -1,  2,  3])

### `from_networkit(g: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]`

Converts a :class:`networkit.Graph` to a
:obj:`(edge_index, edge_weight)` tuple.
If the :class:`networkit.Graph` is not weighted, the returned
:obj:`edge_weight` will be :obj:`None`.

Args:
    g (networkkit.graph.Graph): A :obj:`networkit` graph object.

### `from_networkx(G: Any, group_node_attrs: Union[List[str], Literal['all'], NoneType] = None, group_edge_attrs: Union[List[str], Literal['all'], NoneType] = None) -> 'torch_geometric.data.Data'`

Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
:class:`torch_geometric.data.Data` instance.

Args:
    G (networkx.Graph or networkx.DiGraph): A networkx graph.
    group_node_attrs (List[str] or "all", optional): The node attributes to
        be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
    group_edge_attrs (List[str] or "all", optional): The edge attributes to
        be concatenated and added to :obj:`data.edge_attr`.
        (default: :obj:`None`)

.. note::

    All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
    be numeric.

Examples:
    >>> edge_index = torch.tensor([
    ...     [0, 1, 1, 2, 2, 3],
    ...     [1, 0, 2, 1, 3, 2],
    ... ])
    >>> data = Data(edge_index=edge_index, num_nodes=4)
    >>> g = to_networkx(data)
    >>> # A `Data` object is returned
    >>> from_networkx(g)
    Data(edge_index=[2, 6], num_nodes=4)

### `from_scipy_sparse_matrix(A: Any) -> Tuple[torch.Tensor, torch.Tensor]`

Converts a scipy sparse matrix to edge indices and edge attributes.

Args:
    A (scipy.sparse): A sparse matrix.

Examples:
    >>> edge_index = torch.tensor([
    ...     [0, 1, 1, 2, 2, 3],
    ...     [1, 0, 2, 1, 3, 2],
    ... ])
    >>> adj = to_scipy_sparse_matrix(edge_index)
    >>> # `edge_index` and `edge_weight` are both returned
    >>> from_scipy_sparse_matrix(adj)
    (tensor([[0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2]]),
    tensor([1., 1., 1., 1., 1., 1.]))

### `from_trimesh(mesh: Any) -> 'torch_geometric.data.Data'`

Converts a :obj:`trimesh.Trimesh` to a
:class:`torch_geometric.data.Data` instance.

Args:
    mesh (trimesh.Trimesh): A :obj:`trimesh` mesh.

Example:
    >>> pos = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
    ...                    dtype=torch.float)
    >>> face = torch.tensor([[0, 1, 2], [1, 2, 3]]).t()

    >>> data = Data(pos=pos, face=face)
    >>> mesh = to_trimesh(data)
    >>> from_trimesh(mesh)
    Data(pos=[4, 3], face=[3, 2])

### `maybe_num_nodes(edge_index: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch_geometric.typing.SparseTensor], num_nodes: Optional[int] = None) -> int`

### `to_cugraph(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, relabel_nodes: bool = True, directed: bool = True) -> Any`

Converts a graph given by :obj:`edge_index` and optional
:obj:`edge_weight` into a :obj:`cugraph` graph object.

Args:
    edge_index (torch.Tensor): The edge indices of the graph.
    edge_weight (torch.Tensor, optional): The edge weights of the graph.
        (default: :obj:`None`)
    relabel_nodes (bool, optional): If set to :obj:`True`,
        :obj:`cugraph` will remove any isolated nodes, leading to a
        relabeling of nodes. (default: :obj:`True`)
    directed (bool, optional): If set to :obj:`False`, the graph will be
        undirected. (default: :obj:`True`)

### `to_dgl(data: Union[ForwardRef('torch_geometric.data.Data'), ForwardRef('torch_geometric.data.HeteroData')]) -> Any`

Converts a :class:`torch_geometric.data.Data` or
:class:`torch_geometric.data.HeteroData` instance to a :obj:`dgl` graph
object.

Args:
    data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
        The data object.

Example:
    >>> edge_index = torch.tensor([[0, 1, 1, 2, 3, 0], [1, 0, 2, 1, 4, 4]])
    >>> x = torch.randn(5, 3)
    >>> edge_attr = torch.randn(6, 2)
    >>> data = Data(x=x, edge_index=edge_index, edge_attr=y)
    >>> g = to_dgl(data)
    >>> g
    Graph(num_nodes=5, num_edges=6,
        ndata_schemes={'x': Scheme(shape=(3,))}
        edata_schemes={'edge_attr': Scheme(shape=(2, ))})

    >>> data = HeteroData()
    >>> data['paper'].x = torch.randn(5, 3)
    >>> data['author'].x = torch.ones(5, 3)
    >>> edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    >>> data['author', 'cites', 'paper'].edge_index = edge_index
    >>> g = to_dgl(data)
    >>> g
    Graph(num_nodes={'author': 5, 'paper': 5},
        num_edges={('author', 'cites', 'paper'): 5},
        metagraph=[('author', 'paper', 'cites')])

### `to_networkit(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, num_nodes: Optional[int] = None, directed: bool = True) -> Any`

Converts a :obj:`(edge_index, edge_weight)` tuple to a
:class:`networkit.Graph`.

Args:
    edge_index (torch.Tensor): The edge indices of the graph.
    edge_weight (torch.Tensor, optional): The edge weights of the graph.
        (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes in the graph.
        (default: :obj:`None`)
    directed (bool, optional): If set to :obj:`False`, the graph will be
        undirected. (default: :obj:`True`)

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

### `to_scipy_sparse_matrix(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, num_nodes: Optional[int] = None) -> Any`

Converts a graph given by edge indices and edge attributes to a scipy
sparse matrix.

Args:
    edge_index (LongTensor): The edge indices.
    edge_attr (Tensor, optional): Edge weights or multi-dimensional
        edge features. (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

Examples:
    >>> edge_index = torch.tensor([
    ...     [0, 1, 1, 2, 2, 3],
    ...     [1, 0, 2, 1, 3, 2],
    ... ])
    >>> to_scipy_sparse_matrix(edge_index)
    <4x4 sparse matrix of type '<class 'numpy.float32'>'
        with 6 stored elements in COOrdinate format>

### `to_trimesh(data: 'torch_geometric.data.Data') -> Any`

Converts a :class:`torch_geometric.data.Data` instance to a
:obj:`trimesh.Trimesh`.

Args:
    data (torch_geometric.data.Data): The data object.

Example:
    >>> pos = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
    ...                    dtype=torch.float)
    >>> face = torch.tensor([[0, 1, 2], [1, 2, 3]]).t()

    >>> data = Data(pos=pos, face=face)
    >>> to_trimesh(data)
    <trimesh.Trimesh(vertices.shape=(4, 3), faces.shape=(2, 3))>

## Classes (3)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.

### `defaultdict`

defaultdict(default_factory=None, /, [...]) --> dict with default factory

The default factory is called without arguments to produce
a new value when a key is not present, in __getitem__ only.
A defaultdict compares equal to a dict with the same items.
All remaining arguments are treated the same as if they were
passed to the dict constructor, including keyword arguments.

#### Methods

- **`copy(...)`**
  D.copy() -> a shallow copy of D.
