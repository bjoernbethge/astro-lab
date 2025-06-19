# gdc

Part of `torch_geometric.transforms`
Module: `torch_geometric.transforms.gdc`

## Functions (8)

### `add_self_loops(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, fill_value: Union[float, torch.Tensor, str, NoneType] = None, num_nodes: Union[int, Tuple[int, int], NoneType] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]`

Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
:math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
In case the graph is weighted or has multi-dimensional edge features
(:obj:`edge_attr != None`), edge features of self-loops will be added
according to :obj:`fill_value`.

Args:
    edge_index (LongTensor): The edge indices.
    edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
        features. (default: :obj:`None`)
    fill_value (float or Tensor or str, optional): The way to generate
        edge features of self-loops (in case :obj:`edge_attr != None`).
        If given as :obj:`float` or :class:`torch.Tensor`, edge features of
        self-loops will be directly given by :obj:`fill_value`.
        If given as :obj:`str`, edge features of self-loops are computed by
        aggregating all features of edges that point to the specific node,
        according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
        :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)
    num_nodes (int or Tuple[int, int], optional): The number of nodes,
        *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
        If given as a tuple, then :obj:`edge_index` is interpreted as a
        bipartite graph with shape :obj:`(num_src_nodes, num_dst_nodes)`.
        (default: :obj:`None`)

:rtype: (:class:`LongTensor`, :class:`Tensor`)

Examples:
    >>> edge_index = torch.tensor([[0, 1, 0],
    ...                            [1, 0, 0]])
    >>> edge_weight = torch.tensor([0.5, 0.5, 0.5])
    >>> add_self_loops(edge_index)
    (tensor([[0, 1, 0, 0, 1],
            [1, 0, 0, 0, 1]]),
    None)

    >>> add_self_loops(edge_index, edge_weight)
    (tensor([[0, 1, 0, 0, 1],
            [1, 0, 0, 0, 1]]),
    tensor([0.5000, 0.5000, 0.5000, 1.0000, 1.0000]))

    >>> # edge features of self-loops are filled by constant `2.0`
    >>> add_self_loops(edge_index, edge_weight,
    ...                fill_value=2.)
    (tensor([[0, 1, 0, 0, 1],
            [1, 0, 0, 0, 1]]),
    tensor([0.5000, 0.5000, 0.5000, 2.0000, 2.0000]))

    >>> # Use 'add' operation to merge edge features for self-loops
    >>> add_self_loops(edge_index, edge_weight,
    ...                fill_value='add')
    (tensor([[0, 1, 0, 0, 1],
            [1, 0, 0, 0, 1]]),
    tensor([0.5000, 0.5000, 0.5000, 1.0000, 0.5000]))

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

### `functional_transform(name: str) -> Callable`

### `get_ppr(edge_index: torch.Tensor, alpha: float = 0.2, eps: float = 1e-05, target: Optional[torch.Tensor] = None, num_nodes: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]`

Calculates the personalized PageRank (PPR) vector for all or a subset
of nodes using a variant of the `Andersen algorithm
<https://mathweb.ucsd.edu/~fan/wp/localpartition.pdf>`_.

Args:
    edge_index (torch.Tensor): The indices of the graph.
    alpha (float, optional): The alpha value of the PageRank algorithm.
        (default: :obj:`0.2`)
    eps (float, optional): The threshold for stopping the PPR calculation
        (:obj:`edge_weight >= eps * out_degree`). (default: :obj:`1e-5`)
    target (torch.Tensor, optional): The target nodes to compute PPR for.
        If not given, calculates PPR vectors for all nodes.
        (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes. (default: :obj:`None`)

:rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`)

### `is_undirected(edge_index: torch.Tensor, edge_attr: Union[torch.Tensor, NoneType, List[torch.Tensor]] = None, num_nodes: Optional[int] = None) -> bool`

Returns :obj:`True` if the graph given by :attr:`edge_index` is
undirected.

Args:
    edge_index (LongTensor): The edge indices.
    edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
        dimensional edge features.
        If given as a list, will check for equivalence in all its entries.
        (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max(edge_index) + 1`. (default: :obj:`None`)

:rtype: bool

Examples:
    >>> edge_index = torch.tensor([[0, 1, 0],
    ...                         [1, 0, 0]])
    >>> weight = torch.tensor([0, 0, 1])
    >>> is_undirected(edge_index, weight)
    True

    >>> weight = torch.tensor([0, 1, 1])
    >>> is_undirected(edge_index, weight)
    False

### `scatter(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None, reduce: str = 'sum') -> torch.Tensor`

Reduces all values from the :obj:`src` tensor at the indices
specified in the :obj:`index` tensor along a given dimension
:obj:`dim`. See the `documentation
<https://pytorch-scatter.readthedocs.io/en/latest/functions/
scatter.html>`__ of the :obj:`torch_scatter` package for more
information.

Args:
    src (torch.Tensor): The source tensor.
    index (torch.Tensor): The index tensor.
    dim (int, optional): The dimension along which to index.
        (default: :obj:`0`)
    dim_size (int, optional): The size of the output tensor at
        dimension :obj:`dim`. If set to :obj:`None`, will create a
        minimal-sized output tensor according to
        :obj:`index.max() + 1`. (default: :obj:`None`)
    reduce (str, optional): The reduce operation (:obj:`"sum"`,
        :obj:`"mean"`, :obj:`"mul"`, :obj:`"min"` or :obj:`"max"`,
        :obj:`"any"`). (default: :obj:`"sum"`)

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

### `to_dense_adj(edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None, edge_attr: Optional[torch.Tensor] = None, max_num_nodes: Optional[int] = None, batch_size: Optional[int] = None) -> torch.Tensor`

Converts batched sparse adjacency matrices given by edge indices and
edge attributes to a single dense batched adjacency matrix.

Args:
    edge_index (LongTensor): The edge indices.
    batch (LongTensor, optional): Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        node to a specific example. (default: :obj:`None`)
    edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
        features.
        If :obj:`edge_index` contains duplicated edges, the dense adjacency
        matrix output holds the summed up entries of :obj:`edge_attr` for
        duplicated edges. (default: :obj:`None`)
    max_num_nodes (int, optional): The size of the output node dimension.
        (default: :obj:`None`)
    batch_size (int, optional): The batch size. (default: :obj:`None`)

:rtype: :class:`Tensor`

Examples:
    >>> edge_index = torch.tensor([[0, 0, 1, 2, 3],
    ...                            [0, 1, 0, 3, 0]])
    >>> batch = torch.tensor([0, 0, 1, 1])
    >>> to_dense_adj(edge_index, batch)
    tensor([[[1., 1.],
            [1., 0.]],
            [[0., 1.],
            [1., 0.]]])

    >>> to_dense_adj(edge_index, batch, max_num_nodes=4)
    tensor([[[1., 1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]],
            [[0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]]])

    >>> edge_attr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> to_dense_adj(edge_index, batch, edge_attr)
    tensor([[[1., 2.],
            [3., 0.]],
            [[0., 4.],
            [5., 0.]]])

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

### `GDC`

Processes the graph via Graph Diffusion Convolution (GDC) from the
`"Diffusion Improves Graph Learning" <https://arxiv.org/abs/1911.05485>`_
paper (functional name: :obj:`gdc`).

.. note::

    The paper offers additional advice on how to choose the
    hyperparameters.
    For an example of using GCN with GDC, see `examples/gcn.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    gcn.py>`_.

Args:
    self_loop_weight (float, optional): Weight of the added self-loop.
        Set to :obj:`None` to add no self-loops. (default: :obj:`1`)
    normalization_in (str, optional): Normalization of the transition
        matrix on the original (input) graph. Possible values:
        :obj:`"sym"`, :obj:`"col"`, and :obj:`"row"`.
        See :func:`GDC.transition_matrix` for details.
        (default: :obj:`"sym"`)
    normalization_out (str, optional): Normalization of the transition
        matrix on the transformed GDC (output) graph. Possible values:
        :obj:`"sym"`, :obj:`"col"`, :obj:`"row"`, and :obj:`None`.
        See :func:`GDC.transition_matrix` for details.
        (default: :obj:`"col"`)
    diffusion_kwargs (dict, optional): Dictionary containing the parameters
        for diffusion.
        `method` specifies the diffusion method (:obj:`"ppr"`,
        :obj:`"heat"` or :obj:`"coeff"`).
        Each diffusion method requires different additional parameters.
        See :func:`GDC.diffusion_matrix_exact` or
        :func:`GDC.diffusion_matrix_approx` for details.
        (default: :obj:`dict(method='ppr', alpha=0.15)`)
    sparsification_kwargs (dict, optional): Dictionary containing the
        parameters for sparsification.
        `method` specifies the sparsification method (:obj:`"threshold"` or
        :obj:`"topk"`).
        Each sparsification method requires different additional
        parameters.
        See :func:`GDC.sparsify_dense` for details.
        (default: :obj:`dict(method='threshold', avg_degree=64)`)
    exact (bool, optional): Whether to exactly calculate the diffusion
        matrix.
        Note that the exact variants are not scalable.
        They densify the adjacency matrix and calculate either its inverse
        or its matrix exponential.
        However, the approximate variants do not support edge weights and
        currently only personalized PageRank and sparsification by
        threshold are implemented as fast, approximate versions.
        (default: :obj:`True`)

:rtype: :class:`torch_geometric.data.Data`

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

- **`transition_matrix(self, edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int, normalization: str) -> Tuple[torch.Tensor, torch.Tensor]`**
  Calculate the approximate, sparse diffusion on a given sparse

- **`diffusion_matrix_exact(self, edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int, method: str, **kwargs: Any) -> torch.Tensor`**
  Calculate the (dense) diffusion on a given sparse graph.

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
