# shadow

Part of `torch_geometric.loader`
Module: `torch_geometric.loader.shadow`

## Classes (5)

### `Batch`

A data object describing a batch of graphs as one big (disconnected)
graph.
Inherits from :class:`torch_geometric.data.Data` or
:class:`torch_geometric.data.HeteroData`.
In addition, single graphs can be identified via the assignment vector
:obj:`batch`, which maps each node to its respective graph identifier.

:pyg:`PyG` allows modification to the underlying batching procedure by
overwriting the :meth:`~Data.__inc__` and :meth:`~Data.__cat_dim__`
functionalities.
The :meth:`~Data.__inc__` method defines the incremental count between two
consecutive graph attributes.
By default, :pyg:`PyG` increments attributes by the number of nodes
whenever their attribute names contain the substring :obj:`index`
(for historical reasons), which comes in handy for attributes such as
:obj:`edge_index` or :obj:`node_index`.
However, note that this may lead to unexpected behavior for attributes
whose names contain the substring :obj:`index` but should not be
incremented.
To make sure, it is best practice to always double-check the output of
batching.
Furthermore, :meth:`~Data.__cat_dim__` defines in which dimension graph
tensors of the same attribute should be concatenated together.

#### Methods

- **`get_example(self, idx: int) -> torch_geometric.data.data.BaseData`**
  Gets the :class:`~torch_geometric.data.Data` or

- **`index_select(self, idx: Union[slice, torch.Tensor, numpy.ndarray, collections.abc.Sequence]) -> List[torch_geometric.data.data.BaseData]`**
  Creates a subset of :class:`~torch_geometric.data.Data` or

- **`to_data_list(self) -> List[torch_geometric.data.data.BaseData]`**
  Reconstructs the list of :class:`~torch_geometric.data.Data` or

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

### `ShaDowKHopSampler`

The ShaDow :math:`k`-hop sampler from the `"Decoupling the Depth and
Scope of Graph Neural Networks" <https://arxiv.org/abs/2201.07858>`_ paper.
Given a graph in a :obj:`data` object, the sampler will create shallow,
localized subgraphs.
A deep GNN on this local graph then smooths the informative local signals.

.. note::

    For an example of using :class:`ShaDowKHopSampler`, see
    `examples/shadow.py <https://github.com/pyg-team/
    pytorch_geometric/blob/master/examples/shadow.py>`_.

Args:
    data (torch_geometric.data.Data): The graph data object.
    depth (int): The depth/number of hops of the localized subgraph.
    num_neighbors (int): The number of neighbors to sample for each node in
        each hop.
    node_idx (LongTensor or BoolTensor, optional): The nodes that should be
        considered for creating mini-batches.
        If set to :obj:`None`, all nodes will be
        considered.
    replace (bool, optional): If set to :obj:`True`, will sample neighbors
        with replacement. (default: :obj:`False`)
    **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size` or
        :obj:`num_workers`.

### `SparseTensor`

#### Methods

- **`size(self, dim: int) -> int`**

- **`nnz(self) -> int`**

- **`is_cuda(self) -> bool`**

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
