# neighbor_sampler

Part of `torch_geometric.loader`
Module: `torch_geometric.loader.neighbor_sampler`

## Functions (1)

### `NamedTuple(typename, fields=None, /, **kwargs)`

Typed version of namedtuple.

Usage::

    class Employee(NamedTuple):
        name: str
        id: int

This is equivalent to::

    Employee = collections.namedtuple('Employee', ['name', 'id'])

The resulting class has an extra __annotations__ attribute, giving a
dict that maps field names to types.  (The field names are also in
the _fields attribute, which is part of the namedtuple API.)
An alternative equivalent functional syntax is also accepted::

    Employee = NamedTuple('Employee', [('name', str), ('id', int)])

## Classes (5)

### `Adj`

Adj(adj_t, e_id, size)

#### Methods

- **`to(self, *args, **kwargs)`**

### `EdgeIndex`

EdgeIndex(edge_index, e_id, size)

#### Methods

- **`to(self, *args, **kwargs)`**

### `NeighborSampler`

The neighbor sampler from the `"Inductive Representation Learning on
Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, which allows
for mini-batch training of GNNs on large-scale graphs where full-batch
training is not feasible.

Given a GNN with :math:`L` layers and a specific mini-batch of nodes
:obj:`node_idx` for which we want to compute embeddings, this module
iteratively samples neighbors and constructs bipartite graphs that simulate
the actual computation flow of GNNs.

More specifically, :obj:`sizes` denotes how much neighbors we want to
sample for each node in each layer.
This module then takes in these :obj:`sizes` and iteratively samples
:obj:`sizes[l]` for each node involved in layer :obj:`l`.
In the next layer, sampling is repeated for the union of nodes that were
already encountered.
The actual computation graphs are then returned in reverse-mode, meaning
that we pass messages from a larger set of nodes to a smaller one, until we
reach the nodes for which we originally wanted to compute embeddings.

Hence, an item returned by :class:`NeighborSampler` holds the current
:obj:`batch_size`, the IDs :obj:`n_id` of all nodes involved in the
computation, and a list of bipartite graph objects via the tuple
:obj:`(edge_index, e_id, size)`, where :obj:`edge_index` represents the
bipartite edges between source and target nodes, :obj:`e_id` denotes the
IDs of original edges in the full graph, and :obj:`size` holds the shape
of the bipartite graph.
For each bipartite graph, target nodes are also included at the beginning
of the list of source nodes so that one can easily apply skip-connections
or add self-loops.

.. warning::

    :class:`~torch_geometric.loader.NeighborSampler` is deprecated and will
    be removed in a future release.
    Use :class:`torch_geometric.loader.NeighborLoader` instead.

.. note::

    For an example of using :obj:`NeighborSampler`, see
    `examples/reddit.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    reddit.py>`_ or
    `examples/ogbn_products_sage.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    ogbn_products_sage.py>`_.

Args:
    edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
        :class:`torch_sparse.SparseTensor` that defines the underlying
        graph connectivity/message passing flow.
        :obj:`edge_index` holds the indices of a (sparse) symmetric
        adjacency matrix.
        If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its shape
        must be defined as :obj:`[2, num_edges]`, where messages from nodes
        :obj:`edge_index[0]` are sent to nodes in :obj:`edge_index[1]`
        (in case :obj:`flow="source_to_target"`).
        If :obj:`edge_index` is of type :class:`torch_sparse.SparseTensor`,
        its sparse indices :obj:`(row, col)` should relate to
        :obj:`row = edge_index[1]` and :obj:`col = edge_index[0]`.
        The major difference between both formats is that we need to input
        the *transposed* sparse adjacency matrix.
    sizes ([int]): The number of neighbors to sample for each node in each
        layer. If set to :obj:`sizes[l] = -1`, all neighbors are included
        in layer :obj:`l`.
    node_idx (LongTensor, optional): The nodes that should be considered
        for creating mini-batches. If set to :obj:`None`, all nodes will be
        considered.
    num_nodes (int, optional): The number of nodes in the graph.
        (default: :obj:`None`)
    return_e_id (bool, optional): If set to :obj:`False`, will not return
        original edge indices of sampled edges. This is only useful in case
        when operating on graphs without edge features to save memory.
        (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in
        a sampled mini-batch and returns a transformed version.
        (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
        :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.

#### Methods

- **`sample(self, batch)`**

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
