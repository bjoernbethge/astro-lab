# generalconv

Part of `torch_geometric.graphgym`
Module: `torch_geometric.graphgym.contrib.layer.generalconv`

## Functions (4)

### `add_remaining_self_loops(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, fill_value: Union[float, torch.Tensor, str, NoneType] = None, num_nodes: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]`

Adds remaining self-loop :math:`(i,i) \in \mathcal{E}` to every node
:math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
In case the graph is weighted or has multi-dimensional edge features
(:obj:`edge_attr != None`), edge features of non-existing self-loops will
be added according to :obj:`fill_value`.

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
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

:rtype: (:class:`LongTensor`, :class:`Tensor`)

Example:
    >>> edge_index = torch.tensor([[0, 1],
    ...                            [1, 0]])
    >>> edge_weight = torch.tensor([0.5, 0.5])
    >>> add_remaining_self_loops(edge_index, edge_weight)
    (tensor([[0, 1, 0, 1],
            [1, 0, 0, 1]]),
    tensor([0.5000, 0.5000, 1.0000, 1.0000]))

### `glorot(value: Any)`

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

### `zeros(value: Any)`

## Classes (4)

### `GeneralConvLayer`

A general GNN layer.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None)`**

- **`forward(self, x, edge_index, edge_weight=None, edge_feature=None)`**
  Runs the forward pass of the module.

### `GeneralEdgeConvLayer`

General GNN layer, with edge features.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None)`**

- **`forward(self, x, edge_index, edge_weight=None, edge_feature=None)`**
  Runs the forward pass of the module.

### `MessagePassing`

Base class for creating message passing layers.

Message passing layers follow the form

.. math::
    \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
    \bigoplus_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
    \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) \right),

where :math:`\bigoplus` denotes a differentiable, permutation invariant
function, *e.g.*, sum, mean, min, max or mul, and
:math:`\gamma_{\mathbf{\Theta}}` and :math:`\phi_{\mathbf{\Theta}}` denote
differentiable functions such as MLPs.
See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
create_gnn.html>`__ for the accompanying tutorial.

Args:
    aggr (str or [str] or Aggregation, optional): The aggregation scheme
        to use, *e.g.*, :obj:`"sum"` :obj:`"mean"`, :obj:`"min"`,
        :obj:`"max"` or :obj:`"mul"`.
        In addition, can be any
        :class:`~torch_geometric.nn.aggr.Aggregation` module (or any string
        that automatically resolves to it).
        If given as a list, will make use of multiple aggregations in which
        different outputs will get concatenated in the last dimension.
        If set to :obj:`None`, the :class:`MessagePassing` instantiation is
        expected to implement its own aggregation logic via
        :meth:`aggregate`. (default: :obj:`"add"`)
    aggr_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective aggregation function in case it gets automatically
        resolved. (default: :obj:`None`)
    flow (str, optional): The flow direction of message passing
        (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
        (default: :obj:`"source_to_target"`)
    node_dim (int, optional): The axis along which to propagate.
        (default: :obj:`-2`)
    decomposed_layers (int, optional): The number of feature decomposition
        layers, as introduced in the `"Optimizing Memory Efficiency of
        Graph Neural Networks on Edge Computing Platforms"
        <https://arxiv.org/abs/2104.03058>`_ paper.
        Feature decomposition reduces the peak memory usage by slicing
        the feature dimensions into separated feature decomposition layers
        during GNN aggregation.
        This method can accelerate GNN execution on CPU-based platforms
        (*e.g.*, 2-3x speedup on the
        :class:`~torch_geometric.datasets.Reddit` dataset) for common GNN
        models such as :class:`~torch_geometric.nn.models.GCN`,
        :class:`~torch_geometric.nn.models.GraphSAGE`,
        :class:`~torch_geometric.nn.models.GIN`, etc.
        However, this method is not applicable to all GNN operators
        available, in particular for operators in which message computation
        can not easily be decomposed, *e.g.* in attention-based GNNs.
        The selection of the optimal value of :obj:`decomposed_layers`
        depends both on the specific graph dataset and available hardware
        resources.
        A value of :obj:`2` is suitable in most cases.
        Although the peak memory usage is directly associated with the
        granularity of feature decomposition, the same is not necessarily
        true for execution speedups. (default: :obj:`1`)

#### Methods

- **`reset_parameters(self) -> None`**
  Resets all learnable parameters of the module.

- **`forward(self, *args: Any, **kwargs: Any) -> Any`**
  Runs the forward pass of the module.

- **`propagate(self, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], size: Optional[Tuple[int, int]] = None, **kwargs: Any) -> torch.Tensor`**
  The initial call to start propagating messages.

### `Parameter`

A kind of Tensor that is to be considered a module parameter.

Parameters are :class:`~torch.Tensor` subclasses, that have a
very special property when used with :class:`Module` s - when they're
assigned as Module attributes they are automatically added to the list of
its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
Assigning a Tensor doesn't have such effect. This is because one might
want to cache some temporary state, like last hidden state of the RNN, in
the model. If there was no such class as :class:`Parameter`, these
temporaries would get registered too.

Args:
    data (Tensor): parameter tensor.
    requires_grad (bool, optional): if the parameter requires gradient. Note that
        the torch.no_grad() context does NOT affect the default behavior of
        Parameter creation--the Parameter will still have `requires_grad=True` in
        :class:`~no_grad` mode. See :ref:`locally-disable-grad-doc` for more
        details. Default: `True`
