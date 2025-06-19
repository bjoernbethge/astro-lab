# to_hetero_with_bases_transformer

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.to_hetero_with_bases_transformer`

## Functions (10)

### `get_edge_offset_dict(input_dict: Dict[Tuple[str, str, str], Union[torch.Tensor, torch_geometric.typing.SparseTensor]], type2id: Dict[Tuple[str, str, str], int]) -> Dict[Tuple[str, str, str], int]`

### `get_edge_type(input_dict: Dict[Tuple[str, str, str], Union[torch.Tensor, torch_geometric.typing.SparseTensor]], type2id: Dict[Tuple[str, str, str], int]) -> torch.Tensor`

### `get_node_offset_dict(input_dict: Dict[str, Union[torch.Tensor, torch_geometric.typing.SparseTensor]], type2id: Dict[str, int]) -> Dict[str, int]`

### `get_unused_node_types(node_types: List[str], edge_types: List[Tuple[str, str, str]]) -> Set[str]`

### `group_edge_placeholder(input_dict: Dict[Tuple[str, str, str], Union[torch.Tensor, torch_geometric.typing.SparseTensor]], type2id: Dict[Tuple[str, str, str], int], offset_dict: Dict[str, int] = None) -> Union[torch.Tensor, torch_geometric.typing.SparseTensor]`

### `group_node_placeholder(input_dict: Dict[str, torch.Tensor], type2id: Dict[str, int]) -> torch.Tensor`

### `hook(module, inputs, output)`

### `key2str(key: Union[str, Tuple[str, str, str]]) -> str`

### `split_output(output: torch.Tensor, offset_dict: Union[Dict[str, int], Dict[Tuple[str, str, str], int]]) -> Union[Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor]]`

### `to_hetero_with_bases(module: torch.nn.modules.module.Module, metadata: Tuple[List[str], List[Tuple[str, str, str]]], num_bases: int, in_channels: Optional[Dict[str, int]] = None, input_map: Optional[Dict[str, str]] = None, debug: bool = False) -> torch.fx.graph_module.GraphModule`

Converts a homogeneous GNN model into its heterogeneous equivalent
via the basis-decomposition technique introduced in the
`"Modeling Relational Data with Graph Convolutional Networks"
<https://arxiv.org/abs/1703.06103>`_ paper.

For this, the heterogeneous graph is mapped to a typed homogeneous graph,
in which its feature representations are aligned and grouped to a single
representation.
All GNN layers inside the model will then perform message passing via
basis-decomposition regularization.
This transformation is especially useful in highly multi-relational data,
such that the number of parameters no longer depend on the number of
relations of the input graph:

.. code-block:: python

    import torch
    from torch_geometric.nn import SAGEConv, to_hetero_with_bases

    class GNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv((16, 16), 32)
            self.conv2 = SAGEConv((32, 32), 32)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
            return x

    model = GNN()

    node_types = ['paper', 'author']
    edge_types = [
        ('paper', 'cites', 'paper'),
        ('paper', 'written_by', 'author'),
        ('author', 'writes', 'paper'),
    ]
    metadata = (node_types, edge_types)

    model = to_hetero_with_bases(model, metadata, num_bases=3,
                                 in_channels={'x': 16})
    model(x_dict, edge_index_dict)

where :obj:`x_dict` and :obj:`edge_index_dict` denote dictionaries that
hold node features and edge connectivity information for each node type and
edge type, respectively.
In case :obj:`in_channels` is given for a specific input argument, its
heterogeneous feature information is first aligned to the given
dimensionality.

The below illustration shows the original computation graph of the
homogeneous model on the left, and the newly obtained computation graph of
the regularized heterogeneous model on the right:

.. figure:: ../_figures/to_hetero_with_bases.svg
  :align: center
  :width: 90%

  Transforming a model via :func:`to_hetero_with_bases`.

Here, each :class:`~torch_geometric.nn.conv.MessagePassing` instance
:math:`f_{\theta}^{(\ell)}` is duplicated :obj:`num_bases` times and
stored in a set :math:`\{ f_{\theta}^{(\ell, b)} : b \in \{ 1, \ldots, B \}
\}` (one instance for each basis in
:obj:`num_bases`), and message passing in layer :math:`\ell` is performed
via

.. math::

    \mathbf{h}^{(\ell)}_v = \sum_{r \in \mathcal{R}} \sum_{b=1}^B
    f_{\theta}^{(\ell, b)} ( \mathbf{h}^{(\ell - 1)}_v, \{
    a^{(\ell)}_{r, b} \cdot \mathbf{h}^{(\ell - 1)}_w :
    w \in \mathcal{N}^{(r)}(v) \}),

where :math:`\mathcal{N}^{(r)}(v)` denotes the neighborhood of :math:`v \in
\mathcal{V}` under relation :math:`r \in \mathcal{R}`.
Notably, only the trainable basis coefficients :math:`a^{(\ell)}_{r, b}`
depend on the relations in :math:`\mathcal{R}`.

Args:
    module (torch.nn.Module): The homogeneous model to transform.
    metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
        of the heterogeneous graph, *i.e.* its node and edge types given
        by a list of strings and a list of string triplets, respectively.
        See :meth:`torch_geometric.data.HeteroData.metadata` for more
        information.
    num_bases (int): The number of bases to use.
    in_channels (Dict[str, int], optional): A dictionary holding
        information about the desired input feature dimensionality of
        input arguments of :obj:`module.forward`.
        In case :obj:`in_channels` is given for a specific input argument,
        its heterogeneous feature information is first aligned to the given
        dimensionality.
        This allows handling of node and edge features with varying feature
        dimensionality across different types. (default: :obj:`None`)
    input_map (Dict[str, str], optional): A dictionary holding information
        about the type of input arguments of :obj:`module.forward`.
        For example, in case :obj:`arg` is a node-level argument, then
        :obj:`input_map['arg'] = 'node'`, and
        :obj:`input_map['arg'] = 'edge'` otherwise.
        In case :obj:`input_map` is not further specified, will try to
        automatically determine the correct type of input arguments.
        (default: :obj:`None`)
    debug (bool, optional): If set to :obj:`True`, will perform
        transformation in debug mode. (default: :obj:`False`)

## Classes (15)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `Graph`

``Graph`` is the main data structure used in the FX Intermediate Representation.
It consists of a series of ``Node`` s, each representing callsites (or other
syntactic constructs). The list of ``Node`` s, taken together, constitute a
valid Python function.

For example, the following code

.. code-block:: python

    import torch
    import torch.fx


    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.rand(3, 4))
            self.linear = torch.nn.Linear(4, 5)

        def forward(self, x):
            return torch.topk(
                torch.sum(self.linear(x + self.linear.weight).relu(), dim=-1), 3
            )


    m = MyModule()
    gm = torch.fx.symbolic_trace(m)

Will produce the following Graph::

    print(gm.graph)

.. code-block:: text

    graph(x):
        %linear_weight : [num_users=1] = self.linear.weight
        %add_1 : [num_users=1] = call_function[target=operator.add](args = (%x, %linear_weight), kwargs = {})
        %linear_1 : [num_users=1] = call_module[target=linear](args = (%add_1,), kwargs = {})
        %relu_1 : [num_users=1] = call_method[target=relu](args = (%linear_1,), kwargs = {})
        %sum_1 : [num_users=1] = call_function[target=torch.sum](args = (%relu_1,), kwargs = {dim: -1})
        %topk_1 : [num_users=1] = call_function[target=torch.topk](args = (%sum_1, 3), kwargs = {})
        return topk_1

For the semantics of operations represented in the ``Graph``, please see :class:`Node`.

.. note::
    Backwards-compatibility for this API is guaranteed.

#### Methods

- **`output_node(self) -> torch.fx.node.Node`**
  .. warning::

- **`find_nodes(self, *, op: str, target: Optional[ForwardRef('Target')] = None, sort: bool = True)`**
  Allows for fast query of nodes

- **`graph_copy(self, g: 'Graph', val_map: dict[torch.fx.node.Node, torch.fx.node.Node], return_output_node=False) -> 'Optional[Argument]'`**
  Copy all nodes from a given graph into ``self``.

### `GraphModule`

GraphModule is an nn.Module generated from an fx.Graph. Graphmodule has a
``graph`` attribute, as well as ``code`` and ``forward`` attributes generated
from that ``graph``.

.. warning::

    When ``graph`` is reassigned, ``code`` and ``forward`` will be automatically
    regenerated. However, if you edit the contents of the ``graph`` without reassigning
    the ``graph`` attribute itself, you must call ``recompile()`` to update the generated
    code.

.. note::
    Backwards-compatibility for this API is guaranteed.

#### Methods

- **`to_folder(self, folder: Union[str, os.PathLike], module_name: str = 'FxModule')`**
  Dumps out module to ``folder`` with ``module_name`` so that it can be

- **`add_submodule(self, target: str, m: torch.nn.modules.module.Module) -> bool`**
  Adds the given submodule to ``self``.

- **`delete_submodule(self, target: str) -> bool`**
  Deletes the given submodule from ``self``.

### `HeteroBasisConv`

Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the submodules as regular attributes::

    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will also have their
parameters converted when you call :meth:`to`, etc.

.. note::
    As per the example above, an ``__init__()`` call to the parent class
    must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or
                evaluation mode.
:vartype training: bool

#### Methods

- **`reset_parameters(self)`**

- **`forward(self, edge_type: torch.Tensor, *args, **kwargs) -> torch.Tensor`**
  Define the computation performed at every call.

### `Linear`

Applies a linear transformation to the incoming data.

.. math::
    \mathbf{x}^{\prime} = \mathbf{x} \mathbf{W}^{\top} + \mathbf{b}

In contrast to :class:`torch.nn.Linear`, it supports lazy initialization
and customizable weight and bias initialization.

Args:
    in_channels (int): Size of each input sample. Will be initialized
        lazily in case it is given as :obj:`-1`.
    out_channels (int): Size of each output sample.
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    weight_initializer (str, optional): The initializer for the weight
        matrix (:obj:`"glorot"`, :obj:`"uniform"`, :obj:`"kaiming_uniform"`
        or :obj:`None`).
        If set to :obj:`None`, will match default weight initialization of
        :class:`torch.nn.Linear`. (default: :obj:`None`)
    bias_initializer (str, optional): The initializer for the bias vector
        (:obj:`"zeros"` or :obj:`None`).
        If set to :obj:`None`, will match default bias initialization of
        :class:`torch.nn.Linear`. (default: :obj:`None`)

Shapes:
    - **input:** features :math:`(*, F_{in})`
    - **output:** features :math:`(*, F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor) -> torch.Tensor`**
  Forward pass.

- **`initialize_parameters(self, module, input)`**

### `LinearAlign`

Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the submodules as regular attributes::

    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will also have their
parameters converted when you call :meth:`to`, etc.

.. note::
    As per the example above, an ``__init__()`` call to the parent class
    must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or
                evaluation mode.
:vartype training: bool

#### Methods

- **`forward(self, x_dict: Dict[Union[str, Tuple[str, str, str]], torch.Tensor]) -> Dict[Union[str, Tuple[str, str, str]], torch.Tensor]`**
  Define the computation performed at every call.

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

### `Module`

Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the submodules as regular attributes::

    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will also have their
parameters converted when you call :meth:`to`, etc.

.. note::
    As per the example above, an ``__init__()`` call to the parent class
    must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or
                evaluation mode.
:vartype training: bool

#### Methods

- **`forward(self, *input: Any) -> None`**
  Define the computation performed at every call.

- **`register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None`**
  Add a buffer to the module.

- **`register_parameter(self, name: str, param: Optional[torch.nn.parameter.Parameter]) -> None`**
  Add a parameter to the module.

### `Node`

``Node`` is the data structure that represents individual operations within
a ``Graph``. For the most part, Nodes represent callsites to various entities,
such as operators, methods, and Modules (some exceptions include nodes that
specify function inputs and outputs). Each ``Node`` has a function specified
by its ``op`` property. The ``Node`` semantics for each value of ``op`` are as follows:

- ``placeholder`` represents a function input. The ``name`` attribute specifies the name this value will take on.
  ``target`` is similarly the name of the argument. ``args`` holds either: 1) nothing, or 2) a single argument
  denoting the default parameter of the function input. ``kwargs`` is don't-care. Placeholders correspond to
  the function parameters (e.g. ``x``) in the graph printout.
- ``get_attr`` retrieves a parameter from the module hierarchy. ``name`` is similarly the name the result of the
  fetch is assigned to. ``target`` is the fully-qualified name of the parameter's position in the module hierarchy.
  ``args`` and ``kwargs`` are don't-care
- ``call_function`` applies a free function to some values. ``name`` is similarly the name of the value to assign
  to. ``target`` is the function to be applied. ``args`` and ``kwargs`` represent the arguments to the function,
  following the Python calling convention
- ``call_module`` applies a module in the module hierarchy's ``forward()`` method to given arguments. ``name`` is
  as previous. ``target`` is the fully-qualified name of the module in the module hierarchy to call.
  ``args`` and ``kwargs`` represent the arguments to invoke the module on, *excluding the self argument*.
- ``call_method`` calls a method on a value. ``name`` is as similar. ``target`` is the string name of the method
  to apply to the ``self`` argument. ``args`` and ``kwargs`` represent the arguments to invoke the module on,
  *including the self argument*
- ``output`` contains the output of the traced function in its ``args[0]`` attribute. This corresponds to the "return" statement
  in the Graph printout.

.. note::
    Backwards-compatibility for this API is guaranteed.

#### Methods

- **`prepend(self, x: 'Node') -> None`**
  Insert x before this node in the list of nodes in the graph. Example::

- **`append(self, x: 'Node') -> None`**
  Insert ``x`` after this node in the list of nodes in the graph.

- **`update_arg(self, idx: int, arg: Union[tuple['Argument', ...], collections.abc.Sequence['Argument'], collections.abc.Mapping[str, 'Argument'], slice, range, ForwardRef('Node'), str, int, float, bool, complex, torch.dtype, torch.Tensor, torch.device, torch.memory_format, torch.layout, torch._ops.OpOverload, torch.SymInt, torch.SymBool, torch.SymFloat, NoneType]) -> None`**
  Update an existing positional argument to contain the new value

### `NodeType`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

#### Methods

- **`encode(self, /, encoding='utf-8', errors='strict')`**
  Encode the string using the codec registered for encoding.

- **`replace(self, old, new, count=-1, /)`**
  Return a copy with all occurrences of substring old replaced by new.

- **`split(self, /, sep=None, maxsplit=-1)`**
  Return a list of the substrings in the string, using sep as the separator string.

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

### `ToHeteroWithBasesTransformer`

A :class:`Transformer` executes an FX graph node-by-node, applies
transformations to each node, and produces a new :class:`torch.nn.Module`.
It exposes a :func:`transform` method that returns the transformed
:class:`~torch.nn.Module`.
:class:`Transformer` works entirely symbolically.

Methods in the :class:`Transformer` class can be overridden to customize
the behavior of transformation.

.. code-block:: none

    transform()
        +-- Iterate over each node in the graph
            +-- placeholder()
            +-- get_attr()
            +-- call_function()
            +-- call_method()
            +-- call_module()
            +-- call_message_passing_module()
            +-- call_global_pooling_module()
            +-- output()
        +-- Erase unused nodes in the graph
        +-- Iterate over each children module
            +-- init_submodule()

In contrast to the :class:`torch.fx.Transformer` class, the
:class:`Transformer` exposes additional functionality:

#. It subdivides :func:`call_module` into nodes that call a regular
   :class:`torch.nn.Module` (:func:`call_module`), a
   :class:`MessagePassing` module (:func:`call_message_passing_module`),
   or a :class:`GlobalPooling` module (:func:`call_global_pooling_module`).

#. It allows to customize or initialize new children modules via
   :func:`init_submodule`

#. It allows to infer whether a node returns node-level or edge-level
   information via :meth:`is_edge_level`.

Args:
    module (torch.nn.Module): The module to be transformed.
    input_map (Dict[str, str], optional): A dictionary holding information
        about the type of input arguments of :obj:`module.forward`.
        For example, in case :obj:`arg` is a node-level argument, then
        :obj:`input_map['arg'] = 'node'`, and
        :obj:`input_map['arg'] = 'edge'` otherwise.
        In case :obj:`input_map` is not further specified, will try to
        automatically determine the correct type of input arguments.
        (default: :obj:`None`)
    debug (bool, optional): If set to :obj:`True`, will perform
        transformation in debug mode. (default: :obj:`False`)

#### Methods

- **`validate(self)`**

- **`transform(self) -> torch.fx.graph_module.GraphModule`**
  Transforms :obj:`self.module` and returns a transformed

- **`placeholder(self, node: torch.fx.node.Node, target: Any, name: str)`**

### `Transformer`

A :class:`Transformer` executes an FX graph node-by-node, applies
transformations to each node, and produces a new :class:`torch.nn.Module`.
It exposes a :func:`transform` method that returns the transformed
:class:`~torch.nn.Module`.
:class:`Transformer` works entirely symbolically.

Methods in the :class:`Transformer` class can be overridden to customize
the behavior of transformation.

.. code-block:: none

    transform()
        +-- Iterate over each node in the graph
            +-- placeholder()
            +-- get_attr()
            +-- call_function()
            +-- call_method()
            +-- call_module()
            +-- call_message_passing_module()
            +-- call_global_pooling_module()
            +-- output()
        +-- Erase unused nodes in the graph
        +-- Iterate over each children module
            +-- init_submodule()

In contrast to the :class:`torch.fx.Transformer` class, the
:class:`Transformer` exposes additional functionality:

#. It subdivides :func:`call_module` into nodes that call a regular
   :class:`torch.nn.Module` (:func:`call_module`), a
   :class:`MessagePassing` module (:func:`call_message_passing_module`),
   or a :class:`GlobalPooling` module (:func:`call_global_pooling_module`).

#. It allows to customize or initialize new children modules via
   :func:`init_submodule`

#. It allows to infer whether a node returns node-level or edge-level
   information via :meth:`is_edge_level`.

Args:
    module (torch.nn.Module): The module to be transformed.
    input_map (Dict[str, str], optional): A dictionary holding information
        about the type of input arguments of :obj:`module.forward`.
        For example, in case :obj:`arg` is a node-level argument, then
        :obj:`input_map['arg'] = 'node'`, and
        :obj:`input_map['arg'] = 'edge'` otherwise.
        In case :obj:`input_map` is not further specified, will try to
        automatically determine the correct type of input arguments.
        (default: :obj:`None`)
    debug (bool, optional): If set to :obj:`True`, will perform
        transformation in debug mode. (default: :obj:`False`)

#### Methods

- **`placeholder(self, node: torch.fx.node.Node, target: Any, name: str)`**

- **`get_attr(self, node: torch.fx.node.Node, target: Any, name: str)`**

- **`call_message_passing_module(self, node: torch.fx.node.Node, target: Any, name: str)`**
