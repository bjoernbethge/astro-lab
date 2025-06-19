# fx

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.fx`

## Functions (4)

### `get_submodule(module: torch.nn.modules.module.Module, target: str) -> torch.nn.modules.module.Module`

### `is_global_pooling_op(module: torch.nn.modules.module.Module, op: str, target: str) -> bool`

### `is_message_passing_op(module: torch.nn.modules.module.Module, op: str, target: str) -> bool`

### `symbolic_trace(module: torch.nn.modules.module.Module, concrete_args: Optional[Dict[str, Any]] = None) -> torch.fx.graph_module.GraphModule`

## Classes (9)

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

### `ModuleDict`

Holds submodules in a dictionary.

:class:`~torch.nn.ModuleDict` can be indexed like a regular Python dictionary,
but modules it contains are properly registered, and will be visible by all
:class:`~torch.nn.Module` methods.

:class:`~torch.nn.ModuleDict` is an **ordered** dictionary that respects

* the order of insertion, and

* in :meth:`~torch.nn.ModuleDict.update`, the order of the merged
  ``OrderedDict``, ``dict`` (started from Python 3.6) or another
  :class:`~torch.nn.ModuleDict` (the argument to
  :meth:`~torch.nn.ModuleDict.update`).

Note that :meth:`~torch.nn.ModuleDict.update` with other unordered mapping
types (e.g., Python's plain ``dict`` before Python version 3.6) does not
preserve the order of the merged mapping.

Args:
    modules (iterable, optional): a mapping (dictionary) of (string: module)
        or an iterable of key-value pairs of type (string, module)

Example::

    class MyModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.choices = nn.ModuleDict({
                    'conv': nn.Conv2d(10, 10, 3),
                    'pool': nn.MaxPool2d(3)
            })
            self.activations = nn.ModuleDict([
                    ['lrelu', nn.LeakyReLU()],
                    ['prelu', nn.PReLU()]
            ])

        def forward(self, x, choice, act):
            x = self.choices[choice](x)
            x = self.activations[act](x)
            return x

#### Methods

- **`clear(self) -> None`**
  Remove all items from the ModuleDict.

- **`pop(self, key: str) -> torch.nn.modules.module.Module`**
  Remove key from the ModuleDict and return its module.

- **`keys(self) -> collections.abc.Iterable[str]`**
  Return an iterable of the ModuleDict keys.

### `ModuleList`

Holds submodules in a list.

:class:`~torch.nn.ModuleList` can be indexed like a regular Python list, but
modules it contains are properly registered, and will be visible by all
:class:`~torch.nn.Module` methods.

Args:
    modules (iterable, optional): an iterable of modules to add

Example::

    class MyModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

        def forward(self, x):
            # ModuleList can act as an iterable, or be indexed using ints
            for i, l in enumerate(self.linears):
                x = self.linears[i // 2](x) + l(x)
            return x

#### Methods

- **`insert(self, index: int, module: torch.nn.modules.module.Module) -> None`**
  Insert a given module before a given index in the list.

- **`append(self, module: torch.nn.modules.module.Module) -> 'ModuleList'`**
  Append a given module to the end of the list.

- **`pop(self, key: Union[int, slice]) -> torch.nn.modules.module.Module`**

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

### `Sequential`

A sequential container.

Modules will be added to it in the order they are passed in the
constructor. Alternatively, an ``OrderedDict`` of modules can be
passed in. The ``forward()`` method of ``Sequential`` accepts any
input and forwards it to the first module it contains. It then
"chains" outputs to inputs sequentially for each subsequent module,
finally returning the output of the last module.

The value a ``Sequential`` provides over manually calling a sequence
of modules is that it allows treating the whole container as a
single module, such that performing a transformation on the
``Sequential`` applies to each of the modules it stores (which are
each a registered submodule of the ``Sequential``).

What's the difference between a ``Sequential`` and a
:class:`torch.nn.ModuleList`? A ``ModuleList`` is exactly what it
sounds like--a list for storing ``Module`` s! On the other hand,
the layers in a ``Sequential`` are connected in a cascading way.

Example::

    # Using Sequential to create a small model. When `model` is run,
    # input will first be passed to `Conv2d(1,20,5)`. The output of
    # `Conv2d(1,20,5)` will be used as the input to the first
    # `ReLU`; the output of the first `ReLU` will become the input
    # for `Conv2d(20,64,5)`. Finally, the output of
    # `Conv2d(20,64,5)` will be used as input to the second `ReLU`
    model = nn.Sequential(
              nn.Conv2d(1,20,5),
              nn.ReLU(),
              nn.Conv2d(20,64,5),
              nn.ReLU()
            )

    # Using Sequential with OrderedDict. This is functionally the
    # same as the above code
    model = nn.Sequential(OrderedDict([
              ('conv1', nn.Conv2d(1,20,5)),
              ('relu1', nn.ReLU()),
              ('conv2', nn.Conv2d(20,64,5)),
              ('relu2', nn.ReLU())
            ]))

#### Methods

- **`pop(self, key: Union[int, slice]) -> torch.nn.modules.module.Module`**

- **`forward(self, input)`**
  Define the computation performed at every call.

- **`append(self, module: torch.nn.modules.module.Module) -> 'Sequential'`**
  Append a given module to the end.

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
