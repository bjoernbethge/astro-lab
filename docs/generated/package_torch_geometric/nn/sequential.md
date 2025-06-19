# sequential

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.sequential`

## Functions (4)

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

### `eval_type(value: Any, _globals: Dict[str, Any]) -> Type`

Returns the type hint of a string.

### `module_from_template(module_name: str, template_path: str, tmp_dirname: str, **kwargs: Any) -> Any`

### `split(content: str, sep: str) -> List[str]`

Splits :obj:`content` based on :obj:`sep`.
:obj:`sep` inside parentheses or square brackets are ignored.

## Classes (6)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `Child`

Child(name, param_names, return_names)

### `Parameter`

Parameter(name, type, type_repr, default)

### `Sequential`

An extension of the :class:`torch.nn.Sequential` container in order to
define a sequential GNN model.

Since GNN operators take in multiple input arguments,
:class:`torch_geometric.nn.Sequential` additionally expects both global
input arguments, and function header definitions of individual operators.
If omitted, an intermediate module will operate on the *output* of its
preceding module:

.. code-block:: python

    from torch.nn import Linear, ReLU
    from torch_geometric.nn import Sequential, GCNConv

    model = Sequential('x, edge_index', [
        (GCNConv(in_channels, 64), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (GCNConv(64, 64), 'x, edge_index -> x'),
        ReLU(inplace=True),
        Linear(64, out_channels),
    ])

Here, :obj:`'x, edge_index'` defines the input arguments of :obj:`model`,
and :obj:`'x, edge_index -> x'` defines the function header, *i.e.* input
arguments *and* return types of :class:`~torch_geometric.nn.conv.GCNConv`.

In particular, this also allows to create more sophisticated models,
such as utilizing :class:`~torch_geometric.nn.models.JumpingKnowledge`:

.. code-block:: python

    from torch.nn import Linear, ReLU, Dropout
    from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
    from torch_geometric.nn import global_mean_pool

    model = Sequential('x, edge_index, batch', [
        (Dropout(p=0.5), 'x -> x'),
        (GCNConv(dataset.num_features, 64), 'x, edge_index -> x1'),
        ReLU(inplace=True),
        (GCNConv(64, 64), 'x1, edge_index -> x2'),
        ReLU(inplace=True),
        (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
        (JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
        (global_mean_pool, 'x, batch -> x'),
        Linear(2 * 64, dataset.num_classes),
    ])

Args:
    input_args (str): The input arguments of the model.
    modules ([(Callable, str) or Callable]): A list of modules (with
        optional function header definitions). Alternatively, an
        :obj:`OrderedDict` of modules (and function header definitions) can
        be passed.

#### Methods

- **`reset_parameters(self) -> None`**
  Resets all learnable parameters of the module.

- **`forward(self, *args: Any, **kwargs: Any) -> Any`**

### `Signature`

Signature(param_dict, return_type, return_type_repr)

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
