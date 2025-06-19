# temporal

Part of `torch_geometric.data`
Module: `torch_geometric.data.temporal`

## Functions (3)

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

### `prepare_idx(idx)`

### `size_repr(key: Any, value: Any, indent: int = 0) -> str`

## Classes (8)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `BaseData`

#### Methods

- **`stores_as(self, data: Self)`**

- **`to_dict(self) -> Dict[str, Any]`**
  Returns a dictionary of stored key/value pairs.

- **`to_namedtuple(self) -> <function NamedTuple at 0x000001FE17E66F20>`**
  Returns a :obj:`NamedTuple` of stored key/value pairs.

### `BaseStorage`

A MutableMapping is a generic container for associating
key/value pairs.

This class provides concrete generic implementations of all
methods except for __getitem__, __setitem__, __delitem__,
__iter__, and __len__.

#### Methods

- **`keys(self, *args: str) -> torch_geometric.data.view.KeysView`**
  D.keys() -> a set-like object providing a view on D's keys

- **`values(self, *args: str) -> torch_geometric.data.view.ValuesView`**
  D.values() -> an object providing a view on D's values

- **`items(self, *args: str) -> torch_geometric.data.view.ItemsView`**
  D.items() -> a set-like object providing a view on D's items

### `EdgeStorage`

A storage for edge-level information.

We support multiple ways to store edge connectivity in a
:class:`EdgeStorage` object:

* :obj:`edge_index`: A :class:`torch.LongTensor` holding edge indices in
  COO format with shape :obj:`[2, num_edges]` (the default format)

* :obj:`adj`: A :class:`torch_sparse.SparseTensor` holding edge indices in
  a sparse format, supporting both COO and CSR format.

* :obj:`adj_t`: A **transposed** :class:`torch_sparse.SparseTensor` holding
  edge indices in a sparse format, supporting both COO and CSR format.
  This is the most efficient one for graph-based deep learning models as
  indices are sorted based on target nodes.

#### Methods

- **`size(self, dim: Optional[int] = None) -> Union[Tuple[Optional[int], Optional[int]], int, NoneType]`**

- **`is_node_attr(self, key: str) -> bool`**

- **`is_edge_attr(self, key: str) -> bool`**

### `GlobalStorage`

A storage for both node-level and edge-level information.

#### Methods

- **`size(self, dim: Optional[int] = None) -> Union[Tuple[Optional[int], Optional[int]], int, NoneType]`**

- **`is_node_attr(self, key: str) -> bool`**

- **`is_edge_attr(self, key: str) -> bool`**

### `NodeStorage`

A storage for node-level information.

#### Methods

- **`is_node_attr(self, key: str) -> bool`**

- **`is_edge_attr(self, key: str) -> bool`**

- **`node_attrs(self) -> List[str]`**

### `TemporalData`

A data object composed by a stream of events describing a temporal
graph.
The :class:`~torch_geometric.data.TemporalData` object can hold a list of
events (that can be understood as temporal edges in a graph) with
structured messages.
An event is composed by a source node, a destination node, a timestamp
and a message. Any *Continuous-Time Dynamic Graph* (CTDG) can be
represented with these four values.

In general, :class:`~torch_geometric.data.TemporalData` tries to mimic
the behavior of a regular :python:`Python` dictionary.
In addition, it provides useful functionality for analyzing graph
structures, and provides basic PyTorch tensor functionalities.

.. code-block:: python

    from torch import Tensor
    from torch_geometric.data import TemporalData

    events = TemporalData(
        src=Tensor([1,2,3,4]),
        dst=Tensor([2,3,4,5]),
        t=Tensor([1000,1010,1100,2000]),
        msg=Tensor([1,1,0,0])
    )

    # Add additional arguments to `events`:
    events.y = Tensor([1,1,0,0])

    # It is also possible to set additional arguments in the constructor
    events = TemporalData(
        ...,
        y=Tensor([1,1,0,0])
    )

    # Get the number of events:
    events.num_events
    >>> 4

    # Analyzing the graph structure:
    events.num_nodes
    >>> 5

    # PyTorch tensor functionality:
    events = events.pin_memory()
    events = events.to('cuda:0', non_blocking=True)

Args:
    src (torch.Tensor, optional): A list of source nodes for the events
        with shape :obj:`[num_events]`. (default: :obj:`None`)
    dst (torch.Tensor, optional): A list of destination nodes for the
        events with shape :obj:`[num_events]`. (default: :obj:`None`)
    t (torch.Tensor, optional): The timestamps for each event with shape
        :obj:`[num_events]`. (default: :obj:`None`)
    msg (torch.Tensor, optional): Messages feature matrix with shape
        :obj:`[num_events, num_msg_features]`. (default: :obj:`None`)
    **kwargs (optional): Additional attributes.

.. note::
    The shape of :obj:`src`, :obj:`dst`, :obj:`t` and the first dimension
    of :obj`msg` should be the same (:obj:`num_events`).

#### Methods

- **`index_select(self, idx: Any) -> 'TemporalData'`**

- **`stores_as(self, data: 'TemporalData')`**

- **`to_dict(self) -> Dict[str, Any]`**
  Returns a dictionary of stored key/value pairs.

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
