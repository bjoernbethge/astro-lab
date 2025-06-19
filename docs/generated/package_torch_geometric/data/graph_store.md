# graph_store

Part of `torch_geometric.data`
Module: `torch_geometric.data.graph_store`

## Description

This class defines the abstraction for a backend-agnostic graph store. The
goal of the graph store is to abstract away all graph edge index memory
management so that varying implementations can allow for independent scale-out.

This particular graph store abstraction makes a few key assumptions:
* The edge indices we care about storing are represented either in COO, CSC,
  or CSR format. They can be uniquely identified by an edge type (in PyG,
  this is a tuple of the source node, relation type, and destination node).
* Edge indices are static once they are stored in the graph. That is, we do not
  support dynamic modification of edge indices once they have been inserted
  into the graph store.

It is the job of a graph store implementor class to handle these assumptions
properly. For example, a simple in-memory graph store implementation may
concatenate all metadata values with an edge index and use this as a unique
index in a KV store. More complicated implementations may choose to partition
the graph in interesting manners based on the provided metadata.

## Functions (5)

### `abstractmethod(funcobj)`

A decorator indicating abstract methods.

Requires that the metaclass is ABCMeta or derived from it.  A
class that has a metaclass derived from ABCMeta cannot be
instantiated unless all of its abstract methods are overridden.
The abstract methods can be called using any of the normal
'super' call mechanisms.  abstractmethod() may be used to declare
abstract methods for properties and descriptors.

Usage:

    class C(metaclass=ABCMeta):
        @abstractmethod
        def my_abstract_method(self, arg1, arg2, argN):
            ...

### `dataclass(cls=None, /, *, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False, weakref_slot=False)`

Add dunder methods based on the fields defined in the class.

Examines PEP 526 __annotations__ to determine fields.

If init is true, an __init__() method is added to the class. If repr
is true, a __repr__() method is added. If order is true, rich
comparison dunder methods are added. If unsafe_hash is true, a
__hash__() method is added. If frozen is true, fields may not be
assigned to after instance creation. If match_args is true, the
__match_args__ tuple is added. If kw_only is true, then by default
all fields are keyword-only. If slots is true, a new class with a
__slots__ attribute is returned.

### `index2ptr(index: torch.Tensor, size: Optional[int] = None) -> torch.Tensor`

### `index_sort(inputs: torch.Tensor, max_value: Optional[int] = None, stable: bool = False) -> Tuple[torch.Tensor, torch.Tensor]`

Sorts the elements of the :obj:`inputs` tensor in ascending order.
It is expected that :obj:`inputs` is one-dimensional and that it only
contains positive integer values. If :obj:`max_value` is given, it can
be used by the underlying algorithm for better performance.

Args:
    inputs (torch.Tensor): A vector with positive integer values.
    max_value (int, optional): The maximum value stored inside
        :obj:`inputs`. This value can be an estimation, but needs to be
        greater than or equal to the real maximum.
        (default: :obj:`None`)
    stable (bool, optional): Makes the sorting routine stable, which
        guarantees that the order of equivalent elements is preserved.
        (default: :obj:`False`)

### `ptr2index(ptr: torch.Tensor, output_size: Optional[int] = None) -> torch.Tensor`

## Classes (9)

### `ABC`

Helper class that provides a standard way to create an ABC using
inheritance.

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `CastMixin`

### `EdgeAttr`

Defines the attributes of a :obj:`GraphStore` edge.
It holds all the parameters necessary to uniquely identify an edge from
the :class:`GraphStore`.

Note that the order of the attributes is important; this is the order in
which attributes must be provided for indexing calls. :class:`GraphStore`
implementations can define a different ordering by overriding
:meth:`EdgeAttr.__init__`.

### `EdgeLayout`

Create a collection of name/value pairs.

Example enumeration:

>>> class Color(Enum):
...     RED = 1
...     BLUE = 2
...     GREEN = 3

Access them by:

- attribute access::

>>> Color.RED
<Color.RED: 1>

- value lookup:

>>> Color(1)
<Color.RED: 1>

- name lookup:

>>> Color['RED']
<Color.RED: 1>

Enumerations can be iterated over, and know how many members they have:

>>> len(Color)
3

>>> list(Color)
[<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]

Methods can be added to enumerations, and members can have their own
attributes -- see the documentation for details.

### `Enum`

Create a collection of name/value pairs.

Example enumeration:

>>> class Color(Enum):
...     RED = 1
...     BLUE = 2
...     GREEN = 3

Access them by:

- attribute access::

>>> Color.RED
<Color.RED: 1>

- value lookup:

>>> Color(1)
<Color.RED: 1>

- name lookup:

>>> Color['RED']
<Color.RED: 1>

Enumerations can be iterated over, and know how many members they have:

>>> len(Color)
3

>>> list(Color)
[<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]

Methods can be added to enumerations, and members can have their own
attributes -- see the documentation for details.

### `GraphStore`

An abstract base class to access edges from a remote graph store.

Args:
    edge_attr_cls (EdgeAttr, optional): A user-defined
        :class:`EdgeAttr` class to customize the required attributes and
        their ordering to uniquely identify edges. (default: :obj:`None`)

#### Methods

- **`put_edge_index(self, edge_index: Tuple[torch.Tensor, torch.Tensor], *args, **kwargs) -> bool`**
  Synchronously adds an :obj:`edge_index` tuple to the

- **`get_edge_index(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]`**
  Synchronously obtains an :obj:`edge_index` tuple from the

- **`remove_edge_index(self, *args, **kwargs) -> bool`**
  Synchronously deletes an :obj:`edge_index` tuple from the

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
