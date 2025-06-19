# utils

Part of `torch_geometric.profile`
Module: `torch_geometric.profile.utils`

## Functions (9)

### `byte_to_megabyte(value: int, digits: int = 2) -> float`

### `count_parameters(model: torch.nn.modules.module.Module) -> int`

Given a :class:`torch.nn.Module`, count its trainable parameters.

Args:
    model (torch.nn.Model): The model.

### `get_cpu_memory_from_gc() -> int`

Returns the used CPU memory in bytes, as reported by the
:python:`Python` garbage collector.

### `get_data_size(data: torch_geometric.data.data.BaseData) -> int`

Given a :class:`torch_geometric.data.Data` object, get its theoretical
memory usage in bytes.

Args:
    data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
        The :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` graph object.

### `get_gpu_memory_from_gc(device: int = 0) -> int`

Returns the used GPU memory in bytes, as reported by the
:python:`Python` garbage collector.

Args:
    device (int, optional): The GPU device identifier. (default: :obj:`1`)

### `get_gpu_memory_from_ipex(device: int = 0, digits=2) -> Tuple[float, float, float]`

Returns the XPU memory statistics.

Args:
    device (int, optional): The GPU device identifier. (default: :obj:`0`)
    digits (int): The number of decimals to use for megabytes.
        (default: :obj:`2`)

### `get_gpu_memory_from_nvidia_smi(device: int = 0, digits: int = 2) -> Tuple[float, float]`

Returns the free and used GPU memory in megabytes, as reported by
:obj:`nivdia-smi`.

.. note::

    :obj:`nvidia-smi` will generally overestimate the amount of memory used
    by the actual program, see `here <https://pytorch.org/docs/stable/
    notes/faq.html#my-gpu-memory-isn-t-freed-properly>`__.

Args:
    device (int, optional): The GPU device identifier. (default: :obj:`1`)
    digits (int): The number of decimals to use for megabytes.
        (default: :obj:`2`)

### `get_model_size(model: torch.nn.modules.module.Module) -> int`

Given a :class:`torch.nn.Module`, get its actual disk size in bytes.

Args:
    model (torch model): The model.

### `medibyte_to_megabyte(value: int, digits: int = 2) -> float`

## Classes (6)

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

### `Mapping`

A Mapping is a generic container for associating key/value
pairs.

This class provides concrete generic implementations of all
methods except for __getitem__, __iter__, and __len__.

#### Methods

- **`get(self, key, default=None)`**
  D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.

- **`keys(self)`**
  D.keys() -> a set-like object providing a view on D's keys

- **`items(self)`**
  D.items() -> a set-like object providing a view on D's items

### `Sequence`

All the operations on a read-only sequence.

Concrete subclasses must override __new__ or __init__,
__getitem__, and __len__.

#### Methods

- **`index(self, value, start=0, stop=None)`**
  S.index(value, [start, [stop]]) -> integer -- return first index of value.

- **`count(self, value)`**
  S.count(value) -> integer -- return number of occurrences of value

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
