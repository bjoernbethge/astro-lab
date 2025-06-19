# data_parallel

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.data_parallel`

## Functions (1)

### `cumsum(x: torch.Tensor, dim: int = 0) -> torch.Tensor`

Returns the cumulative sum of elements of :obj:`x`.
In contrast to :meth:`torch.cumsum`, prepends the output with zero.

Args:
    x (torch.Tensor): The input tensor.
    dim (int, optional): The dimension to do the operation over.
        (default: :obj:`0`)

Example:
    >>> x = torch.tensor([2, 4, 1])
    >>> cumsum(x)
    tensor([0, 2, 6, 7])

## Classes (3)

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

### `DataParallel`

Implements data parallelism at the module level.

This container parallelizes the application of the given :attr:`module` by
splitting a list of :class:`torch_geometric.data.Data` objects and copying
them as :class:`torch_geometric.data.Batch` objects to each device.
In the forward pass, the module is replicated on each device, and each
replica handles a portion of the input.
During the backwards pass, gradients from each replica are summed into the
original module.

The batch size should be larger than the number of GPUs used.

The parallelized :attr:`module` must have its parameters and buffers on
:obj:`device_ids[0]`.

.. note::

    You need to use the :class:`torch_geometric.loader.DataListLoader` for
    this module.

.. warning::

    It is recommended to use
    :class:`torch.nn.parallel.DistributedDataParallel` instead of
    :class:`DataParallel` for multi-GPU training.
    :class:`DataParallel` is usually much slower than
    :class:`~torch.nn.parallel.DistributedDataParallel` even on a single
    machine.
    Take a look `here <https://github.com/pyg-team/pytorch_geometric/blob/
    master/examples/multi_gpu/distributed_batching.py>`_ for an example on
    how to use :pyg:`PyG` in combination with
    :class:`~torch.nn.parallel.DistributedDataParallel`.

Args:
    module (Module): Module to be parallelized.
    device_ids (list of int or torch.device): CUDA devices.
        (default: all devices)
    output_device (int or torch.device): Device location of output.
        (default: :obj:`device_ids[0]`)
    follow_batch (list or tuple, optional): Creates assignment batch
        vectors for each key in the list. (default: :obj:`None`)
    exclude_keys (list or tuple, optional): Will exclude each key in the
        list. (default: :obj:`None`)

#### Methods

- **`forward(self, data_list)`**

- **`scatter(self, data_list, device_ids)`**

### `chain`

chain(*iterables) --> chain object

Return a chain object whose .__next__() method returns elements from the
first iterable until it is exhausted, then elements from the next
iterable, until all of the iterables are exhausted.

#### Methods

- **`from_iterable(type, iterable, /)`**
  Alternative chain() constructor taking a single iterable argument that evaluates lazily.
