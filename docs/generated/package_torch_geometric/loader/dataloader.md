# dataloader

Part of `torch_geometric.loader`
Module: `torch_geometric.loader.dataloader`

## Functions (1)

### `default_collate(batch)`

Take in a batch of data and put the elements within the batch into a tensor with an additional outer dimension - batch size.

The exact output type can be a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
This is used as the default function for collation when
`batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

Here is the general input type (based on the type of the element within the batch) to output type mapping:

    * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
    * NumPy Arrays -> :class:`torch.Tensor`
    * `float` -> :class:`torch.Tensor`
    * `int` -> :class:`torch.Tensor`
    * `str` -> `str` (unchanged)
    * `bytes` -> `bytes` (unchanged)
    * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
    * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]),
      default_collate([V2_1, V2_2, ...]), ...]`
    * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]),
      default_collate([V2_1, V2_2, ...]), ...]`

Args:
    batch: a single batch to be collated

Examples:
    >>> # xdoctest: +SKIP
    >>> # Example with a batch of `int`s:
    >>> default_collate([0, 1, 2, 3])
    tensor([0, 1, 2, 3])
    >>> # Example with a batch of `str`s:
    >>> default_collate(['a', 'b', 'c'])
    ['a', 'b', 'c']
    >>> # Example with `Map` inside the batch:
    >>> default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
    {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
    >>> # Example with `NamedTuple` inside the batch:
    >>> Point = namedtuple('Point', ['x', 'y'])
    >>> default_collate([Point(0, 0), Point(1, 1)])
    Point(x=tensor([0, 1]), y=tensor([0, 1]))
    >>> # Example with `Tuple` inside the batch:
    >>> default_collate([(0, 1), (2, 3)])
    [tensor([0, 2]), tensor([1, 3])]
    >>> # Example with `List` inside the batch:
    >>> default_collate([[0, 1], [2, 3]])
    [tensor([0, 2]), tensor([1, 3])]
    >>> # Two options to extend `default_collate` to handle specific type
    >>> # Option 1: Write custom collate function and invoke `default_collate`
    >>> def custom_collate(batch):
    ...     elem = batch[0]
    ...     if isinstance(elem, CustomType):  # Some custom condition
    ...         return ...
    ...     else:  # Fall back to `default_collate`
    ...         return default_collate(batch)
    >>> # Option 2: In-place modify `default_collate_fn_map`
    >>> def collate_customtype_fn(batch, *, collate_fn_map=None):
    ...     return ...
    >>> default_collate_fn_map.update(CustomType, collate_customtype_fn)
    >>> default_collate(batch)  # Handle `CustomType` automatically

## Classes (10)

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

### `Collater`

### `DataLoader`

A data loader which merges data objects from a
:class:`torch_geometric.data.Dataset` to a mini-batch.
Data objects can be either of type :class:`~torch_geometric.data.Data` or
:class:`~torch_geometric.data.HeteroData`.

Args:
    dataset (Dataset): The dataset from which to load the data.
    batch_size (int, optional): How many samples per batch to load.
        (default: :obj:`1`)
    shuffle (bool, optional): If set to :obj:`True`, the data will be
        reshuffled at every epoch. (default: :obj:`False`)
    follow_batch (List[str], optional): Creates assignment batch
        vectors for each key in the list. (default: :obj:`None`)
    exclude_keys (List[str], optional): Will exclude each key in the
        list. (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`.

### `Dataset`

Dataset base class for creating graph datasets.
See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
create_dataset.html>`__ for the accompanying tutorial.

Args:
    root (str, optional): Root directory where the dataset should be saved.
        (optional: :obj:`None`)
    transform (callable, optional): A function/transform that takes in a
        :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object and returns a
        transformed version.
        The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        a :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object and returns a
        transformed version.
        The data object will be transformed before being saved to disk.
        (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in a
        :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object and returns a
        boolean value, indicating whether the data object should be
        included in the final dataset. (default: :obj:`None`)
    log (bool, optional): Whether to print any console output while
        downloading and processing the dataset. (default: :obj:`True`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

- **`len(self) -> int`**
  Returns the number of data objects stored in the dataset.

### `DatasetAdapter`

Iterable-style DataPipe.

All DataPipes that represent an iterable of data samples should subclass this.
This style of DataPipes is particularly useful when data come from a stream, or
when the number of samples is too large to fit them all in memory. ``IterDataPipe`` is lazily initialized and its
elements are computed only when ``next()`` is called on the iterator of an ``IterDataPipe``.

All subclasses should overwrite :meth:`__iter__`, which would return an
iterator of samples in this DataPipe. Calling ``__iter__`` of an ``IterDataPipe`` automatically invokes its
method ``reset()``, which by default performs no operation. When writing a custom ``IterDataPipe``, users should
override ``reset()`` if necessary. The common usages include resetting buffers, pointers,
and various state variables within the custom ``IterDataPipe``.

Note:
    Only `one` iterator can be valid for each ``IterDataPipe`` at a time,
    and the creation a second iterator will invalidate the first one. This constraint is necessary because
    some ``IterDataPipe`` have internal buffers, whose states can become invalid if there are multiple iterators.
    The code example below presents details on how this constraint looks in practice.
    If you have any feedback related to this constraint, please see `GitHub IterDataPipe Single Iterator Issue`_.

These DataPipes can be invoked in two ways, using the class constructor or applying their
functional form onto an existing ``IterDataPipe`` (recommended, available to most but not all DataPipes).
You can chain multiple `IterDataPipe` together to form a pipeline that will perform multiple
operations in succession.

.. _GitHub IterDataPipe Single Iterator Issue:
    https://github.com/pytorch/data/issues/45

Note:
    When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
    item in the DataPipe will be yielded from the :class:`~torch.utils.data.DataLoader`
    iterator. When :attr:`num_workers > 0`, each worker process will have a
    different copy of the DataPipe object, so it is often desired to configure
    each copy independently to avoid having duplicate data returned from the
    workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
    process, returns information about the worker. It can be used in either the
    dataset's :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` 's
    :attr:`worker_init_fn` option to modify each copy's behavior.

Examples:
    General Usage:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
        >>> dp = IterableWrapper(range(10))
        >>> map_dp_1 = Mapper(dp, lambda x: x + 1)  # Using class constructor
        >>> map_dp_2 = dp.map(lambda x: x + 1)  # Using functional form (recommended)
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> filter_dp = map_dp_1.filter(lambda x: x % 2 == 0)
        >>> list(filter_dp)
        [2, 4, 6, 8, 10]
    Single Iterator Constraint Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
        >>> source_dp = IterableWrapper(range(10))
        >>> it1 = iter(source_dp)
        >>> list(it1)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> it1 = iter(source_dp)
        >>> it2 = iter(source_dp)  # The creation of a new iterator invalidates `it1`
        >>> next(it2)
        0
        >>> next(it1)  # Further usage of `it1` will raise a `RunTimeError`

#### Methods

- **`is_shardable(self) -> bool`**

- **`apply_sharding(self, num_shards: int, shard_idx: int) -> None`**

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

### `TensorFrame`

### `torch_frame`

The base class of the class hierarchy.

When called, it accepts no arguments and returns a new featureless
instance that has no instance attributes and cannot be given any.
