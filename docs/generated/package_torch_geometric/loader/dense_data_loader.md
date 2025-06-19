# dense_data_loader

Part of `torch_geometric.loader`
Module: `torch_geometric.loader.dense_data_loader`

## Functions (2)

### `collate_fn(data_list: List[torch_geometric.data.data.Data]) -> torch_geometric.data.batch.Batch`

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

## Classes (4)

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

### `Data`

A data object describing a homogeneous graph.
The data object can hold node-level, link-level and graph-level attributes.
In general, :class:`~torch_geometric.data.Data` tries to mimic the
behavior of a regular :python:`Python` dictionary.
In addition, it provides useful functionality for analyzing graph
structures, and provides basic PyTorch tensor functionalities.
See `here <https://pytorch-geometric.readthedocs.io/en/latest/get_started/
introduction.html#data-handling-of-graphs>`__ for the accompanying
tutorial.

.. code-block:: python

    from torch_geometric.data import Data

    data = Data(x=x, edge_index=edge_index, ...)

    # Add additional arguments to `data`:
    data.train_idx = torch.tensor([...], dtype=torch.long)
    data.test_mask = torch.tensor([...], dtype=torch.bool)

    # Analyzing the graph structure:
    data.num_nodes
    >>> 23

    data.is_directed()
    >>> False

    # PyTorch tensor functionality:
    data = data.pin_memory()
    data = data.to('cuda:0', non_blocking=True)

Args:
    x (torch.Tensor, optional): Node feature matrix with shape
        :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
    edge_index (LongTensor, optional): Graph connectivity in COO format
        with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
    edge_attr (torch.Tensor, optional): Edge feature matrix with shape
        :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
    y (torch.Tensor, optional): Graph-level or node-level ground-truth
        labels with arbitrary shape. (default: :obj:`None`)
    pos (torch.Tensor, optional): Node position matrix with shape
        :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
    time (torch.Tensor, optional): The timestamps for each event with shape
        :obj:`[num_edges]` or :obj:`[num_nodes]`. (default: :obj:`None`)
    **kwargs (optional): Additional attributes.

#### Methods

- **`stores_as(self, data: Self)`**

- **`to_dict(self) -> Dict[str, Any]`**
  Returns a dictionary of stored key/value pairs.

- **`to_namedtuple(self) -> <function NamedTuple at 0x000001FE17E66F20>`**
  Returns a :obj:`NamedTuple` of stored key/value pairs.

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

### `DenseDataLoader`

A data loader which batches data objects from a
:class:`torch_geometric.data.dataset` to a
:class:`torch_geometric.data.Batch` object by stacking all attributes in a
new dimension.

.. note::

    To make use of this data loader, all graph attributes in the dataset
    need to have the same shape.
    In particular, this data loader should only be used when working with
    *dense* adjacency matrices.

Args:
    dataset (Dataset): The dataset from which to load the data.
    batch_size (int, optional): How many samples per batch to load.
        (default: :obj:`1`)
    shuffle (bool, optional): If set to :obj:`True`, the data will be
        reshuffled at every epoch. (default: :obj:`False`)
    **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`, such as :obj:`drop_last` or
        :obj:`num_workers`.
