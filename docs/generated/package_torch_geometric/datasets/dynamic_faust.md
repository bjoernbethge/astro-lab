# dynamic_faust

Part of `torch_geometric.datasets`
Module: `torch_geometric.datasets.dynamic_faust`

## Classes (4)

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

### `DynamicFAUST`

The dynamic FAUST humans dataset from the `"Dynamic FAUST: Registering
Human Bodies in Motion"
<http://files.is.tue.mpg.de/black/papers/dfaust2017.pdf>`_ paper.

.. note::

    Data objects hold mesh faces instead of edge indices.
    To convert the mesh to a graph, use the
    :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
    To convert the mesh to a point cloud, use the
    :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
    sample a fixed number of points on the mesh faces according to their
    face area.

Args:
    root (str): Root directory where the dataset should be saved.
    subjects (list, optional): List of subjects to include in the
        dataset. Can include the subjects :obj:`"50002"`, :obj:`"50004"`,
        :obj:`"50007"`, :obj:`"50009"`, :obj:`"50020"`, :obj:`"50021"`,
        :obj:`"50022"`, :obj:`"50025"`, :obj:`"50026"`, :obj:`"50027"`.
        If set to :obj:`None`, the dataset will contain all subjects.
        (default: :obj:`None`)
    categories (list, optional): List of categories to include in the
        dataset. Can include the categories :obj:`"chicken_wings"`,
        :obj:`"hips"`, :obj:`"jiggle_on_toes"`, :obj:`"jumping_jacks"`,
        :obj:`"knees"`, :obj:`"light_hopping_loose"`,
        :obj:`"light_hopping_stiff"`, :obj:`"one_leg_jump"`,
        :obj:`"one_leg_loose"`, :obj:`"personal_move"`, :obj:`"punching"`,
        :obj:`"running_on_spot"`, :obj:`"running_on_spot_bugfix"`,
        :obj:`"shake_arms"`, :obj:`"shake_hips"`, :obj:`"shoulders"`.
        If set to :obj:`None`, the dataset will contain all categories.
        (default: :obj:`None`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `InMemoryDataset`

Dataset base class for creating graph datasets which easily fit
into CPU memory.
See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
create_dataset.html#creating-in-memory-datasets>`__ for the accompanying
tutorial.

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

- **`len(self) -> int`**
  Returns the number of data objects stored in the dataset.

- **`get(self, idx: int) -> torch_geometric.data.data.BaseData`**
  Gets the data object at index :obj:`idx`.

- **`load(self, path: str, data_cls: Type[torch_geometric.data.data.BaseData] = <class 'torch_geometric.data.data.Data'>) -> None`**
  Loads the dataset from the file path :obj:`path`.

### `product`

product(*iterables, repeat=1) --> product object

Cartesian product of input iterables.  Equivalent to nested for-loops.

For example, product(A, B) returns the same as:  ((x,y) for x in A for y in B).
The leftmost iterators are in the outermost for-loop, so the output tuples
cycle in a manner similar to an odometer (with the rightmost element changing
on every iteration).

To compute the product of an iterable with itself, specify the number
of repetitions with the optional repeat keyword argument. For example,
product(A, repeat=4) means the same as product(A, A, A, A).

product('ab', range(3)) --> ('a',0) ('a',1) ('a',2) ('b',0) ('b',1) ('b',2)
product((0,1), (0,1), (0,1)) --> (0,0,0) (0,0,1) (0,1,0) (0,1,1) (1,0,0) ...
