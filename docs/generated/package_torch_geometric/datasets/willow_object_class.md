# willow_object_class

Part of `torch_geometric.datasets`
Module: `torch_geometric.datasets.willow_object_class`

## Functions (2)

### `download_url(url: str, folder: str, log: bool = True, filename: Optional[str] = None)`

Downloads the content of an URL to a specific folder.

Args:
    url (str): The URL.
    folder (str): The folder.
    log (bool, optional): If :obj:`False`, will not print anything to the
        console. (default: :obj:`True`)
    filename (str, optional): The filename of the downloaded file. If set
        to :obj:`None`, will correspond to the filename given by the URL.
        (default: :obj:`None`)

### `extract_zip(path: str, folder: str, log: bool = True) -> None`

Extracts a zip archive to a specific folder.

Args:
    path (str): The path to the tar archive.
    folder (str): The folder.
    log (bool, optional): If :obj:`False`, will not print anything to the
        console. (default: :obj:`True`)

## Classes (5)

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

### `DataLoader`

Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.

The :class:`~torch.utils.data.DataLoader` supports both map-style and
iterable-style datasets with single- or multi-process loading, customizing
loading order and optional automatic batching (collation) and memory pinning.

See :py:mod:`torch.utils.data` documentation page for more details.

Args:
    dataset (Dataset): dataset from which to load the data.
    batch_size (int, optional): how many samples per batch to load
        (default: ``1``).
    shuffle (bool, optional): set to ``True`` to have the data reshuffled
        at every epoch (default: ``False``).
    sampler (Sampler or Iterable, optional): defines the strategy to draw
        samples from the dataset. Can be any ``Iterable`` with ``__len__``
        implemented. If specified, :attr:`shuffle` must not be specified.
    batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
        returns a batch of indices at a time. Mutually exclusive with
        :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
        and :attr:`drop_last`.
    num_workers (int, optional): how many subprocesses to use for data
        loading. ``0`` means that the data will be loaded in the main process.
        (default: ``0``)
    collate_fn (Callable, optional): merges a list of samples to form a
        mini-batch of Tensor(s).  Used when using batched loading from a
        map-style dataset.
    pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
        into device/CUDA pinned memory before returning them.  If your data elements
        are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
        see the example below.
    drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last batch
        will be smaller. (default: ``False``)
    timeout (numeric, optional): if positive, the timeout value for collecting a batch
        from workers. Should always be non-negative. (default: ``0``)
    worker_init_fn (Callable, optional): If not ``None``, this will be called on each
        worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
        input, after seeding and before data loading. (default: ``None``)
    multiprocessing_context (str or multiprocessing.context.BaseContext, optional): If
        ``None``, the default `multiprocessing context`_ of your operating system will
        be used. (default: ``None``)
    generator (torch.Generator, optional): If not ``None``, this RNG will be used
        by RandomSampler to generate random indexes and multiprocessing to generate
        ``base_seed`` for workers. (default: ``None``)
    prefetch_factor (int, optional, keyword-only arg): Number of batches loaded
        in advance by each worker. ``2`` means there will be a total of
        2 * num_workers batches prefetched across all workers. (default value depends
        on the set value for num_workers. If value of num_workers=0 default is ``None``.
        Otherwise, if value of ``num_workers > 0`` default is ``2``).
    persistent_workers (bool, optional): If ``True``, the data loader will not shut down
        the worker processes after a dataset has been consumed once. This allows to
        maintain the workers `Dataset` instances alive. (default: ``False``)
    pin_memory_device (str, optional): the device to :attr:`pin_memory` on if ``pin_memory`` is
        ``True``. If not given, the current :ref:`accelerator<accelerators>` will be the
        default. This argument is discouraged and subject to deprecated.
    in_order (bool, optional): If ``False``, the data loader will not enforce that batches
        are returned in a first-in, first-out order. Only applies when ``num_workers > 0``. (default: ``True``)


.. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
             cannot be an unpicklable object, e.g., a lambda function. See
             :ref:`multiprocessing-best-practices` on more details related
             to multiprocessing in PyTorch.

.. warning:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
             When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
             it instead returns an estimate based on ``len(dataset) / batch_size``, with proper
             rounding depending on :attr:`drop_last`, regardless of multi-process loading
             configurations. This represents the best guess PyTorch can make because PyTorch
             trusts user :attr:`dataset` code in correctly handling multi-process
             loading to avoid duplicate data.

             However, if sharding results in multiple workers having incomplete last batches,
             this estimate can still be inaccurate, because (1) an otherwise complete batch can
             be broken into multiple ones and (2) more than one batch worth of samples can be
             dropped when :attr:`drop_last` is set. Unfortunately, PyTorch can not detect such
             cases in general.

             See `Dataset Types`_ for more details on these two types of datasets and how
             :class:`~torch.utils.data.IterableDataset` interacts with
             `Multi-process data loading`_.

.. warning:: See :ref:`reproducibility`, and :ref:`dataloader-workers-random-seed`, and
             :ref:`data-loading-randomness` notes for random seed related questions.

.. warning:: Setting `in_order` to `False` can harm reproducibility and may lead to a skewed data
             distribution being fed to the trainer in cases with imbalanced data.

.. _multiprocessing context:
    https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods

#### Methods

- **`check_worker_number_rationality(self)`**

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

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.

### `WILLOWObjectClass`

The WILLOW-ObjectClass dataset from the `"Learning Graphs to Match"
<https://www.di.ens.fr/willow/pdfscurrent/cho2013.pdf>`_ paper,
containing 10 equal keypoints of at least 40 images in each category.
The keypoints contain interpolated features from a pre-trained VGG16 model
on ImageNet (:obj:`relu4_2` and :obj:`relu5_1`).

Args:
    root (str): Root directory where the dataset should be saved.
    category (str): The category of the images (one of :obj:`"Car"`,
        :obj:`"Duck"`, :obj:`"Face"`, :obj:`"Motorbike"`,
        :obj:`"Winebottle"`).
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
    device (str or torch.device, optional): The device to use for
        processing the raw data. If set to :obj:`None`, will utilize
        GPU-processing if available. (default: :obj:`None`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.
