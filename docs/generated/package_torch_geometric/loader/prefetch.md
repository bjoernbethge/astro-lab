# prefetch

Part of `torch_geometric.loader`
Module: `torch_geometric.loader.prefetch`

## Classes (6)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

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

### `DeviceHelper`

#### Methods

- **`maybe_init_stream(self) -> None`**

- **`maybe_wait_stream(self) -> None`**

### `PrefetchLoader`

A GPU prefetcher class for asynchronously transferring data of a
:class:`torch.utils.data.DataLoader` from host memory to device memory.

Args:
    loader (torch.utils.data.DataLoader): The data loader.
    device (torch.device, optional): The device to load the data to.
        (default: :obj:`None`)

#### Methods

- **`non_blocking_transfer(self, batch: Any) -> Any`**

### `nullcontext`

Context manager that does no additional processing.

Used as a stand-in for a normal context manager, when a particular
block of code is only sometimes used with a normal context manager:

cm = optional_cm if condition else nullcontext()
with cm:
    # Perform operation, using optional_cm if condition is True

### `partial`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.
