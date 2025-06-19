# dynamic_batch_sampler

Part of `torch_geometric.loader`
Module: `torch_geometric.loader.dynamic_batch_sampler`

## Classes (2)

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

### `DynamicBatchSampler`

Dynamically adds samples to a mini-batch up to a maximum size (either
based on number of nodes or number of edges). When data samples have a
wide range in sizes, specifying a mini-batch size in terms of number of
samples is not ideal and can cause CUDA OOM errors.

Within the :class:`DynamicBatchSampler`, the number of steps per epoch is
ambiguous, depending on the order of the samples. By default the
:meth:`__len__` will be undefined. This is fine for most cases but
progress bars will be infinite. Alternatively, :obj:`num_steps` can be
supplied to cap the number of mini-batches produced by the sampler.

.. code-block:: python

    from torch_geometric.loader import DataLoader, DynamicBatchSampler

    sampler = DynamicBatchSampler(dataset, max_num=10000, mode="node")
    loader = DataLoader(dataset, batch_sampler=sampler, ...)

Args:
    dataset (Dataset): Dataset to sample from.
    max_num (int): Size of mini-batch to aim for in number of nodes or
        edges.
    mode (str, optional): :obj:`"node"` or :obj:`"edge"` to measure
        batch size. (default: :obj:`"node"`)
    shuffle (bool, optional): If set to :obj:`True`, will have the data
        reshuffled at every epoch. (default: :obj:`False`)
    skip_too_big (bool, optional): If set to :obj:`True`, skip samples
        which cannot fit in a batch by itself. (default: :obj:`False`)
    num_steps (int, optional): The number of mini-batches to draw for a
        single epoch. If set to :obj:`None`, will iterate through all the
        underlying examples, but :meth:`__len__` will be :obj:`None` since
        it is ambiguous. (default: :obj:`None`)
