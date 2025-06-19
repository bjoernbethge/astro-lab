# lightning

Part of `torch_geometric.data`
Module: `torch_geometric.data.lightning`

## Classes (3)

### `LightningDataset`

Converts a set of :class:`~torch_geometric.data.Dataset` objects into a
:class:`pytorch_lightning.LightningDataModule` variant. It can then be
automatically used as a :obj:`datamodule` for multi-GPU graph-level
training via :lightning:`null`
`PyTorch Lightning <https://www.pytorchlightning.ai>`__.
:class:`LightningDataset` will take care of providing mini-batches via
:class:`~torch_geometric.loader.DataLoader`.

.. note::

    Currently only the
    :class:`pytorch_lightning.strategies.SingleDeviceStrategy` and
    :class:`pytorch_lightning.strategies.DDPStrategy` training
    strategies of :lightning:`null` `PyTorch Lightning
    <https://pytorch-lightning.readthedocs.io/en/latest/guides/
    speed.html>`__ are supported in order to correctly share data across
    all devices/processes:

    .. code-block:: python

        import pytorch_lightning as pl
        trainer = pl.Trainer(strategy="ddp_spawn", accelerator="gpu",
                             devices=4)
        trainer.fit(model, datamodule)

Args:
    train_dataset (Dataset): The training dataset.
    val_dataset (Dataset, optional): The validation dataset.
        (default: :obj:`None`)
    test_dataset (Dataset, optional): The test dataset.
        (default: :obj:`None`)
    pred_dataset (Dataset, optional): The prediction dataset.
        (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.loader.DataLoader`.

#### Methods

- **`dataloader(self, dataset: torch_geometric.data.dataset.Dataset, **kwargs: Any) -> torch_geometric.loader.dataloader.DataLoader`**

- **`train_dataloader(self) -> torch_geometric.loader.dataloader.DataLoader`**
  An iterable or collection of iterables specifying training samples.

- **`val_dataloader(self) -> torch_geometric.loader.dataloader.DataLoader`**
  An iterable or collection of iterables specifying validation samples.

### `LightningLinkData`

Converts a :class:`~torch_geometric.data.Data` or
:class:`~torch_geometric.data.HeteroData` object into a
:class:`pytorch_lightning.LightningDataModule` variant. It can then be
automatically used as a :obj:`datamodule` for multi-GPU link-level
training via :lightning:`null`
`PyTorch Lightning <https://www.pytorchlightning.ai>`__.
:class:`LightningDataset` will take care of providing mini-batches via
:class:`~torch_geometric.loader.LinkNeighborLoader`.

.. note::

    Currently only the
    :class:`pytorch_lightning.strategies.SingleDeviceStrategy` and
    :class:`pytorch_lightning.strategies.DDPStrategy` training
    strategies of :lightning:`null` `PyTorch Lightning
    <https://pytorch-lightning.readthedocs.io/en/latest/guides/
    speed.html>`__ are supported in order to correctly share data across
    all devices/processes:

    .. code-block:: python

        import pytorch_lightning as pl
        trainer = pl.Trainer(strategy="ddp_spawn", accelerator="gpu",
                             devices=4)
        trainer.fit(model, datamodule)

Args:
    data (Data or HeteroData or Tuple[FeatureStore, GraphStore]): The
        :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` graph object, or a
        tuple of a :class:`~torch_geometric.data.FeatureStore` and
        :class:`~torch_geometric.data.GraphStore` objects.
    input_train_edges (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
        The training edges. (default: :obj:`None`)
    input_train_labels (torch.Tensor, optional):
        The labels of training edges. (default: :obj:`None`)
    input_train_time (torch.Tensor, optional): The timestamp
        of training edges. (default: :obj:`None`)
    input_val_edges (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
        The validation edges. (default: :obj:`None`)
    input_val_labels (torch.Tensor, optional):
        The labels of validation edges. (default: :obj:`None`)
    input_val_time (torch.Tensor, optional): The timestamp
        of validation edges. (default: :obj:`None`)
    input_test_edges (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
        The test edges. (default: :obj:`None`)
    input_test_labels (torch.Tensor, optional):
        The labels of test edges. (default: :obj:`None`)
    input_test_time (torch.Tensor, optional): The timestamp
        of test edges. (default: :obj:`None`)
    input_pred_edges (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
        The prediction edges. (default: :obj:`None`)
    input_pred_labels (torch.Tensor, optional):
        The labels of prediction edges. (default: :obj:`None`)
    input_pred_time (torch.Tensor, optional): The timestamp
        of prediction edges. (default: :obj:`None`)
    loader (str): The scalability technique to use (:obj:`"full"`,
        :obj:`"neighbor"`). (default: :obj:`"neighbor"`)
    link_sampler (BaseSampler, optional): A custom sampler object to
        generate mini-batches. If set, will ignore the :obj:`loader`
        option. (default: :obj:`None`)
    eval_loader_kwargs (Dict[str, Any], optional): Custom keyword arguments
        that override the
        :class:`torch_geometric.loader.LinkNeighborLoader` configuration
        during evaluation. (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.loader.LinkNeighborLoader`.

#### Methods

- **`dataloader(self, input_edges: Union[torch.Tensor, NoneType, Tuple[str, str, str], Tuple[Tuple[str, str, str], Optional[torch.Tensor]]], input_labels: Optional[torch.Tensor] = None, input_time: Optional[torch.Tensor] = None, input_id: Optional[torch.Tensor] = None, link_sampler: Optional[torch_geometric.sampler.base.BaseSampler] = None, **kwargs: Any) -> torch.utils.data.dataloader.DataLoader`**

- **`train_dataloader(self) -> torch.utils.data.dataloader.DataLoader`**
  An iterable or collection of iterables specifying training samples.

- **`val_dataloader(self) -> torch.utils.data.dataloader.DataLoader`**
  An iterable or collection of iterables specifying validation samples.

### `LightningNodeData`

Converts a :class:`~torch_geometric.data.Data` or
:class:`~torch_geometric.data.HeteroData` object into a
:class:`pytorch_lightning.LightningDataModule` variant. It can then be
automatically used as a :obj:`datamodule` for multi-GPU node-level
training via :lightning:`null`
`PyTorch Lightning <https://www.pytorchlightning.ai>`__.
:class:`LightningDataset` will take care of providing mini-batches via
:class:`~torch_geometric.loader.NeighborLoader`.

.. note::

    Currently only the
    :class:`pytorch_lightning.strategies.SingleDeviceStrategy` and
    :class:`pytorch_lightning.strategies.DDPStrategy` training
    strategies of :lightning:`null` `PyTorch Lightning
    <https://pytorch-lightning.readthedocs.io/en/latest/guides/
    speed.html>`__ are supported in order to correctly share data across
    all devices/processes:

    .. code-block:: python

        import pytorch_lightning as pl
        trainer = pl.Trainer(strategy="ddp_spawn", accelerator="gpu",
                             devices=4)
        trainer.fit(model, datamodule)

Args:
    data (Data or HeteroData): The :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` graph object.
    input_train_nodes (torch.Tensor or str or (str, torch.Tensor)): The
        indices of training nodes.
        If not given, will try to automatically infer them from the
        :obj:`data` object by searching for :obj:`train_mask`,
        :obj:`train_idx`, or :obj:`train_index` attributes.
        (default: :obj:`None`)
    input_train_time (torch.Tensor, optional): The timestamp
        of training nodes. (default: :obj:`None`)
    input_val_nodes (torch.Tensor or str or (str, torch.Tensor)): The
        indices of validation nodes.
        If not given, will try to automatically infer them from the
        :obj:`data` object by searching for :obj:`val_mask`,
        :obj:`valid_mask`, :obj:`val_idx`, :obj:`valid_idx`,
        :obj:`val_index`, or :obj:`valid_index` attributes.
        (default: :obj:`None`)
    input_val_time (torch.Tensor, optional): The timestamp
        of validation edges. (default: :obj:`None`)
    input_test_nodes (torch.Tensor or str or (str, torch.Tensor)): The
        indices of test nodes.
        If not given, will try to automatically infer them from the
        :obj:`data` object by searching for :obj:`test_mask`,
        :obj:`test_idx`, or :obj:`test_index` attributes.
        (default: :obj:`None`)
    input_test_time (torch.Tensor, optional): The timestamp
        of test nodes. (default: :obj:`None`)
    input_pred_nodes (torch.Tensor or str or (str, torch.Tensor)): The
        indices of prediction nodes.
        If not given, will try to automatically infer them from the
        :obj:`data` object by searching for :obj:`pred_mask`,
        :obj:`pred_idx`, or :obj:`pred_index` attributes.
        (default: :obj:`None`)
    input_pred_time (torch.Tensor, optional): The timestamp
        of prediction nodes. (default: :obj:`None`)
    loader (str): The scalability technique to use (:obj:`"full"`,
        :obj:`"neighbor"`). (default: :obj:`"neighbor"`)
    node_sampler (BaseSampler, optional): A custom sampler object to
        generate mini-batches. If set, will ignore the :obj:`loader`
        option. (default: :obj:`None`)
    eval_loader_kwargs (Dict[str, Any], optional): Custom keyword arguments
        that override the :class:`torch_geometric.loader.NeighborLoader`
        configuration during evaluation. (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.loader.NeighborLoader`.

#### Methods

- **`dataloader(self, input_nodes: Union[torch.Tensor, NoneType, str, Tuple[str, Optional[torch.Tensor]]], input_time: Optional[torch.Tensor] = None, input_id: Optional[torch.Tensor] = None, node_sampler: Optional[torch_geometric.sampler.base.BaseSampler] = None, **kwargs: Any) -> torch.utils.data.dataloader.DataLoader`**

- **`train_dataloader(self) -> torch.utils.data.dataloader.DataLoader`**
  An iterable or collection of iterables specifying training samples.

- **`val_dataloader(self) -> torch.utils.data.dataloader.DataLoader`**
  An iterable or collection of iterables specifying validation samples.
