# graphgym Submodule

Part of the `torch_geometric` package
Module: `torch_geometric.graphgym`

## Functions (54)

### `GNNLayer(dim_in: int, dim_out: int, has_act: bool = True) -> torch_geometric.graphgym.models.layer.GeneralLayer`

Creates a GNN layer, given the specified input and output dimensions
and the underlying configuration in :obj:`cfg`.

Args:
    dim_in (int): The input dimension
    dim_out (int): The output dimension.
    has_act (bool, optional): Whether to apply an activation function
        after the layer. (default: :obj:`True`)

### `GNNPreMP(dim_in: int, dim_out: int, num_layers: int) -> torch_geometric.graphgym.models.layer.GeneralMultiLayer`

Creates a NN layer used before message passing, given the specified
input and output dimensions and the underlying configuration in :obj:`cfg`.

Args:
    dim_in (int): The input dimension
    dim_out (int): The output dimension.
    num_layers (int): The number of layers.

### `agg_batch(dir, metric_best='auto')`

Aggregate across results from multiple experiments via grid search.

Args:
    dir (str): Directory of the results, containing multiple experiments
    metric_best (str, optional): The metric for selecting the best
    validation performance. Options: auto, accuracy, auc.

### `agg_runs(dir, metric_best='auto')`

Aggregate over different random seeds of a single experiment.

Args:
    dir (str): Directory of the results, containing 1 experiment
    metric_best (str, optional): The metric for selecting the best
    validation performance. Options: auto, accuracy, auc.

### `auto_select_device()`

Auto select device for the current experiment.

### `clean_ckpt()`

Removes all but the last model checkpoint.

### `compute_loss(pred, true)`

Compute loss and prediction score.

Args:
    pred (torch.tensor): Unnormalized prediction
    true (torch.tensor): Grou

Returns: Loss, normalized prediction score

### `create_loader()`

Create data loader object.

Returns: List of PyTorch data loaders

### `create_logger()`

Create logger for the experiment.

### `create_model(to_device=True, dim_in=None, dim_out=None) -> torch_geometric.graphgym.model_builder.GraphGymModule`

Create model for graph machine learning.

Args:
    to_device (bool, optional): Whether to transfer the model to the
        specified device. (default: :obj:`True`)
    dim_in (int, optional): Input dimension to the model
    dim_out (int, optional): Output dimension to the model

### `create_optimizer(params: Iterator[torch.nn.parameter.Parameter], cfg: Any) -> Any`

Creates a config-driven optimizer.

### `create_scheduler(optimizer: torch.optim.optimizer.Optimizer, cfg: Any) -> Any`

Creates a config-driven learning rate scheduler.

### `dict_list_to_json(dict_list, fname)`

Dump a list of :python:`Python` dictionaries to a JSON file.

Args:
    dict_list (list of dict): List of :python:`Python` dictionaries.
    fname (str): the output file name.

### `dict_to_json(dict, fname)`

Dump a :python:`Python` dictionary to a JSON file.

Args:
    dict (dict): The :python:`Python` dictionary.
    fname (str): The output file name.

### `dict_to_tb(dict, writer, epoch)`

Add a dictionary of statistics to a Tensorboard writer.

Args:
    dict (dict): Statistics of experiments, the keys are attribute names,
    the values are the attribute values
    writer: Tensorboard writer object
    epoch (int): The current epoch

### `dump_cfg(cfg)`

Dumps the config to the output directory specified in
:obj:`cfg.out_dir`.

Args:
    cfg (CfgNode): Configuration node

### `get_current_gpu_usage()`

Get the current GPU memory usage.

### `get_fname(fname)`

Extract filename from file name path.

Args:
    fname (str): Filename for the yaml format configuration file

### `global_add_pool(x: torch.Tensor, batch: Optional[torch.Tensor], size: Optional[int] = None) -> torch.Tensor`

Returns batch-wise graph-level-outputs by adding node features
across the node dimension.

For a single graph :math:`\mathcal{G}_i`, its output is computed by

.. math::
    \mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n.

Functional method of the
:class:`~torch_geometric.nn.aggr.SumAggregation` module.

Args:
    x (torch.Tensor): Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
    batch (torch.Tensor, optional): The batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
        each node to a specific example.
    size (int, optional): The number of examples :math:`B`.
        Automatically calculated if not given. (default: :obj:`None`)

### `global_max_pool(x: torch.Tensor, batch: Optional[torch.Tensor], size: Optional[int] = None) -> torch.Tensor`

Returns batch-wise graph-level-outputs by taking the channel-wise
maximum across the node dimension.

For a single graph :math:`\mathcal{G}_i`, its output is computed by

.. math::
    \mathbf{r}_i = \mathrm{max}_{n=1}^{N_i} \, \mathbf{x}_n.

Functional method of the
:class:`~torch_geometric.nn.aggr.MaxAggregation` module.

Args:
    x (torch.Tensor): Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
    batch (torch.Tensor, optional): The batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
        each element to a specific example.
    size (int, optional): The number of examples :math:`B`.
        Automatically calculated if not given. (default: :obj:`None`)

### `global_mean_pool(x: torch.Tensor, batch: Optional[torch.Tensor], size: Optional[int] = None) -> torch.Tensor`

Returns batch-wise graph-level-outputs by averaging node features
across the node dimension.

For a single graph :math:`\mathcal{G}_i`, its output is computed by

.. math::
    \mathbf{r}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n.

Functional method of the
:class:`~torch_geometric.nn.aggr.MeanAggregation` module.

Args:
    x (torch.Tensor): Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
    batch (torch.Tensor, optional): The batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
        each node to a specific example.
    size (int, optional): The number of examples :math:`B`.
        Automatically calculated if not given. (default: :obj:`None`)

### `init_weights(m)`

Performs weight initialization.

Args:
    m (nn.Module): PyTorch module

### `is_ckpt_epoch(cur_epoch)`

Determines if the model should be evaluated at the current epoch.

### `is_eval_epoch(cur_epoch)`

Determines if the model should be evaluated at the current epoch.

### `load_cfg(cfg, args)`

Load configurations from file system and command line.

Args:
    cfg (CfgNode): Configuration node
    args (ArgumentParser): Command argument parser

### `load_ckpt(model: torch.nn.modules.module.Module, optimizer: Optional[torch.optim.optimizer.Optimizer] = None, scheduler: Optional[Any] = None, epoch: int = -1) -> int`

Loads the model checkpoint at a given epoch.

### `makedirs_rm_exist(dir)`

Make a directory, remove any existing data.

Args:
    dir (str): The directory to be created.

### `match_baseline_cfg(cfg_dict, cfg_dict_baseline, verbose=True)`

Match the computational budget of a given baseline model. The current
configuration dictionary will be modifed and returned.

Args:
    cfg_dict (dict): Current experiment's configuration
    cfg_dict_baseline (dict): Baseline configuration
    verbose (str, optional): If printing matched paramter conunts

### `params_count(model)`

Computes the number of parameters.

Args:
    model (nn.Module): PyTorch model

### `parse_args() -> argparse.Namespace`

Parses the command line arguments.

### `register_act(key: str, module: Any = None)`

Registers an activation function in GraphGym.

### `register_base(mapping: Dict[str, Any], key: str, module: Any = None) -> Optional[Callable]`

Base function for registering a module in GraphGym.

Args:
    mapping (dict): :python:`Python` dictionary to register the module.
        hosting all the registered modules
    key (str): The name of the module.
    module (any, optional): The module. If set to :obj:`None`, will return
        a decorator to register a module.

### `register_config(key: str, module: Any = None)`

Registers a configuration group in GraphGym.

### `register_dataset(key: str, module: Any = None)`

Registers a dataset in GraphGym.

### `register_edge_encoder(key: str, module: Any = None)`

Registers an edge feature encoder in GraphGym.

### `register_head(key: str, module: Any = None)`

Registers a GNN prediction head in GraphGym.

### `register_layer(key: str, module: Any = None)`

Registers a GNN layer in GraphGym.

### `register_loader(key: str, module: Any = None)`

Registers a data loader in GraphGym.

### `register_loss(key: str, module: Any = None)`

Registers a loss function in GraphGym.

### `register_metric(key: str, module: Any = None)`

Register a metric function in GraphGym.

### `register_network(key: str, module: Any = None)`

Registers a GNN model in GraphGym.

### `register_node_encoder(key: str, module: Any = None)`

Registers a node feature encoder in GraphGym.

### `register_optimizer(key: str, module: Any = None)`

Registers an optimizer in GraphGym.

### `register_pooling(key: str, module: Any = None)`

Registers a GNN global pooling/readout layer in GraphGym.

### `register_scheduler(key: str, module: Any = None)`

Registers a learning rate scheduler in GraphGym.

### `register_stage(key: str, module: Any = None)`

Registers a customized GNN stage in GraphGym.

### `register_train(key: str, module: Any = None)`

Registers a training function in GraphGym.

### `remove_ckpt(epoch: int = -1)`

Removes the model checkpoint at a given epoch.

### `save_ckpt(model: torch.nn.modules.module.Module, optimizer: Optional[torch.optim.optimizer.Optimizer] = None, scheduler: Optional[Any] = None, epoch: int = 0)`

Saves the model checkpoint at a given epoch.

### `set_cfg(cfg)`

This function sets the default config value.

1) Note that for an experiment, only part of the arguments will be used
   The remaining unused arguments won't affect anything.
   So feel free to register any argument in graphgym.contrib.config
2) We support *at most* two levels of configs, *e.g.*,
   :obj:`cfg.dataset.name`.

:return: Configuration use by the experiment.

### `set_out_dir(out_dir, fname)`

Create the directory for full experiment run.

Args:
    out_dir (str): Directory for output, specified in :obj:`cfg.out_dir`
    fname (str): Filename for the yaml format configuration file

### `set_printing()`

Set up printing options.

### `set_run_dir(out_dir)`

Create the directory for each random seed experiment run.

Args:
    out_dir (str): Directory for output, specified in :obj:`cfg.out_dir`

### `train(model: torch_geometric.graphgym.model_builder.GraphGymModule, datamodule: torch_geometric.graphgym.train.GraphGymDataModule, logger: bool = True, trainer_config: Optional[Dict[str, Any]] = None)`

Trains a GraphGym model using PyTorch Lightning.

Args:
    model (GraphGymModule): The GraphGym model.
    datamodule (GraphGymDataModule): The GraphGym data module.
    logger (bool, optional): Whether to enable logging during training.
        (default: :obj:`True`)
    trainer_config (dict, optional): Additional trainer configuration.

## Important Data Types (15)

### `GNN`
**Type**: `<class 'type'>`

A general Graph Neural Network (GNN) model.

The GNN model consists of three main components:

1. An encoder to transform input features into a fixed-size embedding
   space.
2. A processing or message passing stage for information exchange between
   nodes.
3. A head to produce the final output features/predictions.

The configuration of each component is determined by the underlying
configuration in :obj:`cfg`.

Args:
    dim_in (int): The input feature dimension.
    dim_out (int): The output feature dimension.
    **kwargs (optional): Additional keyword arguments.

*(has methods, callable)*

### `MLP`
**Type**: `<class 'type'>`

A basic MLP model.

Args:
    layer_config (LayerConfig): The configuration of the layer.
    **kwargs (optional): Additional keyword arguments.

*(has methods, callable)*

### `Linear`
**Type**: `<class 'type'>`

A basic Linear layer.

Args:
    layer_config (LayerConfig): The configuration of the layer.
    **kwargs (optional): Additional keyword arguments.

*(has methods, callable)*

### `GATConv`
**Type**: `<class 'type'>`

A Graph Attention Network (GAT) layer.

*(has methods, callable)*

### `GCNConv`
**Type**: `<class 'type'>`

A Graph Convolutional Network (GCN) layer.

*(has methods, callable)*

### `GINConv`
**Type**: `<class 'type'>`

A Graph Isomorphism Network (GIN) layer.

*(has methods, callable)*

### `SAGEConv`
**Type**: `<class 'type'>`

A GraphSAGE layer.

*(has methods, callable)*

### `SplineConv`
**Type**: `<class 'type'>`

A SplineCNN layer.

*(has methods, callable)*

### `AtomEncoder`
**Type**: `<class 'type'>`

The atom encoder used in OGB molecule dataset.

Args:
    emb_dim (int): The output embedding dimension.

Example:
    >>> encoder = AtomEncoder(emb_dim=16)
    >>> batch = torch.randint(0, 10, (10, 3))
    >>> encoder(batch).size()
    torch.Size([10, 16])

*(has methods, callable)*

### `BondEncoder`
**Type**: `<class 'type'>`

The bond encoder used in OGB molecule dataset.

Args:
    emb_dim (int): The output embedding dimension.

Example:
    >>> encoder = BondEncoder(emb_dim=16)
    >>> batch = torch.randint(0, 10, (10, 3))
    >>> encoder(batch).size()
    torch.Size([10, 16])

*(has methods, callable)*

### `GNNEdgeHead`
**Type**: `<class 'type'>`

A GNN prediction head for edge-level/link-level prediction tasks.

Args:
    dim_in (int): The input feature dimension.
    dim_out (int): The output feature dimension.

*(has methods, callable)*

### `GNNNodeHead`
**Type**: `<class 'type'>`

A GNN prediction head for node-level prediction tasks.

Args:
    dim_in (int): The input feature dimension.
    dim_out (int): The output feature dimension.

*(has methods, callable)*

### `GeneralConv`
**Type**: `<class 'type'>`

A general GNN layer.

*(has methods, callable)*

### `GNNGraphHead`
**Type**: `<class 'type'>`

A GNN prediction head for graph-level prediction tasks.
A post message passing layer (as specified by :obj:`cfg.gnn.post_mp`) is
used to transform the pooled graph-level embeddings using an MLP.

Args:
    dim_in (int): The input feature dimension.
    dim_out (int): The output feature dimension.

*(has methods, callable)*

### `GeneralLayer`
**Type**: `<class 'type'>`

A general wrapper for layers.

Args:
    name (str): The registered name of the layer.
    layer_config (LayerConfig): The configuration of the layer.
    **kwargs (optional): Additional keyword arguments.

*(has methods, callable)*

## Classes (24)

### `AtomEncoder`

The atom encoder used in OGB molecule dataset.

Args:
    emb_dim (int): The output embedding dimension.

Example:
    >>> encoder = AtomEncoder(emb_dim=16)
    >>> batch = torch.randint(0, 10, (10, 3))
    >>> encoder(batch).size()
    torch.Size([10, 16])

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `BatchNorm1dEdge`

A batch normalization layer for edge-level features.

Args:
    layer_config (LayerConfig): The configuration of the layer.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `BatchNorm1dNode`

A batch normalization layer for node-level features.

Args:
    layer_config (LayerConfig): The configuration of the layer.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `BondEncoder`

The bond encoder used in OGB molecule dataset.

Args:
    emb_dim (int): The output embedding dimension.

Example:
    >>> encoder = BondEncoder(emb_dim=16)
    >>> batch = torch.randint(0, 10, (10, 3))
    >>> encoder(batch).size()
    torch.Size([10, 16])

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `FeatureEncoder`

Encodes node and edge features, given the specified input dimension and
the underlying configuration in :obj:`cfg`.

Args:
    dim_in (int): The input feature dimension.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GATConv`

A Graph Attention Network (GAT) layer.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GCNConv`

A Graph Convolutional Network (GCN) layer.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GINConv`

A Graph Isomorphism Network (GIN) layer.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GNN`

A general Graph Neural Network (GNN) model.

The GNN model consists of three main components:

1. An encoder to transform input features into a fixed-size embedding
   space.
2. A processing or message passing stage for information exchange between
   nodes.
3. A head to produce the final output features/predictions.

The configuration of each component is determined by the underlying
configuration in :obj:`cfg`.

Args:
    dim_in (int): The input feature dimension.
    dim_out (int): The output feature dimension.
    **kwargs (optional): Additional keyword arguments.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GNNEdgeHead`

A GNN prediction head for edge-level/link-level prediction tasks.

Args:
    dim_in (int): The input feature dimension.
    dim_out (int): The output feature dimension.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GNNGraphHead`

A GNN prediction head for graph-level prediction tasks.
A post message passing layer (as specified by :obj:`cfg.gnn.post_mp`) is
used to transform the pooled graph-level embeddings using an MLP.

Args:
    dim_in (int): The input feature dimension.
    dim_out (int): The output feature dimension.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GNNNodeHead`

A GNN prediction head for node-level prediction tasks.

Args:
    dim_in (int): The input feature dimension.
    dim_out (int): The output feature dimension.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GNNStackStage`

Stacks a number of GNN layers.

Args:
    dim_in (int): The input dimension
    dim_out (int): The output dimension.
    num_layers (int): The number of layers.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GeneralConv`

A general GNN layer.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GeneralEdgeConv`

A general GNN layer with edge feature support.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GeneralLayer`

A general wrapper for layers.

Args:
    name (str): The registered name of the layer.
    layer_config (LayerConfig): The configuration of the layer.
    **kwargs (optional): Additional keyword arguments.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GeneralMultiLayer`

A general wrapper class for a stacking multiple NN layers.

Args:
    name (str): The registered name of the layer.
    layer_config (LayerConfig): The configuration of the layer.
    **kwargs (optional): Additional keyword arguments.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GeneralSampleEdgeConv`

A general GNN layer that supports edge features and edge sampling.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `IntegerFeatureEncoder`

Provides an encoder for integer node features.

Args:
    emb_dim (int): The output embedding dimension.
    num_classes (int): The number of classes/integers.

Example:
    >>> encoder = IntegerFeatureEncoder(emb_dim=16, num_classes=10)
    >>> batch = torch.randint(0, 10, (10, 2))
    >>> encoder(batch).size()
    torch.Size([10, 16])

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `Linear`

A basic Linear layer.

Args:
    layer_config (LayerConfig): The configuration of the layer.
    **kwargs (optional): Additional keyword arguments.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `MLP`

A basic MLP model.

Args:
    layer_config (LayerConfig): The configuration of the layer.
    **kwargs (optional): Additional keyword arguments.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `SAGEConv`

A GraphSAGE layer.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `SplineConv`

A SplineCNN layer.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `dummy_context`

Default context manager that does nothing.

## Nested Submodules (23)

Each nested submodule is documented in a separate file:

### [act](./graphgym/act.md)
Module: `torch_geometric.graphgym.contrib.act`

*Contains: 4 functions*

### [checkpoint](./graphgym/checkpoint.md)
Module: `torch_geometric.graphgym.checkpoint`

*Contains: 8 functions, 1 classes*

### [cmd_args](./graphgym/cmd_args.md)
Module: `torch_geometric.graphgym.cmd_args`

*Contains: 1 functions*

### [config](./graphgym/config.md)
Module: `torch_geometric.graphgym.contrib.config`

*Contains: 4 functions*

### [encoder](./graphgym/encoder.md)
Module: `torch_geometric.graphgym.contrib.encoder`

*Contains: 4 functions*

### [generalconv](./graphgym/generalconv.md)
Module: `torch_geometric.graphgym.contrib.layer.generalconv`

*Contains: 4 functions, 4 classes*

### [head](./graphgym/head.md)
Module: `torch_geometric.graphgym.contrib.head`

*Contains: 4 functions*

### [imports](./graphgym/imports.md)
Module: `torch_geometric.graphgym.imports`

*Contains: 2 classes*

### [init](./graphgym/init.md)
Module: `torch_geometric.graphgym.init`

*Contains: 1 functions*

### [layer](./graphgym/layer.md)
Module: `torch_geometric.graphgym.contrib.layer`

*Contains: 4 functions*

### [loader](./graphgym/loader.md)
Module: `torch_geometric.graphgym.loader`

*Contains: 15 functions, 15 classes*

### [logger](./graphgym/logger.md)
Module: `torch_geometric.graphgym.logger`

*Contains: 6 functions, 4 classes*

### [loss](./graphgym/loss.md)
Module: `torch_geometric.graphgym.loss`

*Contains: 1 functions*

### [model_builder](./graphgym/model_builder.md)
Module: `torch_geometric.graphgym.model_builder`

*Contains: 5 functions, 4 classes*

### [models](./graphgym/models.md)
Module: `torch_geometric.graphgym.models`

*Contains: 5 functions, 23 classes*

### [network](./graphgym/network.md)
Module: `torch_geometric.graphgym.contrib.network`

*Contains: 4 functions*

### [optim](./graphgym/optim.md)
Module: `torch_geometric.graphgym.optim`

*Contains: 10 functions, 10 classes*

### [optimizer](./graphgym/optimizer.md)
Module: `torch_geometric.graphgym.contrib.optimizer`

*Contains: 4 functions*

### [pooling](./graphgym/pooling.md)
Module: `torch_geometric.graphgym.contrib.pooling`

*Contains: 4 functions*

### [register](./graphgym/register.md)
Module: `torch_geometric.graphgym.register`

*Contains: 17 functions, 1 classes*

### [stage](./graphgym/stage.md)
Module: `torch_geometric.graphgym.contrib.stage`

*Contains: 4 functions*

### [transform](./graphgym/transform.md)
Module: `torch_geometric.graphgym.contrib.transform`

*Contains: 4 functions*

### [utils](./graphgym/utils.md)
Module: `torch_geometric.graphgym.utils`

*Contains: 12 functions, 1 classes*
