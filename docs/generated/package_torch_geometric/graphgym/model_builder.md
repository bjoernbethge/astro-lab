# model_builder

Part of `torch_geometric.graphgym`
Module: `torch_geometric.graphgym.model_builder`

## Functions (5)

### `compute_loss(pred, true)`

Compute loss and prediction score.

Args:
    pred (torch.tensor): Unnormalized prediction
    true (torch.tensor): Grou

Returns: Loss, normalized prediction score

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

### `register_network(key: str, module: Any = None)`

Registers a GNN model in GraphGym.

## Classes (4)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

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

### `GraphGymModule`

Hooks to be used in LightningModule.

#### Methods

- **`forward(self, *args, **kwargs)`**
  Same as :meth:`torch.nn.Module.forward`.

- **`configure_optimizers(self) -> Tuple[Any, Any]`**
  Choose what optimizers and learning-rate schedulers to use in your optimization. Normally you'd need one.

- **`training_step(self, batch, *args, **kwargs)`**
  Here you compute and return the training loss and some additional metrics for e.g. the progress bar or

### `LightningModule`

Hooks to be used in LightningModule.

#### Methods

- **`optimizers(self, use_pl_optimizer: bool = True) -> Union[torch.optim.optimizer.Optimizer, pytorch_lightning.core.optimizer.LightningOptimizer, lightning_fabric.wrappers._FabricOptimizer, list[torch.optim.optimizer.Optimizer], list[pytorch_lightning.core.optimizer.LightningOptimizer], list[lightning_fabric.wrappers._FabricOptimizer]]`**
  Returns the optimizer(s) that are being used during training. Useful for manual optimization.

- **`lr_schedulers(self) -> Union[NoneType, list[Union[torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau]], torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau]`**
  Returns the learning rate scheduler(s) that are being used during training. Useful for manual optimization.

- **`print(self, *args: Any, **kwargs: Any) -> None`**
  Prints only from process 0. Use this in any distributed mode to log only once.
