# imports

Part of `torch_geometric.graphgym`
Module: `torch_geometric.graphgym.imports`

## Classes (2)

### `Callback`

Abstract base class used to build new callbacks.

Subclass this class and override any of the relevant hooks

#### Methods

- **`setup(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', stage: str) -> None`**
  Called when fit, validate, test, predict, or tune begins.

- **`teardown(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', stage: str) -> None`**
  Called when fit, validate, test, predict, or tune ends.

- **`on_fit_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None`**
  Called when fit begins.

### `LightningModule`

Hooks to be used in LightningModule.

#### Methods

- **`optimizers(self, use_pl_optimizer: bool = True) -> Union[torch.optim.optimizer.Optimizer, pytorch_lightning.core.optimizer.LightningOptimizer, lightning_fabric.wrappers._FabricOptimizer, list[torch.optim.optimizer.Optimizer], list[pytorch_lightning.core.optimizer.LightningOptimizer], list[lightning_fabric.wrappers._FabricOptimizer]]`**
  Returns the optimizer(s) that are being used during training. Useful for manual optimization.

- **`lr_schedulers(self) -> Union[NoneType, list[Union[torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau]], torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau]`**
  Returns the learning rate scheduler(s) that are being used during training. Useful for manual optimization.

- **`print(self, *args: Any, **kwargs: Any) -> None`**
  Prints only from process 0. Use this in any distributed mode to log only once.
