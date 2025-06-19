# checkpoint

Part of `torch_geometric.graphgym`
Module: `torch_geometric.graphgym.checkpoint`

## Functions (8)

### `clean_ckpt()`

Removes all but the last model checkpoint.

### `get_ckpt_dir() -> str`

### `get_ckpt_epoch(epoch: int) -> int`

### `get_ckpt_epochs() -> List[int]`

### `get_ckpt_path(epoch: Union[int, str]) -> str`

### `load_ckpt(model: torch.nn.modules.module.Module, optimizer: Optional[torch.optim.optimizer.Optimizer] = None, scheduler: Optional[Any] = None, epoch: int = -1) -> int`

Loads the model checkpoint at a given epoch.

### `remove_ckpt(epoch: int = -1)`

Removes the model checkpoint at a given epoch.

### `save_ckpt(model: torch.nn.modules.module.Module, optimizer: Optional[torch.optim.optimizer.Optimizer] = None, scheduler: Optional[Any] = None, epoch: int = 0)`

Saves the model checkpoint at a given epoch.

## Classes (1)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.
