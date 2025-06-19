# logger

Part of `torch_geometric.graphgym`
Module: `torch_geometric.graphgym.logger`

## Functions (6)

### `create_logger()`

Create logger for the experiment.

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

### `get_current_gpu_usage()`

Get the current GPU memory usage.

### `infer_task()`

### `set_printing()`

Set up printing options.

## Classes (4)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

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

### `Logger`

#### Methods

- **`reset(self)`**

- **`basic(self)`**

- **`custom(self)`**

### `LoggerCallback`

Abstract base class used to build new callbacks.

Subclass this class and override any of the relevant hooks

#### Methods

- **`close(self)`**

- **`on_train_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule')`**
  Called when the train epoch begins.

- **`on_validation_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule')`**
  Called when the val epoch begins.
