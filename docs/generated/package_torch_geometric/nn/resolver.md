# resolver

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.resolver`

## Functions (8)

### `activation_resolver(query: Union[Any, str] = 'relu', *args, **kwargs)`

### `aggregation_resolver(query: Union[Any, str], *args, **kwargs)`

### `lr_scheduler_resolver(query: Union[Any, str], optimizer: torch.optim.optimizer.Optimizer, warmup_ratio_or_steps: Union[float, int, NoneType] = 0.1, num_training_steps: Optional[int] = None, **kwargs) -> Union[torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau]`

A resolver to obtain a learning rate scheduler implemented in either
PyG or PyTorch from its name or type.

Args:
    query (Any or str): The query name of the learning rate scheduler.
    optimizer (Optimizer): The optimizer to be scheduled.
    warmup_ratio_or_steps (float or int, optional): The number of warmup
        steps. If given as a `float`, it will act as a ratio that gets
        multiplied with the number of training steps to obtain the number
        of warmup steps. Only required for warmup-based LR schedulers.
        (default: :obj:`0.1`)
    num_training_steps (int, optional): The total number of training steps.
        (default: :obj:`None`)
    **kwargs (optional): Additional arguments of the LR scheduler.

### `normalization_resolver(query: Union[Any, str], *args, **kwargs)`

### `normalize_string(s: str) -> str`

### `optimizer_resolver(query: Union[Any, str], *args, **kwargs)`

### `resolver(classes: List[Any], class_dict: Dict[str, Any], query: Union[Any, str], base_cls: Optional[Any], base_cls_repr: Optional[str], *args: Any, **kwargs: Any) -> Any`

### `swish(x: torch.Tensor) -> torch.Tensor`

## Classes (10)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `ConstantWithWarmupLR`

Creates a LR scheduler with a constant learning rate preceded by a
warmup period during which the learning rate increases linearly between
:obj:`0` and the initial LR set in the optimizer.

Args:
    optimizer (Optimizer): The optimizer to be scheduled.
    num_warmup_steps (int): The number of steps for the warmup phase.
    last_epoch (int, optional): The index of the last epoch when resuming
        training. (default: :obj:`-1`)

### `CosineWithWarmupLR`

Creates a LR scheduler with a learning rate that decreases following
the values of the cosine function between the initial LR set in the
optimizer to :obj:`0`, after a warmup period during which it increases
linearly between :obj:`0` and the initial LR set in the optimizer.

Args:
    optimizer (Optimizer): The optimizer to be scheduled.
    num_warmup_steps (int): The number of steps for the warmup phase.
    num_training_steps (int): The total number of training steps.
    num_cycles (float, optional): The number of waves in the cosine
        schedule (the default decreases LR from the max value to :obj:`0`
        following a half-cosine). (default: :obj:`0.5`)
    last_epoch (int, optional): The index of the last epoch when resuming
        training. (default: :obj:`-1`)

### `CosineWithWarmupRestartsLR`

Creates a LR scheduler with a learning rate that decreases following
the values of the cosine function between the initial LR set in the
optimizer to :obj:`0`, with several hard restarts, after a warmup period
during which it increases linearly between :obj:`0` and the initial LR set
in the optimizer.

Args:
    optimizer (Optimizer): The optimizer to be scheduled.
    num_warmup_steps (int): The number of steps for the warmup phase.
    num_training_steps (int): The total number of training steps.
    num_cycles (int, optional): The number of hard restarts to use.
        (default: :obj:`3`)
    last_epoch (int, optional): The index of the last epoch when resuming
        training. (default: :obj:`-1`)

### `LRScheduler`

Adjusts the learning rate during optimization.

#### Methods

- **`state_dict(self)`**
  Return the state of the scheduler as a :class:`dict`.

- **`load_state_dict(self, state_dict: dict[str, typing.Any])`**
  Load the scheduler's state.

- **`get_last_lr(self) -> list[float]`**
  Return last computed learning rate by current scheduler.

### `LinearWithWarmupLR`

Creates a LR scheduler with a learning rate that decreases linearly
from the initial LR set in the optimizer to :obj:`0`, after a warmup period
during which it increases linearly from :obj:`0` to the initial LR set in
the optimizer.

Args:
    optimizer (Optimizer): The optimizer to be scheduled.
    num_warmup_steps (int): The number of steps for the warmup phase.
    num_training_steps (int): The total number of training steps.
    last_epoch (int, optional): The index of the last epoch when resuming
        training. (default: :obj:`-1`)

### `Optimizer`

Base class for all optimizers.

.. warning::
    Parameters need to be specified as collections that have a deterministic
    ordering that is consistent between runs. Examples of objects that don't
    satisfy those properties are sets and iterators over values of dictionaries.

Args:
    params (iterable): an iterable of :class:`torch.Tensor` s or
        :class:`dict` s. Specifies what Tensors should be optimized.
    defaults: (dict): a dict containing default values of optimization
        options (used when a parameter group doesn't specify them).

#### Methods

- **`OptimizerPreHook(*args, **kwargs)`**

- **`OptimizerPostHook(*args, **kwargs)`**

- **`profile_hook_step(func: Callable[~_P, ~R]) -> Callable[~_P, ~R]`**

### `PolynomialWithWarmupLR`

Creates a LR scheduler with a learning rate that decreases as a
polynomial decay from the initial LR set in the optimizer to end LR defined
by `lr_end`, after a warmup period during which it increases linearly from
:obj:`0` to the initial LR set in the optimizer.

Args:
    optimizer (Optimizer): The optimizer to be scheduled.
    num_warmup_steps (int): The number of steps for the warmup phase.
    num_training_steps (int): The total number of training steps.
    lr_end (float, optional): The end learning rate. (default: :obj:`1e-7`)
    power (float, optional): The power factor of the polynomial decay.
        (default: :obj:`1.0`)
    last_epoch (int, optional): The index of the last epoch when resuming
        training. (default: :obj:`-1`)

### `ReduceLROnPlateau`

Reduce learning rate when a metric has stopped improving.

Models often benefit from reducing the learning rate by a factor
of 2-10 once learning stagnates. This scheduler reads a metrics
quantity and if no improvement is seen for a 'patience' number
of epochs, the learning rate is reduced.

Args:
    optimizer (Optimizer): Wrapped optimizer.
    mode (str): One of `min`, `max`. In `min` mode, lr will
        be reduced when the quantity monitored has stopped
        decreasing; in `max` mode it will be reduced when the
        quantity monitored has stopped increasing. Default: 'min'.
    factor (float): Factor by which the learning rate will be
        reduced. new_lr = lr * factor. Default: 0.1.
    patience (int): The number of allowed epochs with no improvement after
        which the learning rate will be reduced.
        For example, consider the case of having no patience (`patience = 0`).
        In the first epoch, a baseline is established and is always considered good as there's no previous baseline.
        In the second epoch, if the performance is worse than the baseline,
        we have what is considered an intolerable epoch.
        Since the count of intolerable epochs (1) is greater than the patience level (0),
        the learning rate is reduced at the end of this epoch.
        From the third epoch onwards, the learning rate continues to be reduced at the end of each epoch
        if the performance is worse than the baseline. If the performance improves or remains the same,
        the learning rate is not adjusted.
        Default: 10.
    threshold (float): Threshold for measuring the new optimum,
        to only focus on significant changes. Default: 1e-4.
    threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
        dynamic_threshold = best * ( 1 + threshold ) in 'max'
        mode or best * ( 1 - threshold ) in `min` mode.
        In `abs` mode, dynamic_threshold = best + threshold in
        `max` mode or best - threshold in `min` mode. Default: 'rel'.
    cooldown (int): Number of epochs to wait before resuming
        normal operation after lr has been reduced. Default: 0.
    min_lr (float or list): A scalar or a list of scalars. A
        lower bound on the learning rate of all param groups
        or each group respectively. Default: 0.
    eps (float): Minimal decay applied to lr. If the difference
        between new and old lr is smaller than eps, the update is
        ignored. Default: 1e-8.

Example:
    >>> # xdoctest: +SKIP
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
    >>> for epoch in range(10):
    >>>     train(...)
    >>>     val_loss = validate(...)
    >>>     # Note that step should be called after validate()
    >>>     scheduler.step(val_loss)

#### Methods

- **`step(self, metrics: <class 'SupportsFloat'>, epoch=None)`**
  Perform a step.

- **`is_better(self, a, best)`**

- **`state_dict(self)`**
  Return the state of the scheduler as a :class:`dict`.

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
