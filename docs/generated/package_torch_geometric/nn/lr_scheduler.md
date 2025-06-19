# lr_scheduler

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.lr_scheduler`

## Classes (7)

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

### `LambdaLR`

Sets the initial learning rate.

The learning rate of each parameter group is set to the initial lr
times a given function. When last_epoch=-1, sets initial lr as lr.

Args:
    optimizer (Optimizer): Wrapped optimizer.
    lr_lambda (function or list): A function which computes a multiplicative
        factor given an integer parameter epoch, or a list of such
        functions, one for each group in optimizer.param_groups.
    last_epoch (int): The index of last epoch. Default: -1.

Example:
    >>> # xdoctest: +SKIP
    >>> # Assuming optimizer has two groups.
    >>> lambda1 = lambda epoch: epoch // 30
    >>> lambda2 = lambda epoch: 0.95 ** epoch
    >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
    >>> for epoch in range(100):
    >>>     train(...)
    >>>     validate(...)
    >>>     scheduler.step()

#### Methods

- **`state_dict(self)`**
  Return the state of the scheduler as a :class:`dict`.

- **`load_state_dict(self, state_dict)`**
  Load the scheduler's state.

- **`get_lr(self)`**
  Compute learning rate.

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
