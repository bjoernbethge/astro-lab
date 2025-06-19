# optim

Part of `torch_geometric.graphgym`
Module: `torch_geometric.graphgym.optim`

## Functions (10)

### `adam_optimizer(params: Iterator[torch.nn.parameter.Parameter], base_lr: float, weight_decay: float) -> torch.optim.adam.Adam`

### `cos_scheduler(optimizer: torch.optim.optimizer.Optimizer, max_epoch: int) -> torch.optim.lr_scheduler.CosineAnnealingLR`

### `create_optimizer(params: Iterator[torch.nn.parameter.Parameter], cfg: Any) -> Any`

Creates a config-driven optimizer.

### `create_scheduler(optimizer: torch.optim.optimizer.Optimizer, cfg: Any) -> Any`

Creates a config-driven learning rate scheduler.

### `dataclass(cls=None, /, *, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False, weakref_slot=False)`

Add dunder methods based on the fields defined in the class.

Examines PEP 526 __annotations__ to determine fields.

If init is true, an __init__() method is added to the class. If repr
is true, a __repr__() method is added. If order is true, rich
comparison dunder methods are added. If unsafe_hash is true, a
__hash__() method is added. If frozen is true, fields may not be
assigned to after instance creation. If match_args is true, the
__match_args__ tuple is added. If kw_only is true, then by default
all fields are keyword-only. If slots is true, a new class with a
__slots__ attribute is returned.

### `field(*, default=<dataclasses._MISSING_TYPE object at 0x000001FE19EEB990>, default_factory=<dataclasses._MISSING_TYPE object at 0x000001FE19EEB990>, init=True, repr=True, hash=None, compare=True, metadata=None, kw_only=<dataclasses._MISSING_TYPE object at 0x000001FE19EEB990>)`

Return an object to identify dataclass fields.

default is the default value of the field.  default_factory is a
0-argument function called to initialize a field's value.  If init
is true, the field will be a parameter to the class's __init__()
function.  If repr is true, the field will be included in the
object's repr().  If hash is true, the field will be included in the
object's hash().  If compare is true, the field will be used in
comparison functions.  metadata, if specified, must be a mapping
which is stored but not otherwise examined by dataclass.  If kw_only
is true, the field will become a keyword-only parameter to
__init__().

It is an error to specify both default and default_factory.

### `from_config(func)`

### `none_scheduler(optimizer: torch.optim.optimizer.Optimizer, max_epoch: int) -> torch.optim.lr_scheduler.StepLR`

### `sgd_optimizer(params: Iterator[torch.nn.parameter.Parameter], base_lr: float, momentum: float, weight_decay: float) -> torch.optim.sgd.SGD`

### `step_scheduler(optimizer: torch.optim.optimizer.Optimizer, steps: List[int], lr_decay: float) -> torch.optim.lr_scheduler.MultiStepLR`

## Classes (10)

### `Adam`

Implements Adam algorithm.

.. math::
   \begin{aligned}
        &\rule{110mm}{0.4pt}                                                                 \\
        &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
            \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)}          \\
        &\hspace{13mm}      \lambda \text{ (weight decay)},  \: \textit{amsgrad},
            \:\textit{maximize},  \: \epsilon \text{ (epsilon)}                              \\
        &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
            v_0\leftarrow 0 \text{ (second moment)},\: v_0^{max}\leftarrow 0          \\[-1.ex]
        &\rule{110mm}{0.4pt}                                                                 \\
        &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

        &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
        &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
        &\hspace{5mm}\textbf{else}                                                           \\
        &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
        &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
        &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
        &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
        &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
        &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
        &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
        &\hspace{10mm} v_t^{max} \leftarrow \mathrm{max}(v_{t-1}^{max},v_t)                  \\
        &\hspace{10mm}\widehat{v_t} \leftarrow v_t^{max}/\big(1-\beta_2^t \big)              \\
        &\hspace{5mm}\textbf{else}                                                           \\
        &\hspace{10mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                  \\
        &\hspace{5mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
            \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
        &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
        &\bf{return} \:  \theta_t                                                     \\[-1.ex]
        &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
   \end{aligned}

For further details regarding the algorithm we refer to `Adam: A Method for Stochastic Optimization`_.

Args:
    params (iterable): iterable of parameters or named_parameters to optimize
        or iterable of dicts defining parameter groups. When using named_parameters,
        all parameters in all groups should be named
    lr (float, Tensor, optional): learning rate (default: 1e-3). A tensor LR
        is not yet supported for all our implementations. Please use a float
        LR if you are not also specifying fused=True or capturable=True.
    betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps (float, optional): term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    decoupled_weight_decay (bool, optional): if True, this optimizer is
        equivalent to AdamW and the algorithm will not accumulate weight
        decay in the momentum nor variance. (default: False)
    amsgrad (bool, optional): whether to use the AMSGrad variant of this
        algorithm from the paper `On the Convergence of Adam and Beyond`_
        (default: False)
    foreach (bool, optional): whether foreach implementation of optimizer
        is used. If unspecified by the user (so foreach is None), we will try to use
        foreach over the for-loop implementation on CUDA, since it is usually
        significantly more performant. Note that the foreach implementation uses
        ~ sizeof(params) more peak memory than the for-loop version due to the intermediates
        being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
        parameters through the optimizer at a time or switch this flag to False (default: None)
    maximize (bool, optional): maximize the objective with respect to the
        params, instead of minimizing (default: False)
    capturable (bool, optional): whether this instance is safe to
        capture in a CUDA graph. Passing True can impair ungraphed performance,
        so if you don't intend to graph capture this instance, leave it False
        (default: False)
    differentiable (bool, optional): whether autograd should
        occur through the optimizer step in training. Otherwise, the step()
        function runs in a torch.no_grad() context. Setting to True can impair
        performance, so leave it False if you don't intend to run autograd
        through this instance (default: False)
    fused (bool, optional): whether the fused implementation is used.
        Currently, `torch.float64`, `torch.float32`, `torch.float16`, and `torch.bfloat16`
        are supported. (default: None)

.. note:: The foreach and fused implementations are typically faster than the for-loop,
          single-tensor implementation, with fused being theoretically fastest with both
          vertical and horizontal fusion. As such, if the user has not specified either
          flag (i.e., when foreach = fused = None), we will attempt defaulting to the foreach
          implementation when the tensors are all on CUDA. Why not fused? Since the fused
          implementation is relatively new, we want to give it sufficient bake-in time.
          To specify fused, pass True for fused. To force running the for-loop
          implementation, pass False for either foreach or fused. 
.. Note::
    A prototype implementation of Adam and AdamW for MPS supports `torch.float32` and `torch.float16`.
.. _Adam\: A Method for Stochastic Optimization:
    https://arxiv.org/abs/1412.6980
.. _On the Convergence of Adam and Beyond:
    https://openreview.net/forum?id=ryQu7f-RZ

#### Methods

- **`step(self, closure=None)`**
  Perform a single optimization step.

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `CosineAnnealingLR`

Set the learning rate of each parameter group using a cosine annealing schedule.

The :math:`\eta_{max}` is set to the initial lr and
:math:`T_{cur}` is the number of epochs since the last restart in SGDR:

.. math::
    \begin{aligned}
        \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
        + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
        & T_{cur} \neq (2k+1)T_{max}; \\
        \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
        \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
        & T_{cur} = (2k+1)T_{max}.
    \end{aligned}

When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
is defined recursively, the learning rate can be simultaneously modified
outside this scheduler by other operators. If the learning rate is set
solely by this scheduler, the learning rate at each step becomes:

.. math::
    \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
    \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

It has been proposed in
`SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
implements the cosine annealing part of SGDR, and not the restarts.

Args:
    optimizer (Optimizer): Wrapped optimizer.
    T_max (int): Maximum number of iterations.
    eta_min (float): Minimum learning rate. Default: 0.
    last_epoch (int): The index of last epoch. Default: -1.

.. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
    https://arxiv.org/abs/1608.03983

#### Methods

- **`get_lr(self)`**
  Retrieve the learning rate of each parameter group.

### `MultiStepLR`

Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.

Notice that such decay can happen simultaneously with other changes to the learning rate
from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

Args:
    optimizer (Optimizer): Wrapped optimizer.
    milestones (list): List of epoch indices. Must be increasing.
    gamma (float): Multiplicative factor of learning rate decay.
        Default: 0.1.
    last_epoch (int): The index of last epoch. Default: -1.

Example:
    >>> # xdoctest: +SKIP
    >>> # Assuming optimizer uses lr = 0.05 for all groups
    >>> # lr = 0.05     if epoch < 30
    >>> # lr = 0.005    if 30 <= epoch < 80
    >>> # lr = 0.0005   if epoch >= 80
    >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    >>> for epoch in range(100):
    >>>     train(...)
    >>>     validate(...)
    >>>     scheduler.step()

#### Methods

- **`get_lr(self)`**
  Compute the learning rate of each parameter group.

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

### `OptimizerConfig`

OptimizerConfig(optimizer: str = 'adam', base_lr: float = 0.01, weight_decay: float = 0.0005, momentum: float = 0.9)

### `Parameter`

A kind of Tensor that is to be considered a module parameter.

Parameters are :class:`~torch.Tensor` subclasses, that have a
very special property when used with :class:`Module` s - when they're
assigned as Module attributes they are automatically added to the list of
its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
Assigning a Tensor doesn't have such effect. This is because one might
want to cache some temporary state, like last hidden state of the RNN, in
the model. If there was no such class as :class:`Parameter`, these
temporaries would get registered too.

Args:
    data (Tensor): parameter tensor.
    requires_grad (bool, optional): if the parameter requires gradient. Note that
        the torch.no_grad() context does NOT affect the default behavior of
        Parameter creation--the Parameter will still have `requires_grad=True` in
        :class:`~no_grad` mode. See :ref:`locally-disable-grad-doc` for more
        details. Default: `True`

### `SGD`

Implements stochastic gradient descent (optionally with momentum).

.. math::
   \begin{aligned}
        &\rule{110mm}{0.4pt}                                                                 \\
        &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
            \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
        &\hspace{13mm} \:\mu \text{ (momentum)}, \:\tau \text{ (dampening)},
        \:\textit{ nesterov,}\:\textit{ maximize}                                     \\[-1.ex]
        &\rule{110mm}{0.4pt}                                                                 \\
        &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
        &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
        &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
        &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
        &\hspace{5mm}\textbf{if} \: \mu \neq 0                                               \\
        &\hspace{10mm}\textbf{if} \: t > 1                                                   \\
        &\hspace{15mm} \textbf{b}_t \leftarrow \mu \textbf{b}_{t-1} + (1-\tau) g_t           \\
        &\hspace{10mm}\textbf{else}                                                          \\
        &\hspace{15mm} \textbf{b}_t \leftarrow g_t                                           \\
        &\hspace{10mm}\textbf{if} \: \textit{nesterov}                                       \\
        &\hspace{15mm} g_t \leftarrow g_{t} + \mu \textbf{b}_t                             \\
        &\hspace{10mm}\textbf{else}                                                   \\[-1.ex]
        &\hspace{15mm} g_t  \leftarrow  \textbf{b}_t                                         \\
        &\hspace{5mm}\textbf{if} \: \textit{maximize}                                          \\
        &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} + \gamma g_t                   \\[-1.ex]
        &\hspace{5mm}\textbf{else}                                                    \\[-1.ex]
        &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma g_t                   \\[-1.ex]
        &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
        &\bf{return} \:  \theta_t                                                     \\[-1.ex]
        &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
   \end{aligned}

Nesterov momentum is based on the formula from
`On the importance of initialization and momentum in deep learning`__.

Args:
    params (iterable): iterable of parameters or named_parameters to optimize
        or iterable of dicts defining parameter groups. When using named_parameters,
        all parameters in all groups should be named
    lr (float, Tensor, optional): learning rate (default: 1e-3)
    momentum (float, optional): momentum factor (default: 0)
    dampening (float, optional): dampening for momentum (default: 0)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    nesterov (bool, optional): enables Nesterov momentum. Only applicable
        when momentum is non-zero. (default: False)
    maximize (bool, optional): maximize the objective with respect to the
        params, instead of minimizing (default: False)
    foreach (bool, optional): whether foreach implementation of optimizer
        is used. If unspecified by the user (so foreach is None), we will try to use
        foreach over the for-loop implementation on CUDA, since it is usually
        significantly more performant. Note that the foreach implementation uses
        ~ sizeof(params) more peak memory than the for-loop version due to the intermediates
        being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
        parameters through the optimizer at a time or switch this flag to False (default: None)
    differentiable (bool, optional): whether autograd should
        occur through the optimizer step in training. Otherwise, the step()
        function runs in a torch.no_grad() context. Setting to True can impair
        performance, so leave it False if you don't intend to run autograd
        through this instance (default: False)
    fused (bool, optional): whether the fused implementation is used.
        Currently, `torch.float64`, `torch.float32`, `torch.float16`, and `torch.bfloat16`
        are supported. (default: None)

.. note:: The foreach and fused implementations are typically faster than the for-loop,
          single-tensor implementation, with fused being theoretically fastest with both
          vertical and horizontal fusion. As such, if the user has not specified either
          flag (i.e., when foreach = fused = None), we will attempt defaulting to the foreach
          implementation when the tensors are all on CUDA. Why not fused? Since the fused
          implementation is relatively new, we want to give it sufficient bake-in time.
          To specify fused, pass True for fused. To force running the for-loop
          implementation, pass False for either foreach or fused. 


Example:
    >>> # xdoctest: +SKIP
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> optimizer.zero_grad()
    >>> loss_fn(model(input), target).backward()
    >>> optimizer.step()

__ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

.. note::
    The implementation of SGD with Momentum/Nesterov subtly differs from
    Sutskever et al. and implementations in some other frameworks.

    Considering the specific case of Momentum, the update can be written as

    .. math::
        \begin{aligned}
            v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
            p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
        \end{aligned}

    where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
    parameters, gradient, velocity, and momentum respectively.

    This is in contrast to Sutskever et al. and
    other frameworks which employ an update of the form

    .. math::
        \begin{aligned}
            v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
            p_{t+1} & = p_{t} - v_{t+1}.
        \end{aligned}

    The Nesterov version is analogously modified.

    Moreover, the initial value of the momentum buffer is set to the
    gradient value at the first step. This is in contrast to some other
    frameworks that initialize it to all zeros.

#### Methods

- **`step(self, closure=None)`**
  Perform a single optimization step.

### `SchedulerConfig`

SchedulerConfig(scheduler: Optional[str] = 'cos', steps: List[int] = <factory>, lr_decay: float = 0.1, max_epoch: int = 200)

### `StepLR`

Decays the learning rate of each parameter group by gamma every step_size epochs.

Notice that such decay can happen simultaneously with other changes to the learning rate
from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

Args:
    optimizer (Optimizer): Wrapped optimizer.
    step_size (int): Period of learning rate decay.
    gamma (float): Multiplicative factor of learning rate decay.
        Default: 0.1.
    last_epoch (int): The index of last epoch. Default: -1.

Example:
    >>> # xdoctest: +SKIP
    >>> # Assuming optimizer uses lr = 0.05 for all groups
    >>> # lr = 0.05     if epoch < 30
    >>> # lr = 0.005    if 30 <= epoch < 60
    >>> # lr = 0.0005   if 60 <= epoch < 90
    >>> # ...
    >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    >>> for epoch in range(100):
    >>>     train(...)
    >>>     validate(...)
    >>>     scheduler.step()

#### Methods

- **`get_lr(self)`**
  Compute the learning rate of each parameter group.
