# profile

Part of `torch_geometric.profile`
Module: `torch_geometric.profile.profile`

## Functions (15)

### `byte_to_megabyte(value: int, digits: int = 2) -> float`

### `contextmanager(func)`

@contextmanager decorator.

Typical usage:

    @contextmanager
    def some_generator(<arguments>):
        <setup>
        try:
            yield <value>
        finally:
            <cleanup>

This makes this:

    with some_generator(<arguments>) as <variable>:
        <body>

equivalent to this:

    <setup>
    try:
        <variable> = <value>
        <body>
    finally:
        <cleanup>

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

### `format_prof_time(time)`

### `get_gpu_memory_from_ipex(device: int = 0, digits=2) -> Tuple[float, float, float]`

Returns the XPU memory statistics.

Args:
    device (int, optional): The GPU device identifier. (default: :obj:`0`)
    digits (int): The number of decimals to use for megabytes.
        (default: :obj:`2`)

### `get_gpu_memory_from_nvidia_smi(device: int = 0, digits: int = 2) -> Tuple[float, float]`

Returns the free and used GPU memory in megabytes, as reported by
:obj:`nivdia-smi`.

.. note::

    :obj:`nvidia-smi` will generally overestimate the amount of memory used
    by the actual program, see `here <https://pytorch.org/docs/stable/
    notes/faq.html#my-gpu-memory-isn-t-freed-properly>`__.

Args:
    device (int, optional): The GPU device identifier. (default: :obj:`1`)
    digits (int): The number of decimals to use for megabytes.
        (default: :obj:`2`)

### `get_stats_summary(stats_list: Union[List[torch_geometric.profile.profile.GPUStats], List[torch_geometric.profile.profile.CUDAStats]]) -> Union[torch_geometric.profile.profile.GPUStatsSummary, torch_geometric.profile.profile.CUDAStatsSummary]`

Creates a summary of collected runtime and memory statistics.
Returns a :obj:`GPUStatsSummary` if list of :obj:`GPUStats` was passed,
otherwise (list of :obj:`CUDAStats` was passed),
returns a :obj:`CUDAStatsSummary`.

Args:
    stats_list (Union[List[GPUStats], List[CUDAStats]]): A list of
        :obj:`GPUStats` or :obj:`CUDAStats` objects, as returned by
        :meth:`~torch_geometric.profile.profileit`.

### `print_time_total(p)`

### `profileit(device: str)`

A decorator to facilitate profiling a function, *e.g.*, obtaining
training runtime and memory statistics of a specific model on a specific
dataset.
Returns a :obj:`GPUStats` if :obj:`device` is :obj:`xpu` or extended
object :obj:`CUDAStats`, if :obj:`device` is :obj:`cuda`.

Args:
    device (str): Target device for profiling. Options are:
        :obj:`cuda` and obj:`xpu`.

.. code-block:: python

    @profileit("cuda")
    def train(model, optimizer, x, edge_index, y):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        return float(loss)

    loss, stats = train(model, x, edge_index, y)

### `read_from_memlab(line_profiler: Any) -> List[float]`

### `rename_profile_file(*args)`

### `save_profile_data(csv_data, events, use_cuda)`

### `torch_profile(export_chrome_trace=True, csv_data=None, write_csv=None)`

### `trace_handler(p)`

### `xpu_profile(export_chrome_trace=True)`

## Classes (10)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `CUDAStats`

CUDAStats(time: float, max_allocated_gpu: float, max_reserved_gpu: float, max_active_gpu: float, nvidia_smi_free_cuda: float, nvidia_smi_used_cuda: float)

### `CUDAStatsSummary`

CUDAStatsSummary(time_mean: float, time_std: float, max_allocated_gpu: float, max_reserved_gpu: float, max_active_gpu: float, min_nvidia_smi_free_cuda: float, max_nvidia_smi_used_cuda: float)

### `ContextDecorator`

A base class or mixin that enables context managers to work as decorators.

### `EventList`

A list of Events (for pretty printing).

#### Methods

- **`table(self, sort_by=None, row_limit=100, max_src_column_width=75, max_name_column_width=55, max_shapes_column_width=80, header=None, top_level_events_only=False)`**
  Print an EventList as a nicely formatted table.

- **`export_chrome_trace(self, path)`**
  Export an EventList as a Chrome tracing tools file.

- **`supported_export_stacks_metrics(self)`**

### `GPUStats`

GPUStats(time: float, max_allocated_gpu: float, max_reserved_gpu: float, max_active_gpu: float)

### `GPUStatsSummary`

GPUStatsSummary(time_mean: float, time_std: float, max_allocated_gpu: float, max_reserved_gpu: float, max_active_gpu: float)

### `ProfilerActivity`

Members:

CPU

XPU

MTIA

CUDA

HPU

PrivateUse1

### `profile`

Profiler context manager.

Args:
    activities (iterable): list of activity groups (CPU, CUDA) to use in profiling, supported values:
        ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``,
        ``torch.profiler.ProfilerActivity.XPU``.
        Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA
        or (when available) ProfilerActivity.XPU.
    schedule (Callable): callable that takes step (int) as a single parameter and returns
        ``ProfilerAction`` value that specifies the profiler action to perform at each step.
    on_trace_ready (Callable): callable that is called at each step when ``schedule``
        returns ``ProfilerAction.RECORD_AND_SAVE`` during the profiling.
    record_shapes (bool): save information about operator's input shapes.
    profile_memory (bool): track tensor memory allocation/deallocation.
    with_stack (bool): record source information (file and line number) for the ops.
    with_flops (bool): use formula to estimate the FLOPs (floating point operations) of specific operators
        (matrix multiplication and 2D convolution).
    with_modules (bool): record module hierarchy (including function names)
        corresponding to the callstack of the op. e.g. If module A's forward call's
        module B's forward which contains an aten::add op,
        then aten::add's module hierarchy is A.B
        Note that this support exist, at the moment, only for TorchScript models
        and not eager mode models.
    experimental_config (_ExperimentalConfig) : A set of experimental options
        used for Kineto library features. Note, backward compatibility is not guaranteed.
    execution_trace_observer (ExecutionTraceObserver) : A PyTorch Execution Trace Observer object.
        `PyTorch Execution Traces <https://arxiv.org/pdf/2305.14516.pdf>`__ offer a graph based
        representation of AI/ML workloads and enable replay benchmarks, simulators, and emulators.
        When this argument is included the observer start() and stop() will be called for the
        same time window as PyTorch profiler. See the examples section below for a code sample.
    acc_events (bool): Enable the accumulation of FunctionEvents across multiple profiling cycles
    use_cuda (bool):
        .. deprecated:: 1.8.1
            use ``activities`` instead.

.. note::
    Use :func:`~torch.profiler.schedule` to generate the callable schedule.
    Non-default schedules are useful when profiling long training jobs
    and allow the user to obtain multiple traces at the different iterations
    of the training process.
    The default schedule simply records all the events continuously for the
    duration of the context manager.

.. note::
    Use :func:`~torch.profiler.tensorboard_trace_handler` to generate result files for TensorBoard:

    ``on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name)``

    After profiling, result files can be found in the specified directory. Use the command:

    ``tensorboard --logdir dir_name``

    to see the results in TensorBoard.
    For more information, see
    `PyTorch Profiler TensorBoard Plugin <https://github.com/pytorch/kineto/tree/master/tb_plugin>`__

.. note::
    Enabling shape and stack tracing results in additional overhead.
    When record_shapes=True is specified, profiler will temporarily hold references to the tensors;
    that may further prevent certain optimizations that depend on the reference count and introduce
    extra tensor copies.


Examples:

.. code-block:: python

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        code_to_profile()
    print(p.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))

Using the profiler's ``schedule``, ``on_trace_ready`` and ``step`` functions:

.. code-block:: python

    # Non-default profiler schedule allows user to turn profiler on and off
    # on different iterations of the training loop;
    # trace_handler is called every time a new trace becomes available
    def trace_handler(prof):
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))
        # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],

        # In this example with wait=1, warmup=1, active=2, repeat=1,
        # profiler will skip the first step/iteration,
        # start warming up on the second, record
        # the third and the forth iterations,
        # after which the trace will become available
        # and on_trace_ready (when set) is called;
        # the cycle repeats starting with the next step

        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2,
            repeat=1),
        on_trace_ready=trace_handler
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
        # used when outputting for tensorboard
        ) as p:
            for iter in range(N):
                code_iteration_to_profile(iter)
                # send a signal to the profiler that the next iteration has started
                p.step()

The following sample shows how to setup up an Execution Trace Observer (`execution_trace_observer`)

.. code-block:: python

    with torch.profiler.profile(
        ...
        execution_trace_observer=(
            ExecutionTraceObserver().register_callback("./execution_trace.json")
        ),
    ) as p:
        for iter in range(N):
            code_iteration_to_profile(iter)
            p.step()

You can also refer to test_execution_trace_with_kineto() in tests/profiler/test_profiler.py.
Note: One can also pass any object satisfying the _ITraceObserver interface.

#### Methods

- **`start(self)`**

- **`stop(self)`**

- **`step(self)`**
  Signals the profiler that the next profiling step has started.

### `timeit`

A context decorator to facilitate timing a function, *e.g.*, obtaining
the runtime of a specific model on a specific dataset.

.. code-block:: python

    @torch.no_grad()
    def test(model, x, edge_index):
        return model(x, edge_index)

    with timeit() as t:
        z = test(model, x, edge_index)
    time = t.duration

Args:
    log (bool, optional): If set to :obj:`False`, will not log any runtime
        to the console. (default: :obj:`True`)
    avg_time_divisor (int, optional): If set to a value greater than
        :obj:`1`, will divide the total time by this value. Useful for
        calculating the average of runtimes within a for-loop.
        (default: :obj:`0`)

#### Methods

- **`reset(self)`**
  Prints the duration and resets current timer.
