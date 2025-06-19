# profile

Part of `torch_geometric.torch_geometric`
Module: `torch_geometric.profile`

## Description

GNN profiling package.

## Functions (15)

### `benchmark(funcs: List[Callable], args: Union[Tuple[Any], List[Tuple[Any]]], num_steps: int, func_names: Optional[List[str]] = None, num_warmups: int = 10, backward: bool = False, per_step: bool = False, progress_bar: bool = False)`

Benchmark a list of functions :obj:`funcs` that receive the same set
of arguments :obj:`args`.

Args:
    funcs ([Callable]): The list of functions to benchmark.
    args ((Any, ) or [(Any, )]): The arguments to pass to the functions.
        Can be a list of arguments for each function in :obj:`funcs` in
        case their headers differ.
        Alternatively, you can pass in functions that generate arguments
        on-the-fly (e.g., useful for benchmarking models on various sizes).
    num_steps (int): The number of steps to run the benchmark.
    func_names ([str], optional): The names of the functions. If not given,
        will try to infer the name from the function itself.
        (default: :obj:`None`)
    num_warmups (int, optional): The number of warmup steps.
        (default: :obj:`10`)
    backward (bool, optional): If set to :obj:`True`, will benchmark both
        forward and backward passes. (default: :obj:`False`)
    per_step (bool, optional): If set to :obj:`True`, will report runtimes
        per step. (default: :obj:`False`)
    progress_bar (bool, optional): If set to :obj:`True`, will print a
        progress bar during benchmarking. (default: :obj:`False`)

### `count_parameters(model: torch.nn.modules.module.Module) -> int`

Given a :class:`torch.nn.Module`, count its trainable parameters.

Args:
    model (torch.nn.Model): The model.

### `get_cpu_memory_from_gc() -> int`

Returns the used CPU memory in bytes, as reported by the
:python:`Python` garbage collector.

### `get_data_size(data: torch_geometric.data.data.BaseData) -> int`

Given a :class:`torch_geometric.data.Data` object, get its theoretical
memory usage in bytes.

Args:
    data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
        The :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` graph object.

### `get_gpu_memory_from_gc(device: int = 0) -> int`

Returns the used GPU memory in bytes, as reported by the
:python:`Python` garbage collector.

Args:
    device (int, optional): The GPU device identifier. (default: :obj:`1`)

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

### `get_model_size(model: torch.nn.modules.module.Module) -> int`

Given a :class:`torch.nn.Module`, get its actual disk size in bytes.

Args:
    model (torch model): The model.

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

### `rename_profile_file(*args)`

### `torch_profile(export_chrome_trace=True, csv_data=None, write_csv=None)`

### `trace_handler(p)`

### `xpu_profile(export_chrome_trace=True)`

## Classes (1)

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
