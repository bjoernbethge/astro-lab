# graph_saint

Part of `torch_geometric.loader`
Module: `torch_geometric.loader.graph_saint`

## Classes (6)

### `GraphSAINTEdgeSampler`

The GraphSAINT edge sampler class (see
:class:`~torch_geometric.loader.GraphSAINTSampler`).

### `GraphSAINTNodeSampler`

The GraphSAINT node sampler class (see
:class:`~torch_geometric.loader.GraphSAINTSampler`).

### `GraphSAINTRandomWalkSampler`

The GraphSAINT random walk sampler class (see
:class:`~torch_geometric.loader.GraphSAINTSampler`).

Args:
    walk_length (int): The length of each random walk.

### `GraphSAINTSampler`

The GraphSAINT sampler base class from the `"GraphSAINT: Graph
Sampling Based Inductive Learning Method"
<https://arxiv.org/abs/1907.04931>`_ paper.
Given a graph in a :obj:`data` object, this class samples nodes and
constructs subgraphs that can be processed in a mini-batch fashion.
Normalization coefficients for each mini-batch are given via
:obj:`node_norm` and :obj:`edge_norm` data attributes.

.. note::

    See :class:`~torch_geometric.loader.GraphSAINTNodeSampler`,
    :class:`~torch_geometric.loader.GraphSAINTEdgeSampler` and
    :class:`~torch_geometric.loader.GraphSAINTRandomWalkSampler` for
    currently supported samplers.
    For an example of using GraphSAINT sampling, see
    `examples/graph_saint.py <https://github.com/pyg-team/
    pytorch_geometric/blob/master/examples/graph_saint.py>`_.

Args:
    data (torch_geometric.data.Data): The graph data object.
    batch_size (int): The approximate number of samples per batch.
    num_steps (int, optional): The number of iterations per epoch.
        (default: :obj:`1`)
    sample_coverage (int): How many samples per node should be used to
        compute normalization statistics. (default: :obj:`0`)
    save_dir (str, optional): If set, will save normalization statistics to
        the :obj:`save_dir` directory for faster re-use.
        (default: :obj:`None`)
    log (bool, optional): If set to :obj:`False`, will not log any
        pre-processing progress. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size` or
        :obj:`num_workers`.

### `SparseTensor`

#### Methods

- **`size(self, dim: int) -> int`**

- **`nnz(self) -> int`**

- **`is_cuda(self) -> bool`**

### `tqdm`

Decorate an iterable object, returning an iterator which acts exactly
like the original iterable, but prints a dynamically updating
progressbar every time a value is requested.

Parameters
----------
iterable  : iterable, optional
    Iterable to decorate with a progressbar.
    Leave blank to manually manage the updates.
desc  : str, optional
    Prefix for the progressbar.
total  : int or float, optional
    The number of expected iterations. If unspecified,
    len(iterable) is used if possible. If float("inf") or as a last
    resort, only basic progress statistics are displayed
    (no ETA, no progressbar).
    If `gui` is True and this parameter needs subsequent updating,
    specify an initial arbitrary large positive number,
    e.g. 9e9.
leave  : bool, optional
    If [default: True], keeps all traces of the progressbar
    upon termination of iteration.
    If `None`, will leave only if `position` is `0`.
file  : `io.TextIOWrapper` or `io.StringIO`, optional
    Specifies where to output the progress messages
    (default: sys.stderr). Uses `file.write(str)` and `file.flush()`
    methods.  For encoding, see `write_bytes`.
ncols  : int, optional
    The width of the entire output message. If specified,
    dynamically resizes the progressbar to stay within this bound.
    If unspecified, attempts to use environment width. The
    fallback is a meter width of 10 and no limit for the counter and
    statistics. If 0, will not print any meter (only stats).
mininterval  : float, optional
    Minimum progress display update interval [default: 0.1] seconds.
maxinterval  : float, optional
    Maximum progress display update interval [default: 10] seconds.
    Automatically adjusts `miniters` to correspond to `mininterval`
    after long display update lag. Only works if `dynamic_miniters`
    or monitor thread is enabled.
miniters  : int or float, optional
    Minimum progress display update interval, in iterations.
    If 0 and `dynamic_miniters`, will automatically adjust to equal
    `mininterval` (more CPU efficient, good for tight loops).
    If > 0, will skip display of specified number of iterations.
    Tweak this and `mininterval` to get very efficient loops.
    If your progress is erratic with both fast and slow iterations
    (network, skipping items, etc) you should set miniters=1.
ascii  : bool or str, optional
    If unspecified or False, use unicode (smooth blocks) to fill
    the meter. The fallback is to use ASCII characters " 123456789#".
disable  : bool, optional
    Whether to disable the entire progressbar wrapper
    [default: False]. If set to None, disable on non-TTY.
unit  : str, optional
    String that will be used to define the unit of each iteration
    [default: it].
unit_scale  : bool or int or float, optional
    If 1 or True, the number of iterations will be reduced/scaled
    automatically and a metric prefix following the
    International System of Units standard will be added
    (kilo, mega, etc.) [default: False]. If any other non-zero
    number, will scale `total` and `n`.
dynamic_ncols  : bool, optional
    If set, constantly alters `ncols` and `nrows` to the
    environment (allowing for window resizes) [default: False].
smoothing  : float, optional
    Exponential moving average smoothing factor for speed estimates
    (ignored in GUI mode). Ranges from 0 (average speed) to 1
    (current/instantaneous speed) [default: 0.3].
bar_format  : str, optional
    Specify a custom bar string formatting. May impact performance.
    [default: '{l_bar}{bar}{r_bar}'], where
    l_bar='{desc}: {percentage:3.0f}%|' and
    r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, '
        '{rate_fmt}{postfix}]'
    Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt,
        percentage, elapsed, elapsed_s, ncols, nrows, desc, unit,
        rate, rate_fmt, rate_noinv, rate_noinv_fmt,
        rate_inv, rate_inv_fmt, postfix, unit_divisor,
        remaining, remaining_s, eta.
    Note that a trailing ": " is automatically removed after {desc}
    if the latter is empty.
initial  : int or float, optional
    The initial counter value. Useful when restarting a progress
    bar [default: 0]. If using float, consider specifying `{n:.3f}`
    or similar in `bar_format`, or specifying `unit_scale`.
position  : int, optional
    Specify the line offset to print this bar (starting from 0)
    Automatic if unspecified.
    Useful to manage multiple bars at once (eg, from threads).
postfix  : dict or *, optional
    Specify additional stats to display at the end of the bar.
    Calls `set_postfix(**postfix)` if possible (dict).
unit_divisor  : float, optional
    [default: 1000], ignored unless `unit_scale` is True.
write_bytes  : bool, optional
    Whether to write bytes. If (default: False) will write unicode.
lock_args  : tuple, optional
    Passed to `refresh` for intermediate output
    (initialisation, iterating, and updating).
nrows  : int, optional
    The screen height. If specified, hides nested bars outside this
    bound. If unspecified, attempts to use environment height.
    The fallback is 20.
colour  : str, optional
    Bar colour (e.g. 'green', '#00ff00').
delay  : float, optional
    Don't display until [default: 0] seconds have elapsed.
gui  : bool, optional
    WARNING: internal parameter - do not use.
    Use tqdm.gui.tqdm(...) instead. If set, will attempt to use
    matplotlib animations for a graphical output [default: False].

Returns
-------
out  : decorated iterator.

#### Methods

- **`format_sizeof(num, suffix='', divisor=1000)`**
  Formats a number (greater than unity) with SI Order of Magnitude

- **`format_interval(t)`**
  Formats a number of seconds as a clock time, [H:]MM:SS

- **`format_num(n)`**
  Intelligent scientific notation (.3g).
