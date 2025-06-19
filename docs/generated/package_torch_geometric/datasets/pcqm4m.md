# pcqm4m

Part of `torch_geometric.datasets`
Module: `torch_geometric.datasets.pcqm4m`

## Functions (2)

### `download_url(url: str, folder: str, log: bool = True, filename: Optional[str] = None)`

Downloads the content of an URL to a specific folder.

Args:
    url (str): The URL.
    folder (str): The folder.
    log (bool, optional): If :obj:`False`, will not print anything to the
        console. (default: :obj:`True`)
    filename (str, optional): The filename of the downloaded file. If set
        to :obj:`None`, will correspond to the filename given by the URL.
        (default: :obj:`None`)

### `extract_zip(path: str, folder: str, log: bool = True) -> None`

Extracts a zip archive to a specific folder.

Args:
    path (str): The path to the tar archive.
    folder (str): The folder.
    log (bool, optional): If :obj:`False`, will not print anything to the
        console. (default: :obj:`True`)

## Classes (6)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `BaseData`

#### Methods

- **`stores_as(self, data: Self)`**

- **`to_dict(self) -> Dict[str, Any]`**
  Returns a dictionary of stored key/value pairs.

- **`to_namedtuple(self) -> <function NamedTuple at 0x000001FE17E66F20>`**
  Returns a :obj:`NamedTuple` of stored key/value pairs.

### `Data`

A data object describing a homogeneous graph.
The data object can hold node-level, link-level and graph-level attributes.
In general, :class:`~torch_geometric.data.Data` tries to mimic the
behavior of a regular :python:`Python` dictionary.
In addition, it provides useful functionality for analyzing graph
structures, and provides basic PyTorch tensor functionalities.
See `here <https://pytorch-geometric.readthedocs.io/en/latest/get_started/
introduction.html#data-handling-of-graphs>`__ for the accompanying
tutorial.

.. code-block:: python

    from torch_geometric.data import Data

    data = Data(x=x, edge_index=edge_index, ...)

    # Add additional arguments to `data`:
    data.train_idx = torch.tensor([...], dtype=torch.long)
    data.test_mask = torch.tensor([...], dtype=torch.bool)

    # Analyzing the graph structure:
    data.num_nodes
    >>> 23

    data.is_directed()
    >>> False

    # PyTorch tensor functionality:
    data = data.pin_memory()
    data = data.to('cuda:0', non_blocking=True)

Args:
    x (torch.Tensor, optional): Node feature matrix with shape
        :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
    edge_index (LongTensor, optional): Graph connectivity in COO format
        with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
    edge_attr (torch.Tensor, optional): Edge feature matrix with shape
        :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
    y (torch.Tensor, optional): Graph-level or node-level ground-truth
        labels with arbitrary shape. (default: :obj:`None`)
    pos (torch.Tensor, optional): Node position matrix with shape
        :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
    time (torch.Tensor, optional): The timestamps for each event with shape
        :obj:`[num_edges]` or :obj:`[num_nodes]`. (default: :obj:`None`)
    **kwargs (optional): Additional attributes.

#### Methods

- **`stores_as(self, data: Self)`**

- **`to_dict(self) -> Dict[str, Any]`**
  Returns a dictionary of stored key/value pairs.

- **`to_namedtuple(self) -> <function NamedTuple at 0x000001FE17E66F20>`**
  Returns a :obj:`NamedTuple` of stored key/value pairs.

### `OnDiskDataset`

Dataset base class for creating large graph datasets which do not
easily fit into CPU memory at once by leveraging a :class:`Database`
backend for on-disk storage and access of data objects.

Args:
    root (str): Root directory where the dataset should be saved.
    transform (callable, optional): A function/transform that takes in a
        :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object and returns a
        transformed version.
        The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in a
        :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object and returns a
        boolean value, indicating whether the data object should be
        included in the final dataset. (default: :obj:`None`)
    backend (str): The :class:`Database` backend to use
        (one of :obj:`"sqlite"` or :obj:`"rocksdb"`).
        (default: :obj:`"sqlite"`)
    schema (Any or Tuple[Any] or Dict[str, Any], optional): The schema of
        the input data.
        Can take :obj:`int`, :obj:`float`, :obj:`str`, :obj:`object`, or a
        dictionary with :obj:`dtype` and :obj:`size` keys (for specifying
        tensor data) as input, and can be nested as a tuple or dictionary.
        Specifying the schema will improve efficiency, since by default the
        database will use python pickling for serializing and
        deserializing. If specified to anything different than
        :obj:`object`, implementations of :class:`OnDiskDataset` need to
        override :meth:`serialize` and :meth:`deserialize` methods.
        (default: :obj:`object`)
    log (bool, optional): Whether to print any console output while
        downloading and processing the dataset. (default: :obj:`True`)

#### Methods

- **`close(self) -> None`**
  Closes the connection to the underlying database.

- **`serialize(self, data: torch_geometric.data.data.BaseData) -> Any`**
  Serializes the :class:`~torch_geometric.data.Data` or

- **`deserialize(self, data: Any) -> torch_geometric.data.data.BaseData`**
  Deserializes the DB entry into a

### `PCQM4Mv2`

The PCQM4Mv2 dataset from the `"OGB-LSC: A Large-Scale Challenge for
Machine Learning on Graphs" <https://arxiv.org/abs/2103.09430>`_ paper.
:class:`PCQM4Mv2` is a quantum chemistry dataset originally curated under
the `PubChemQC project
<https://pubs.acs.org/doi/10.1021/acs.jcim.7b00083>`_.
The task is to predict the DFT-calculated HOMO-LUMO energy gap of molecules
given their 2D molecular graphs.

.. note::
    This dataset uses the :class:`OnDiskDataset` base class to load data
    dynamically from disk.

Args:
    root (str): Root directory where the dataset should be saved.
    split (str, optional): If :obj:`"train"`, loads the training dataset.
        If :obj:`"val"`, loads the validation dataset.
        If :obj:`"test"`, loads the test dataset.
        If :obj:`"holdout"`, loads the holdout dataset.
        (default: :obj:`"train"`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    backend (str): The :class:`Database` backend to use.
        (default: :obj:`"sqlite"`)
    from_smiles (callable, optional): A custom function that takes a SMILES
        string and outputs a :obj:`~torch_geometric.data.Data` object.
        If not set, defaults to :meth:`~torch_geometric.utils.from_smiles`.
        (default: :obj:`None`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

- **`serialize(self, data: torch_geometric.data.data.BaseData) -> Dict[str, Any]`**
  Serializes the :class:`~torch_geometric.data.Data` or

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
