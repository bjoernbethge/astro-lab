# mixin

Part of `torch_geometric.loader`
Module: `torch_geometric.loader.mixin`

## Functions (2)

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

### `get_numa_nodes_cores() -> Dict[str, Any]`

Parses numa nodes information into a dictionary.

..code-block::

    {<node_id>: [(<core_id>, [<sibling_thread_id_0>, <sibling_thread_id_1>
    ...]), ...], ...}

    # For example:
    {0: [(0, [0, 4]), (1, [1, 5])], 1: [(2, [2, 6]), (3, [3, 7])]}

If not available, returns an empty dictionary.

## Classes (6)

### `AffinityMixin`

A context manager to enable CPU affinity for data loader workers
(only used when running on CPU devices).

Affinitization places data loader workers threads on specific CPU cores.
In effect, it allows for more efficient local memory allocation and reduces
remote memory calls.
Every time a process or thread moves from one core to another, registers
and caches need to be flushed and reloaded.
This can become very costly if it happens often, and our threads may also
no longer be close to their data, or be able to share data in a cache.

See `here <https://pytorch-geometric.readthedocs.io/en/latest/advanced/
cpu_affinity.html>`__ for the accompanying tutorial.

.. warning::

    To correctly affinitize compute threads (*i.e.* with
    :obj:`KMP_AFFINITY`), please make sure that you exclude
    :obj:`loader_cores` from the list of cores available for the main
    process.
    This will cause core oversubsription and exacerbate performance.

.. code-block:: python

    loader = NeigborLoader(data, num_workers=3)
    with loader.enable_cpu_affinity(loader_cores=[0, 1, 2]):
        for batch in loader:
            pass

#### Methods

- **`enable_cpu_affinity(self, loader_cores: Union[List[List[int]], List[int], NoneType] = None) -> None`**
  Enables CPU affinity.

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `HeteroData`

A data object describing a heterogeneous graph, holding multiple node
and/or edge types in disjunct storage objects.
Storage objects can hold either node-level, link-level or graph-level
attributes.
In general, :class:`~torch_geometric.data.HeteroData` tries to mimic the
behavior of a regular **nested** :python:`Python` dictionary.
In addition, it provides useful functionality for analyzing graph
structures, and provides basic PyTorch tensor functionalities.

.. code-block::

    from torch_geometric.data import HeteroData

    data = HeteroData()

    # Create two node types "paper" and "author" holding a feature matrix:
    data['paper'].x = torch.randn(num_papers, num_paper_features)
    data['author'].x = torch.randn(num_authors, num_authors_features)

    # Create an edge type "(author, writes, paper)" and building the
    # graph connectivity:
    data['author', 'writes', 'paper'].edge_index = ...  # [2, num_edges]

    data['paper'].num_nodes
    >>> 23

    data['author', 'writes', 'paper'].num_edges
    >>> 52

    # PyTorch tensor functionality:
    data = data.pin_memory()
    data = data.to('cuda:0', non_blocking=True)

Note that there exists multiple ways to create a heterogeneous graph data,
*e.g.*:

* To initialize a node of type :obj:`"paper"` holding a node feature
  matrix :obj:`x_paper` named :obj:`x`:

  .. code-block:: python

    from torch_geometric.data import HeteroData

    # (1) Assign attributes after initialization,
    data = HeteroData()
    data['paper'].x = x_paper

    # or (2) pass them as keyword arguments during initialization,
    data = HeteroData(paper={ 'x': x_paper })

    # or (3) pass them as dictionaries during initialization,
    data = HeteroData({'paper': { 'x': x_paper }})

* To initialize an edge from source node type :obj:`"author"` to
  destination node type :obj:`"paper"` with relation type :obj:`"writes"`
  holding a graph connectivity matrix :obj:`edge_index_author_paper` named
  :obj:`edge_index`:

  .. code-block:: python

    # (1) Assign attributes after initialization,
    data = HeteroData()
    data['author', 'writes', 'paper'].edge_index = edge_index_author_paper

    # or (2) pass them as keyword arguments during initialization,
    data = HeteroData(author__writes__paper={
        'edge_index': edge_index_author_paper
    })

    # or (3) pass them as dictionaries during initialization,
    data = HeteroData({
        ('author', 'writes', 'paper'):
        { 'edge_index': edge_index_author_paper }
    })

#### Methods

- **`stores_as(self, data: Self)`**

- **`node_items(self) -> List[Tuple[str, torch_geometric.data.storage.NodeStorage]]`**
  Returns a list of node type and node storage pairs.

- **`edge_items(self) -> List[Tuple[Tuple[str, str, str], torch_geometric.data.storage.EdgeStorage]]`**
  Returns a list of edge type and edge storage pairs.

### `LogMemoryMixin`

A context manager to enable logging of memory consumption in
:class:`~torch.utils.data.DataLoader` workers.

#### Methods

- **`enable_memory_log(self) -> None`**

### `MultithreadingMixin`

A context manager to enable multi-threading in
:class:`~torch.utils.data.DataLoader` workers.
It changes the default value of threads used in the loader from :obj:`1`
to :obj:`worker_threads`.

#### Methods

- **`enable_multithreading(self, worker_threads: Optional[int] = None) -> None`**
  Enables multithreading in worker subprocesses.

### `WorkerInitWrapper`

Wraps the :attr:`worker_init_fn` argument for
:class:`torch.utils.data.DataLoader` workers.
