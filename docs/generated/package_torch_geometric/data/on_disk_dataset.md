# on_disk_dataset

Part of `torch_geometric.data`
Module: `torch_geometric.data.on_disk_dataset`

## Classes (8)

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

### `Database`

Base class for inserting and retrieving data from a database.

A database acts as a persisted, out-of-memory and index-based key/value
store for tensor and custom data:

.. code-block:: python

    db = Database()
    db[0] = Data(x=torch.randn(5, 16), y=0, z='id_0')
    print(db[0])
    >>> Data(x=[5, 16], y=0, z='id_0')

To improve efficiency, it is recommended to specify the underlying
:obj:`schema` of the data:

.. code-block:: python

    db = Database(schema={  # Custom schema:
        # Tensor information can be specified through a dictionary:
        'x': dict(dtype=torch.float, size=(-1, 16)),
        'y': int,
        'z': str,
    })
    db[0] = dict(x=torch.randn(5, 16), y=0, z='id_0')
    print(db[0])
    >>> {'x': torch.tensor(...), 'y': 0, 'z': 'id_0'}

In addition, databases support batch-wise insert and get, and support
syntactic sugar known from indexing :python:`Python` lists, *e.g.*:

.. code-block:: python

    db = Database()
    db[2:5] = torch.randn(3, 16)
    print(db[torch.tensor([2, 3])])
    >>> [torch.tensor(...), torch.tensor(...)]

Args:
    schema (Any or Tuple[Any] or Dict[str, Any], optional): The schema of
        the input data.
        Can take :obj:`int`, :obj:`float`, :obj:`str`, :obj:`object`, or a
        dictionary with :obj:`dtype` and :obj:`size` keys (for specifying
        tensor data) as input, and can be nested as a tuple or dictionary.
        Specifying the schema will improve efficiency, since by default the
        database will use python pickling for serializing and
        deserializing. (default: :obj:`object`)

#### Methods

- **`connect(self) -> None`**
  Connects to the database.

- **`close(self) -> None`**
  Closes the connection to the database.

- **`insert(self, index: int, data: Any) -> None`**
  Inserts data at the specified index.

### `Dataset`

Dataset base class for creating graph datasets.
See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
create_dataset.html>`__ for the accompanying tutorial.

Args:
    root (str, optional): Root directory where the dataset should be saved.
        (optional: :obj:`None`)
    transform (callable, optional): A function/transform that takes in a
        :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object and returns a
        transformed version.
        The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        a :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object and returns a
        transformed version.
        The data object will be transformed before being saved to disk.
        (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in a
        :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object and returns a
        boolean value, indicating whether the data object should be
        included in the final dataset. (default: :obj:`None`)
    log (bool, optional): Whether to print any console output while
        downloading and processing the dataset. (default: :obj:`True`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

- **`len(self) -> int`**
  Returns the number of data objects stored in the dataset.

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

### `RocksDatabase`

An index-based key/value database based on :obj:`RocksDB`.

.. note::
    This database implementation requires the :obj:`rocksdict` package.

.. warning::
    :class:`RocksDatabase` is currently less optimized than
    :class:`SQLiteDatabase`.

Args:
    path (str): The path to where the database should be saved.
    schema (Any or Tuple[Any] or Dict[str, Any], optional): The schema of
        the input data.
        Can take :obj:`int`, :obj:`float`, :obj:`str`, :obj:`object`, or a
        dictionary with :obj:`dtype` and :obj:`size` keys (for specifying
        tensor data) as input, and can be nested as a tuple or dictionary.
        Specifying the schema will improve efficiency, since by default the
        database will use python pickling for serializing and
        deserializing. (default: :obj:`object`)

#### Methods

- **`connect(self) -> None`**
  Connects to the database.

- **`close(self) -> None`**
  Closes the connection to the database.

- **`to_key(index: int) -> bytes`**

### `SQLiteDatabase`

An index-based key/value database based on :obj:`sqlite3`.

.. note::
    This database implementation requires the :obj:`sqlite3` package.

Args:
    path (str): The path to where the database should be saved.
    name (str): The name of the table to save the data to.
    schema (Any or Tuple[Any] or Dict[str, Any], optional): The schema of
        the input data.
        Can take :obj:`int`, :obj:`float`, :obj:`str`, :obj:`object`, or a
        dictionary with :obj:`dtype` and :obj:`size` keys (for specifying
        tensor data) as input, and can be nested as a tuple or dictionary.
        Specifying the schema will improve efficiency, since by default the
        database will use python pickling for serializing and
        deserializing. (default: :obj:`object`)

#### Methods

- **`connect(self) -> None`**
  Connects to the database.

- **`close(self) -> None`**
  Closes the connection to the database.

- **`insert(self, index: int, data: Any) -> None`**
  Inserts data at the specified index.

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
