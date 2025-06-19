# jodie

Part of `torch_geometric.datasets`
Module: `torch_geometric.datasets.jodie`

## Functions (1)

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

## Classes (3)

### `InMemoryDataset`

Dataset base class for creating graph datasets which easily fit
into CPU memory.
See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
create_dataset.html#creating-in-memory-datasets>`__ for the accompanying
tutorial.

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

- **`len(self) -> int`**
  Returns the number of data objects stored in the dataset.

- **`get(self, idx: int) -> torch_geometric.data.data.BaseData`**
  Gets the data object at index :obj:`idx`.

- **`load(self, path: str, data_cls: Type[torch_geometric.data.data.BaseData] = <class 'torch_geometric.data.data.Data'>) -> None`**
  Loads the dataset from the file path :obj:`path`.

### `JODIEDataset`

The temporal graph datasets
from the `"JODIE: Predicting Dynamic Embedding
Trajectory in Temporal Interaction Networks"
<https://cs.stanford.edu/~srijan/pubs/jodie-kdd2019.pdf>`_ paper.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"Reddit"`,
        :obj:`"Wikipedia"`, :obj:`"MOOC"`, and :obj:`"LastFM"`).
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - Name
      - #nodes
      - #edges
      - #features
      - #classes
    * - Reddit
      - 6,509
      - 25,470
      - 172
      - 1
    * - Wikipedia
      - 9,227
      - 157,474
      - 172
      - 2
    * - MOOC
      - 7,144
      - 411,749
      - 4
      - 2
    * - LastFM
      - 1,980
      - 1,293,103
      - 2
      - 1

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `TemporalData`

A data object composed by a stream of events describing a temporal
graph.
The :class:`~torch_geometric.data.TemporalData` object can hold a list of
events (that can be understood as temporal edges in a graph) with
structured messages.
An event is composed by a source node, a destination node, a timestamp
and a message. Any *Continuous-Time Dynamic Graph* (CTDG) can be
represented with these four values.

In general, :class:`~torch_geometric.data.TemporalData` tries to mimic
the behavior of a regular :python:`Python` dictionary.
In addition, it provides useful functionality for analyzing graph
structures, and provides basic PyTorch tensor functionalities.

.. code-block:: python

    from torch import Tensor
    from torch_geometric.data import TemporalData

    events = TemporalData(
        src=Tensor([1,2,3,4]),
        dst=Tensor([2,3,4,5]),
        t=Tensor([1000,1010,1100,2000]),
        msg=Tensor([1,1,0,0])
    )

    # Add additional arguments to `events`:
    events.y = Tensor([1,1,0,0])

    # It is also possible to set additional arguments in the constructor
    events = TemporalData(
        ...,
        y=Tensor([1,1,0,0])
    )

    # Get the number of events:
    events.num_events
    >>> 4

    # Analyzing the graph structure:
    events.num_nodes
    >>> 5

    # PyTorch tensor functionality:
    events = events.pin_memory()
    events = events.to('cuda:0', non_blocking=True)

Args:
    src (torch.Tensor, optional): A list of source nodes for the events
        with shape :obj:`[num_events]`. (default: :obj:`None`)
    dst (torch.Tensor, optional): A list of destination nodes for the
        events with shape :obj:`[num_events]`. (default: :obj:`None`)
    t (torch.Tensor, optional): The timestamps for each event with shape
        :obj:`[num_events]`. (default: :obj:`None`)
    msg (torch.Tensor, optional): Messages feature matrix with shape
        :obj:`[num_events, num_msg_features]`. (default: :obj:`None`)
    **kwargs (optional): Additional attributes.

.. note::
    The shape of :obj:`src`, :obj:`dst`, :obj:`t` and the first dimension
    of :obj`msg` should be the same (:obj:`num_events`).

#### Methods

- **`index_select(self, idx: Any) -> 'TemporalData'`**

- **`stores_as(self, data: 'TemporalData')`**

- **`to_dict(self) -> Dict[str, Any]`**
  Returns a dictionary of stored key/value pairs.
