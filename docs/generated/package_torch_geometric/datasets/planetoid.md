# planetoid

Part of `torch_geometric.datasets`
Module: `torch_geometric.datasets.planetoid`

## Functions (1)

### `read_planetoid_data(folder: str, prefix: str) -> torch_geometric.data.data.Data`

## Classes (2)

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

### `Planetoid`

The citation network datasets :obj:`"Cora"`, :obj:`"CiteSeer"` and
:obj:`"PubMed"` from the `"Revisiting Semi-Supervised Learning with Graph
Embeddings" <https://arxiv.org/abs/1603.08861>`_ paper.
Nodes represent documents and edges represent citation links.
Training, validation and test splits are given by binary masks.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"Cora"`, :obj:`"CiteSeer"`,
        :obj:`"PubMed"`).
    split (str, optional): The type of dataset split (:obj:`"public"`,
        :obj:`"full"`, :obj:`"geom-gcn"`, :obj:`"random"`).
        If set to :obj:`"public"`, the split will be the public fixed split
        from the `"Revisiting Semi-Supervised Learning with Graph
        Embeddings" <https://arxiv.org/abs/1603.08861>`_ paper.
        If set to :obj:`"full"`, all nodes except those in the validation
        and test sets will be used for training (as in the
        `"FastGCN: Fast Learning with Graph Convolutional Networks via
        Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
        If set to :obj:`"geom-gcn"`, the 10 public fixed splits from the
        `"Geom-GCN: Geometric Graph Convolutional Networks"
        <https://openreview.net/forum?id=S1e2agrFvS>`_ paper are given.
        If set to :obj:`"random"`, train, validation, and test sets will be
        randomly generated, according to :obj:`num_train_per_class`,
        :obj:`num_val` and :obj:`num_test`. (default: :obj:`"public"`)
    num_train_per_class (int, optional): The number of training samples
        per class in case of :obj:`"random"` split. (default: :obj:`20`)
    num_val (int, optional): The number of validation samples in case of
        :obj:`"random"` split. (default: :obj:`500`)
    num_test (int, optional): The number of test samples in case of
        :obj:`"random"` split. (default: :obj:`1000`)
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
    * - Cora
      - 2,708
      - 10,556
      - 1,433
      - 7
    * - CiteSeer
      - 3,327
      - 9,104
      - 3,703
      - 6
    * - PubMed
      - 19,717
      - 88,648
      - 500
      - 3

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.
