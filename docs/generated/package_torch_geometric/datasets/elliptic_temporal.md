# elliptic_temporal

Part of `torch_geometric.datasets`
Module: `torch_geometric.datasets.elliptic_temporal`

## Classes (3)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `EllipticBitcoinDataset`

The Elliptic Bitcoin dataset of Bitcoin transactions from the
`"Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional
Networks for Financial Forensics" <https://arxiv.org/abs/1908.02591>`_
paper.

:class:`EllipticBitcoinDataset` maps Bitcoin transactions to real entities
belonging to licit categories (exchanges, wallet providers, miners,
licit services, etc.) versus illicit ones (scams, malware, terrorist
organizations, ransomware, Ponzi schemes, etc.)

There exists 203,769 node transactions and 234,355 directed edge payments
flows, with two percent of nodes (4,545) labelled as illicit, and
twenty-one percent of nodes (42,019) labelled as licit.
The remaining transactions are unknown.

Args:
    root (str): Root directory where the dataset should be saved.
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
    :widths: 10 10 10 10
    :header-rows: 1

    * - #nodes
      - #edges
      - #features
      - #classes
    * - 203,769
      - 234,355
      - 165
      - 2

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `EllipticBitcoinTemporalDataset`

The time-step aware Elliptic Bitcoin dataset of Bitcoin transactions
from the `"Anti-Money Laundering in Bitcoin: Experimenting with Graph
Convolutional Networks for Financial Forensics"
<https://arxiv.org/abs/1908.02591>`_ paper.

:class:`EllipticBitcoinTemporalDataset` maps Bitcoin transactions to real
entities belonging to licit categories (exchanges, wallet providers,
miners, licit services, etc.) versus illicit ones (scams, malware,
terrorist organizations, ransomware, Ponzi schemes, etc.)

There exists 203,769 node transactions and 234,355 directed edge payments
flows, with two percent of nodes (4,545) labelled as illicit, and
twenty-one percent of nodes (42,019) labelled as licit.
The remaining transactions are unknown.

.. note::

    In contrast to :class:`EllipticBitcoinDataset`, this dataset returns
    Bitcoin transactions only for a given timestamp :obj:`t`.

Args:
    root (str): Root directory where the dataset should be saved.
    t (int): The Timestep for which nodes should be selected (from :obj:`1`
        to :obj:`49`).
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
    :widths: 10 10 10 10
    :header-rows: 1

    * - #nodes
      - #edges
      - #features
      - #classes
    * - 203,769
      - 234,355
      - 165
      - 2
