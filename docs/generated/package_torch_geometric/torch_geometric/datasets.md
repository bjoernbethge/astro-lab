# datasets

Part of `torch_geometric.torch_geometric`
Module: `torch_geometric.datasets`

## Classes (107)

### `AMiner`

The heterogeneous AMiner dataset from the `"metapath2vec: Scalable
Representation Learning for Heterogeneous Networks"
<https://ericdongyx.github.io/papers/
KDD17-dong-chawla-swami-metapath2vec.pdf>`_ paper, consisting of nodes from
type :obj:`"paper"`, :obj:`"author"` and :obj:`"venue"`.
Venue categories and author research interests are available as ground
truth labels for a subset of nodes.

Args:
    root (str): Root directory where the dataset should be saved.
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `AQSOL`

The AQSOL dataset from the `Benchmarking Graph Neural Networks
<http://arxiv.org/abs/2003.00982>`_ paper based on
`AqSolDB <https://www.nature.com/articles/s41597-019-0151-1>`_, a
standardized database of 9,982 molecular graphs with their aqueous
solubility values, collected from 9 different data sources.

The aqueous solubility targets are collected from experimental measurements
and standardized to LogS units in AqSolDB. These final values denote the
property to regress in the :class:`AQSOL` dataset. After filtering out few
graphs with no bonds/edges, the total number of molecular graphs is 9,833.
For each molecular graph, the node features are the types of heavy atoms
and the edge features are the types of bonds between them, similar as in
the :class:`~torch_geometric.datasets.ZINC` dataset.

Args:
    root (str): Root directory where the dataset should be saved.
    split (str, optional): If :obj:`"train"`, loads the training dataset.
        If :obj:`"val"`, loads the validation dataset.
        If :obj:`"test"`, loads the test dataset.
        (default: :obj:`"train"`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in
        the final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - #graphs
      - #nodes
      - #edges
      - #features
      - #classes
    * - 9,833
      - ~17.6
      - ~35.8
      - 1
      - 1

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

- **`atoms(self) -> List[str]`**

### `Actor`

The actor-only induced subgraph of the film-director-actor-writer
network used in the
`"Geom-GCN: Geometric Graph Convolutional Networks"
<https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
Each node corresponds to an actor, and the edge between two nodes denotes
co-occurrence on the same Wikipedia page.
Node features correspond to some keywords in the Wikipedia pages.
The task is to classify the nodes into five categories in term of words of
actor's Wikipedia.

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
    * - 7,600
      - 30,019
      - 932
      - 5

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `AirfRANS`

The AirfRANS dataset from the `"AirfRANS: High Fidelity Computational
Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier-Stokes
Solutions" <https://arxiv.org/abs/2212.07564>`_ paper, consisting of 1,000
simulations of steady-state aerodynamics over 2D airfoils in a subsonic
flight regime.
The different tasks (:obj:`"full"`, :obj:`"scarce"`, :obj:`"reynolds"`,
:obj:`"aoa"`) define the utilized training and test splits.

Each simulation is given as a point cloud defined as the nodes of the
simulation mesh. Each point of a point cloud is described via 5
features: the inlet velocity (two components in meter per second), the
distance to the airfoil (one component in meter), and the normals (two
components in meter, set to :obj:`0` if the point is not on the airfoil).
Each point is given a target of 4 components for the underyling regression
task: the velocity (two components in meter per second), the pressure
divided by the specific mass (one component in meter squared per second
squared), the turbulent kinematic viscosity (one component in meter squared
per second).
Finaly, a boolean is attached to each point to inform if this point lies on
the airfoil or not.

A library for manipulating simulations of the dataset is available `here
<https://airfrans.readthedocs.io/en/latest/index.html>`_.

The dataset is released under the `ODbL v1.0 License
<https://opendatacommons.org/licenses/odbl/1-0/>`_.

.. note::

    Data objects contain no edge indices to be agnostic to the simulation
    mesh. You are free to build a graph via the
    :obj:`torch_geometric.transforms.RadiusGraph` transform.

Args:
    root (str): Root directory where the dataset should be saved.
    task (str): The task to study (:obj:`"full"`, :obj:`"scarce"`,
        :obj:`"reynolds"`, :obj:`"aoa"`) that defines the utilized training
        and test splits.
    train (bool, optional): If :obj:`True`, loads the training dataset,
        otherwise the test dataset. (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - #graphs
      - #nodes
      - #edges
      - #features
      - #tasks
    * - 1,000
      - ~180,000
      - 0
      - 5
      - 4

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `Airports`

The Airports dataset from the `"struc2vec: Learning Node
Representations from Structural Identity"
<https://arxiv.org/abs/1704.03165>`_ paper, where nodes denote airports
and labels correspond to activity levels.
Features are given by one-hot encoded node identifiers, as described in the
`"GraLSP: Graph Neural Networks with Local Structural Patterns"
` <https://arxiv.org/abs/1911.07675>`_ paper.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"USA"`, :obj:`"Brazil"`,
        :obj:`"Europe"`).
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

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `Amazon`

The Amazon Computers and Amazon Photo networks from the
`"Pitfalls of Graph Neural Network Evaluation"
<https://arxiv.org/abs/1811.05868>`_ paper.
Nodes represent goods and edges represent that two goods are frequently
bought together.
Given product reviews as bag-of-words node features, the task is to
map goods to their respective product category.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"Computers"`,
        :obj:`"Photo"`).
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
    * - Computers
      - 13,752
      - 491,722
      - 767
      - 10
    * - Photo
      - 7,650
      - 238,162
      - 745
      - 8

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `AmazonBook`

A subset of the AmazonBook rating dataset from the
`"LightGCN: Simplifying and Powering Graph Convolution Network for
Recommendation" <https://arxiv.org/abs/2002.02126>`_ paper.
This is a heterogeneous dataset consisting of 52,643 users and 91,599 books
with approximately 2.9 million ratings between them.
No labels or features are provided.

Args:
    root (str): Root directory where the dataset should be saved.
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `AmazonProducts`

The Amazon dataset from the `"GraphSAINT: Graph Sampling Based
Inductive Learning Method" <https://arxiv.org/abs/1907.04931>`_ paper,
containing products and its categories.

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
    * - 1,569,960
      - 264,339,468
      - 200
      - 107

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `AttributedGraphDataset`

A variety of attributed graph datasets from the
`"Scaling Attributed Network Embedding to Massive Graphs"
<https://arxiv.org/abs/2009.00826>`_ paper.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"Wiki"`, :obj:`"Cora"`
        :obj:`"CiteSeer"`, :obj:`"PubMed"`, :obj:`"BlogCatalog"`,
        :obj:`"PPI"`, :obj:`"Flickr"`, :obj:`"Facebook"`, :obj:`"Twitter"`,
        :obj:`"TWeibo"`, :obj:`"MAG"`).
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
    * - Wiki
      - 2,405
      - 17,981
      - 4,973
      - 17
    * - Cora
      - 2,708
      - 5,429
      - 1,433
      - 7
    * - CiteSeer
      - 3,312
      - 4,715
      - 3,703
      - 6
    * - PubMed
      - 19,717
      - 44,338
      - 500
      - 3
    * - BlogCatalog
      - 5,196
      - 343,486
      - 8,189
      - 6
    * - PPI
      - 56,944
      - 1,612,348
      - 50
      - 121
    * - Flickr
      - 7,575
      - 479,476
      - 12,047
      - 9
    * - Facebook
      - 4,039
      - 88,234
      - 1,283
      - 193
    * - TWeibo
      - 2,320,895
      - 9,840,066
      - 1,657
      - 8
    * - MAG
      - 59,249,719
      - 978,147,253
      - 2,000
      - 100

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `BA2MotifDataset`

The synthetic BA-2motifs graph classification dataset for evaluating
explainabilty algorithms, as described in the `"Parameterized Explainer
for Graph Neural Network" <https://arxiv.org/abs/2011.04573>`_ paper.
:class:`~torch_geometric.datasets.BA2MotifDataset` contains 1000 random
Barabasi-Albert (BA) graphs.
Half of the graphs are attached with a
:class:`~torch_geometric.datasets.motif_generator.HouseMotif`, and the rest
are attached with a five-node
:class:`~torch_geometric.datasets.motif_generator.CycleMotif`.
The graphs are assigned to one of the two classes according to the type of
attached motifs.

This dataset is pre-computed from the official implementation. If you want
to create own variations of it, you can make use of the
:class:`~torch_geometric.datasets.ExplainerDataset`:

.. code-block:: python

    import torch
    from torch_geometric.datasets import ExplainerDataset
    from torch_geometric.datasets.graph_generator import BAGraph
    from torch_geometric.datasets.motif_generator import HouseMotif
    from torch_geometric.datasets.motif_generator import CycleMotif

    dataset1 = ExplainerDataset(
        graph_generator=BAGraph(num_nodes=25, num_edges=1),
        motif_generator=HouseMotif(),
        num_motifs=1,
        num_graphs=500,
    )

    dataset2 = ExplainerDataset(
        graph_generator=BAGraph(num_nodes=25, num_edges=1),
        motif_generator=CycleMotif(5),
        num_motifs=1,
        num_graphs=500,
    )

    dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])

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
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - #graphs
      - #nodes
      - #edges
      - #features
      - #classes
    * - 1000
      - 25
      - ~51.0
      - 10
      - 2

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `BAMultiShapesDataset`

The synthetic BA-Multi-Shapes graph classification dataset for
evaluating explainabilty algorithms, as described in the
`"Global Explainability of GNNs via Logic Combination of Learned Concepts"
<https://arxiv.org/abs/2210.07147>`_ paper.

Given three atomic motifs, namely House (H), Wheel (W), and Grid (G),
:class:`~torch_geometric.datasets.BAMultiShapesDataset` contains 1,000
graphs where each graph is obtained by attaching the motifs to a random
Barabasi-Albert (BA) as follows:

* class 0: :math:`\emptyset \lor H \lor W \lor G \lor \{ H, W, G \}`

* class 1: :math:`(H \land W) \lor (H \land G) \lor (W \land G)`

This dataset is pre-computed from the official implementation.

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
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - #graphs
      - #nodes
      - #edges
      - #features
      - #classes
    * - 1000
      - 40
      - ~87.0
      - 10
      - 2

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `BAShapes`

The BA-Shapes dataset from the `"GNNExplainer: Generating Explanations
for Graph Neural Networks" <https://arxiv.org/abs/1903.03894>`__ paper,
containing a Barabasi-Albert (BA) graph with 300 nodes and a set of 80
"house"-structured graphs connected to it.

.. warning::

    :class:`BAShapes` is deprecated and will be removed in a future
    release. Use :class:`ExplainerDataset` in combination with
    :class:`torch_geometric.datasets.graph_generator.BAGraph` instead.

Args:
    connection_distribution (str, optional): Specifies how the houses
        and the BA graph get connected. Valid inputs are :obj:`"random"`
        (random BA graph nodes are selected for connection to the houses),
        and :obj:`"uniform"` (uniformly distributed BA graph nodes are
        selected for connection to the houses). (default: :obj:`"random"`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)

### `BitcoinOTC`

The Bitcoin-OTC dataset from the `"EvolveGCN: Evolving Graph
Convolutional Networks for Dynamic Graphs"
<https://arxiv.org/abs/1902.10191>`_ paper, consisting of 138
who-trusts-whom networks of sequential time steps.

Args:
    root (str): Root directory where the dataset should be saved.
    edge_window_size (int, optional): The window size for the existence of
        an edge in the graph sequence since its initial creation.
        (default: :obj:`10`)
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

    * - #graphs
      - #nodes
      - #edges
      - #features
      - #classes
    * - 138
      - 6,005
      - ~2,573.2
      - 0
      - 0

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `BrcaTcga`

The breast cancer (BRCA TCGA Pan-Cancer Atlas) dataset consisting of
patients with survival information and gene expression data from
`cBioPortal <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4160307/>`_
and a network of biological interactions between those nodes from
`Pathway Commons <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7145667/>`_.
The dataset contains the gene features of 1,082 patients, and the overall
survival time (in months) of each patient as label.

Pre-processing and example model codes on how to use this dataset can be
found `here <https://github.com/cannin/pyg_pathway_commons_cbioportal>`_.

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
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10
    :header-rows: 1

    * - #graphs
      - #nodes
      - #edges
      - #features
    * - 1,082
      - 9,288
      - 271,771
      - 1,082

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `CitationFull`

The full citation network datasets from the
`"Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via
Ranking" <https://arxiv.org/abs/1707.03815>`_ paper.
Nodes represent documents and edges represent citation links.
Datasets include :obj:`"Cora"`, :obj:`"Cora_ML"`, :obj:`"CiteSeer"`,
:obj:`"DBLP"`, :obj:`"PubMed"`.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"Cora"`, :obj:`"Cora_ML"`
        :obj:`"CiteSeer"`, :obj:`"DBLP"`, :obj:`"PubMed"`).
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    to_undirected (bool, optional): Whether the original graph is
        converted to an undirected one. (default: :obj:`True`)
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
      - 19,793
      - 126,842
      - 8,710
      - 70
    * - Cora_ML
      - 2,995
      - 16,316
      - 2,879
      - 7
    * - CiteSeer
      - 4,230
      - 10,674
      - 602
      - 6
    * - DBLP
      - 17,716
      - 105,734
      - 1,639
      - 4
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

### `CoMA`

The CoMA 3D faces dataset from the `"Generating 3D faces using
Convolutional Mesh Autoencoders" <https://arxiv.org/abs/1807.10267>`_
paper, containing 20,466 meshes of extreme expressions captured over 12
different subjects.

.. note::

    Data objects hold mesh faces instead of edge indices.
    To convert the mesh to a graph, use the
    :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
    To convert the mesh to a point cloud, use the
    :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
    sample a fixed number of points on the mesh faces according to their
    face area.

Args:
    root (str): Root directory where the dataset should be saved.
    train (bool, optional): If :obj:`True`, loads the training dataset,
        otherwise the test dataset. (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - #graphs
      - #nodes
      - #edges
      - #features
      - #classes
    * - 20,465
      - 5,023
      - 29,990
      - 3
      - 12

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `Coauthor`

The Coauthor CS and Coauthor Physics networks from the
`"Pitfalls of Graph Neural Network Evaluation"
<https://arxiv.org/abs/1811.05868>`_ paper.
Nodes represent authors that are connected by an edge if they co-authored a
paper.
Given paper keywords for each author's papers, the task is to map authors
to their respective field of study.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"CS"`, :obj:`"Physics"`).
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
    * - CS
      - 18,333
      - 163,788
      - 6,805
      - 15
    * - Physics
      - 34,493
      - 495,924
      - 8,415
      - 5

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `CoraFull`

Alias for :class:`~torch_geometric.datasets.CitationFull` with
:obj:`name="Cora"`.

**STATS:**

.. list-table::
    :widths: 10 10 10 10
    :header-rows: 1

    * - #nodes
      - #edges
      - #features
      - #classes
    * - 19,793
      - 126,842
      - 8,710
      - 70

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `CornellTemporalHyperGraphDataset`

A collection of temporal higher-order network datasets from the
`"Simplicial Closure and higher-order link prediction"
<https://arxiv.org/abs/1802.06916>`_ paper.
Each of the datasets is a timestamped sequence of simplices, where a
simplex is a set of :math:`k` nodes.

See the original `datasets page
<https://www.cs.cornell.edu/~arb/data/>`_ for more details about
individual datasets.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset.
    split (str, optional): If :obj:`"train"`, loads the training dataset.
        If :obj:`"val"`, loads the validation dataset.
        If :obj:`"test"`, loads the test dataset.
        (default: :obj:`"train"`)
    setting (str, optional): If :obj:`"transductive"`, loads the dataset
        for transductive training.
        If :obj:`"inductive"`, loads the dataset for inductive training.
        (default: :obj:`"transductive"`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `DBLP`

A subset of the DBLP computer science bibliography website, as
collected in the `"MAGNN: Metapath Aggregated Graph Neural Network for
Heterogeneous Graph Embedding" <https://arxiv.org/abs/2002.01680>`_ paper.
DBLP is a heterogeneous graph containing four types of entities - authors
(4,057 nodes), papers (14,328 nodes), terms (7,723 nodes), and conferences
(20 nodes).
The authors are divided into four research areas (database, data mining,
artificial intelligence, information retrieval).
Each author is described by a bag-of-words representation of their paper
keywords.

Args:
    root (str): Root directory where the dataset should be saved.
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 20 10 10 10
    :header-rows: 1

    * - Node/Edge Type
      - #nodes/#edges
      - #features
      - #classes
    * - Author
      - 4,057
      - 334
      - 4
    * - Paper
      - 14,328
      - 4,231
      -
    * - Term
      - 7,723
      - 50
      -
    * - Conference
      - 20
      - 0
      -
    * - Author-Paper
      - 196,425
      -
      -
    * - Paper-Term
      - 85,810
      -
      -
    * - Conference-Paper
      - 14,328
      -
      -

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `DBP15K`

The DBP15K dataset from the
`"Cross-lingual Entity Alignment via Joint Attribute-Preserving Embedding"
<https://arxiv.org/abs/1708.05045>`_ paper, where Chinese, Japanese and
French versions of DBpedia were linked to its English version.
Node features are given by pre-trained and aligned monolingual word
embeddings from the `"Cross-lingual Knowledge Graph Alignment via Graph
Matching Neural Network" <https://arxiv.org/abs/1905.11605>`_ paper.

Args:
    root (str): Root directory where the dataset should be saved.
    pair (str): The pair of languages (:obj:`"en_zh"`, :obj:`"en_fr"`,
        :obj:`"en_ja"`, :obj:`"zh_en"`, :obj:`"fr_en"`, :obj:`"ja_en"`).
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

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

- **`process_graph(self, triple_path: str, feature_path: str, embeddings: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`**

### `DGraphFin`

The DGraphFin networks from the
`"DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection"
<https://arxiv.org/abs/2207.03579>`_ paper.
It is a directed, unweighted dynamic graph consisting of millions of
nodes and edges, representing a realistic user-to-user social network
in financial industry.
Node represents a Finvolution user, and an edge from one
user to another means that the user regards the other user
as the emergency contact person. Each edge is associated with a
timestamp ranging from 1 to 821 and a type of emergency contact
ranging from 0 to 11.


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
    * - 3,700,550
      - 4,300,999
      - 17
      - 2

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `DeezerEurope`

The Deezer Europe dataset introduced in the `"Characteristic Functions
on Graphs: Birds of a Feather, from Statistical Descriptors to Parametric
Models" <https://arxiv.org/abs/2005.07959>`_ paper.
Nodes represent European users of Deezer and edges are mutual follower
relationships.
It contains 28,281 nodes, 185,504 edges, 128 node features and 2 classes.

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

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `DynamicFAUST`

The dynamic FAUST humans dataset from the `"Dynamic FAUST: Registering
Human Bodies in Motion"
<http://files.is.tue.mpg.de/black/papers/dfaust2017.pdf>`_ paper.

.. note::

    Data objects hold mesh faces instead of edge indices.
    To convert the mesh to a graph, use the
    :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
    To convert the mesh to a point cloud, use the
    :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
    sample a fixed number of points on the mesh faces according to their
    face area.

Args:
    root (str): Root directory where the dataset should be saved.
    subjects (list, optional): List of subjects to include in the
        dataset. Can include the subjects :obj:`"50002"`, :obj:`"50004"`,
        :obj:`"50007"`, :obj:`"50009"`, :obj:`"50020"`, :obj:`"50021"`,
        :obj:`"50022"`, :obj:`"50025"`, :obj:`"50026"`, :obj:`"50027"`.
        If set to :obj:`None`, the dataset will contain all subjects.
        (default: :obj:`None`)
    categories (list, optional): List of categories to include in the
        dataset. Can include the categories :obj:`"chicken_wings"`,
        :obj:`"hips"`, :obj:`"jiggle_on_toes"`, :obj:`"jumping_jacks"`,
        :obj:`"knees"`, :obj:`"light_hopping_loose"`,
        :obj:`"light_hopping_stiff"`, :obj:`"one_leg_jump"`,
        :obj:`"one_leg_loose"`, :obj:`"personal_move"`, :obj:`"punching"`,
        :obj:`"running_on_spot"`, :obj:`"running_on_spot_bugfix"`,
        :obj:`"shake_arms"`, :obj:`"shake_hips"`, :obj:`"shoulders"`.
        If set to :obj:`None`, the dataset will contain all categories.
        (default: :obj:`None`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

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

### `EmailEUCore`

An e-mail communication network of a large European research
institution, taken from the `"Local Higher-order Graph Clustering"
<https://www-cs.stanford.edu/~jure/pubs/mappr-kdd17.pdf>`_ paper.
Nodes indicate members of the institution.
An edge between a pair of members indicates that they exchanged at least
one email.
Node labels indicate membership to one of the 42 departments.

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

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `Entities`

The relational entities networks :obj:`"AIFB"`, :obj:`"MUTAG"`,
:obj:`"BGS"` and :obj:`"AM"` from the `"Modeling Relational Data with Graph
Convolutional Networks" <https://arxiv.org/abs/1703.06103>`_ paper.
Training and test splits are given by node indices.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"AIFB"`, :obj:`"MUTAG"`,
        :obj:`"BGS"`, :obj:`"AM"`).
    hetero (bool, optional): If set to :obj:`True`, will save the dataset
        as a :class:`~torch_geometric.data.HeteroData` object.
        (default: :obj:`False`)
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
    * - AIFB
      - 8,285
      - 58,086
      - 0
      - 4
    * - AM
      - 1,666,764
      - 11,976,642
      - 0
      - 11
    * - MUTAG
      - 23,644
      - 148,454
      - 0
      - 2
    * - BGS
      - 333,845
      - 1,832,398
      - 0
      - 2

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `ExplainerDataset`

Generates a synthetic dataset for evaluating explainabilty algorithms,
as described in the `"GNNExplainer: Generating Explanations for Graph
Neural Networks" <https://arxiv.org/abs/1903.03894>`__ paper.
The :class:`~torch_geometric.datasets.ExplainerDataset` creates synthetic
graphs coming from a
:class:`~torch_geometric.datasets.graph_generator.GraphGenerator`, and
randomly attaches :obj:`num_motifs` many motifs to it coming from a
:class:`~torch_geometric.datasets.graph_generator.MotifGenerator`.
Ground-truth node-level and edge-level explainabilty masks are given based
on whether nodes and edges are part of a certain motif or not.

For example, to generate a random Barabasi-Albert (BA) graph with 300
nodes, in which we want to randomly attach 80 :obj:`"house"` motifs, write:

.. code-block:: python

    from torch_geometric.datasets import ExplainerDataset
    from torch_geometric.datasets.graph_generator import BAGraph

    dataset = ExplainerDataset(
        graph_generator=BAGraph(num_nodes=300, num_edges=5),
        motif_generator='house',
        num_motifs=80,
    )

.. note::

    For an example of using :class:`ExplainerDataset`, see
    `examples/explain/gnn_explainer_ba_shapes.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    /explain/gnn_explainer_ba_shapes.py>`_.

Args:
    graph_generator (GraphGenerator or str): The graph generator to be
        used, *e.g.*,
        :class:`torch.geometric.datasets.graph_generator.BAGraph`
        (or any string that automatically resolves to it).
    motif_generator (MotifGenerator): The motif generator to be used,
        *e.g.*,
        :class:`torch_geometric.datasets.motif_generator.HouseMotif`
        (or any string that automatically resolves to it).
    num_motifs (int): The number of motifs to attach to the graph.
    num_graphs (int, optional): The number of graphs to generate.
        (default: :obj:`1`)
    graph_generator_kwargs (Dict[str, Any], optional): Arguments passed to
        the respective graph generator module in case it gets automatically
        resolved. (default: :obj:`None`)
    motif_generator_kwargs (Dict[str, Any], optional): Arguments passed to
        the respective motif generator module in case it gets automatically
        resolved. (default: :obj:`None`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)

#### Methods

- **`get_graph(self) -> torch_geometric.explain.explanation.Explanation`**

### `FAUST`

The FAUST humans dataset from the `"FAUST: Dataset and Evaluation for
3D Mesh Registration"
<http://files.is.tue.mpg.de/black/papers/FAUST2014.pdf>`_ paper,
containing 100 watertight meshes representing 10 different poses for 10
different subjects.

.. note::

    Data objects hold mesh faces instead of edge indices.
    To convert the mesh to a graph, use the
    :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
    To convert the mesh to a point cloud, use the
    :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
    sample a fixed number of points on the mesh faces according to their
    face area.

Args:
    root (str): Root directory where the dataset should be saved.
    train (bool, optional): If :obj:`True`, loads the training dataset,
        otherwise the test dataset. (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - #graphs
      - #nodes
      - #edges
      - #features
      - #classes
    * - 100
      - 6,890
      - 41,328
      - 3
      - 10

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `FB15k_237`

The FB15K237 dataset from the `"Translating Embeddings for Modeling
Multi-Relational Data"
<https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling
-multi-relational-data>`_ paper,
containing 14,541 entities, 237 relations and 310,116 fact triples.

.. note::

    The original :class:`FB15k` dataset suffers from major test leakage
    through inverse relations, where a large number of test triples could
    be obtained by inverting triples in the training set.
    In order to create a dataset without this characteristic, the
    :class:`~torch_geometric.datasets.FB15k_237` describes a subset of
    :class:`FB15k` where inverse relations are removed.

Args:
    root (str): Root directory where the dataset should be saved.
    split (str, optional): If :obj:`"train"`, loads the training dataset.
        If :obj:`"val"`, loads the validation dataset.
        If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
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

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `FacebookPagePage`

The Facebook Page-Page network dataset introduced in the
`"Multi-scale Attributed Node Embedding"
<https://arxiv.org/abs/1909.13021>`_ paper.
Nodes represent verified pages on Facebook and edges are mutual likes.
It contains 22,470 nodes, 342,004 edges, 128 node features and 4 classes.

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

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `FakeDataset`

A fake dataset that returns randomly generated
:class:`~torch_geometric.data.Data` objects.

Args:
    num_graphs (int, optional): The number of graphs. (default: :obj:`1`)
    avg_num_nodes (int, optional): The average number of nodes in a graph.
        (default: :obj:`1000`)
    avg_degree (float, optional): The average degree per node.
        (default: :obj:`10.0`)
    num_channels (int, optional): The number of node features.
        (default: :obj:`64`)
    edge_dim (int, optional): The number of edge features.
        (default: :obj:`0`)
    num_classes (int, optional): The number of classes in the dataset.
        (default: :obj:`10`)
    task (str, optional): Whether to return node-level or graph-level
        labels (:obj:`"node"`, :obj:`"graph"`, :obj:`"auto"`).
        If set to :obj:`"auto"`, will return graph-level labels if
        :obj:`num_graphs > 1`, and node-level labels other-wise.
        (default: :obj:`"auto"`)
    is_undirected (bool, optional): Whether the graphs to generate are
        undirected. (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    **kwargs (optional): Additional attributes and their shapes
        *e.g.* :obj:`global_features=5`.

#### Methods

- **`generate_data(self) -> torch_geometric.data.data.Data`**

### `FakeHeteroDataset`

A fake dataset that returns randomly generated
:class:`~torch_geometric.data.HeteroData` objects.

Args:
    num_graphs (int, optional): The number of graphs. (default: :obj:`1`)
    num_node_types (int, optional): The number of node types.
        (default: :obj:`3`)
    num_edge_types (int, optional): The number of edge types.
        (default: :obj:`6`)
    avg_num_nodes (int, optional): The average number of nodes in a graph.
        (default: :obj:`1000`)
    avg_degree (float, optional): The average degree per node.
        (default: :obj:`10.0`)
    avg_num_channels (int, optional): The average number of node features.
        (default: :obj:`64`)
    edge_dim (int, optional): The number of edge features.
        (default: :obj:`0`)
    num_classes (int, optional): The number of classes in the dataset.
        (default: :obj:`10`)
    task (str, optional): Whether to return node-level or graph-level
        labels (:obj:`"node"`, :obj:`"graph"`, :obj:`"auto"`).
        If set to :obj:`"auto"`, will return graph-level labels if
        :obj:`num_graphs > 1`, and node-level labels other-wise.
        (default: :obj:`"auto"`)
    transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    **kwargs (optional): Additional attributes and their shapes
        *e.g.* :obj:`global_features=5`.

#### Methods

- **`generate_data(self) -> torch_geometric.data.hetero_data.HeteroData`**

### `Flickr`

The Flickr dataset from the `"GraphSAINT: Graph Sampling Based
Inductive Learning Method" <https://arxiv.org/abs/1907.04931>`_ paper,
containing descriptions and common properties of images.

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
    * - 89,250
      - 899,756
      - 500
      - 7

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `GDELT`

The Global Database of Events, Language, and Tone (GDELT) dataset used
in the, *e.g.*, `"Recurrent Event Network for Reasoning over Temporal
Knowledge Graphs" <https://arxiv.org/abs/1904.05530>`_ paper, consisting of
events collected from 1/1/2018 to 1/31/2018 (15 minutes time granularity).

Args:
    root (str): Root directory where the dataset should be saved.
    split (str, optional): If :obj:`"train"`, loads the training dataset.
        If :obj:`"val"`, loads the validation dataset.
        If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process_events(self) -> torch.Tensor`**

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `GDELTLite`

The (reduced) version of the Global Database of Events, Language, and
Tone (GDELT) dataset used in the `"Do We Really Need Complicated Model
Architectures for Temporal Networks?" <http://arxiv.org/abs/2302.11636>`_
paper, consisting of events collected from 2016 to 2020.

Each node (actor) holds a 413-dimensional multi-hot feature vector that
represents CAMEO codes attached to the corresponding actor to server.

Each edge (event) holds a timestamp and a 186-dimensional multi-hot vector
representing CAMEO codes attached to the corresponding event to server.

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
    * - 8,831
      - 1,912,909
      - 413
      -

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `GEDDataset`

The GED datasets from the `"Graph Edit Distance Computation via Graph
Neural Networks" <https://arxiv.org/abs/1808.05689>`_ paper.

GEDs can be accessed via the global attributes :obj:`ged` and
:obj:`norm_ged` for all train/train graph pairs and all train/test graph
pairs:

.. code-block:: python

    dataset = GEDDataset(root, name="LINUX")
    data1, data2 = dataset[0], dataset[1]
    ged = dataset.ged[data1.i, data2.i]  # GED between `data1` and `data2`.

Note that GEDs are not available if both graphs are from the test set.
For evaluation, it is recommended to pair up each graph from the test set
with each graph in the training set.

.. note::

    :obj:`ALKANE` is missing GEDs for train/test graph pairs since they are
    not provided in the `official datasets
    <https://github.com/yunshengb/SimGNN>`_.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (one of :obj:`"AIDS700nef"`,
        :obj:`"LINUX"`, :obj:`"ALKANE"`, :obj:`"IMDBMulti"`).
    train (bool, optional): If :obj:`True`, loads the training dataset,
        otherwise the test dataset. (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 20 10 10 10 10 10
    :header-rows: 1

    * - Name
      - #graphs
      - #nodes
      - #edges
      - #features
      - #classes
    * - AIDS700nef
      - 700
      - ~8.9
      - ~17.6
      - 29
      - 0
    * - LINUX
      - 1,000
      - ~7.6
      - ~13.9
      - 0
      - 0
    * - ALKANE
      - 150
      - ~8.9
      - ~15.8
      - 0
      - 0
    * - IMDBMulti
      - 1,500
      - ~13.0
      - ~131.9
      - 0
      - 0

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `GNNBenchmarkDataset`

A variety of artificially and semi-artificially generated graph
datasets from the `"Benchmarking Graph Neural Networks"
<https://arxiv.org/abs/2003.00982>`_ paper.

.. note::
    The ZINC dataset is provided via
    :class:`torch_geometric.datasets.ZINC`.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (one of :obj:`"PATTERN"`,
        :obj:`"CLUSTER"`, :obj:`"MNIST"`, :obj:`"CIFAR10"`,
        :obj:`"TSP"`, :obj:`"CSL"`)
    split (str, optional): If :obj:`"train"`, loads the training dataset.
        If :obj:`"val"`, loads the validation dataset.
        If :obj:`"test"`, loads the test dataset.
        (default: :obj:`"train"`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 20 10 10 10 10 10
    :header-rows: 1

    * - Name
      - #graphs
      - #nodes
      - #edges
      - #features
      - #classes
    * - PATTERN
      - 14,000
      - ~118.9
      - ~6,098.9
      - 3
      - 2
    * - CLUSTER
      - 12,000
      - ~117.2
      - ~4,303.9
      - 7
      - 6
    * - MNIST
      - 70,000
      - ~70.6
      - ~564.5
      - 3
      - 10
    * - CIFAR10
      - 60,000
      - ~117.6
      - ~941.2
      - 5
      - 10
    * - TSP
      - 12,000
      - ~275.4
      - ~6,885.0
      - 2
      - 2
    * - CSL
      - 150
      - ~41.0
      - ~164.0
      - 0
      - 10

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

- **`process_CSL(self) -> List[torch_geometric.data.data.Data]`**

### `GemsecDeezer`

The Deezer User Network datasets introduced in the
`"GEMSEC: Graph Embedding with Self Clustering"
<https://arxiv.org/abs/1802.03997>`_ paper.
Nodes represent Deezer user and edges are mutual friendships.
The task is multi-label multi-class node classification about
the genres liked by the users.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"HU"`, :obj:`"HR"`,
        :obj:`"RO"`).
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

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `GeometricShapes`

Synthetic dataset of various geometric shapes like cubes, spheres or
pyramids.

.. note::

    Data objects hold mesh faces instead of edge indices.
    To convert the mesh to a graph, use the
    :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
    To convert the mesh to a point cloud, use the
    :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
    sample a fixed number of points on the mesh faces according to their
    face area.

Args:
    root (str): Root directory where the dataset should be saved.
    train (bool, optional): If :obj:`True`, loads the training dataset,
        otherwise the test dataset. (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - #graphs
      - #nodes
      - #edges
      - #features
      - #classes
    * - 80
      - ~148.8
      - ~859.5
      - 3
      - 40

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

- **`process_set(self, dataset: str) -> List[torch_geometric.data.data.Data]`**

### `GitHub`

The GitHub Web and ML Developers dataset introduced in the
`"Multi-scale Attributed Node Embedding"
<https://arxiv.org/abs/1909.13021>`_ paper.
Nodes represent developers on :obj:`github:`GitHub` and edges are mutual
follower relationships.
It contains 37,300 nodes, 578,006 edges, 128 node features and 2 classes.

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
    * - 37,700
      - 578,006
      - 0
      - 2

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `HGBDataset`

A variety of heterogeneous graph benchmark datasets from the
`"Are We Really Making Much Progress? Revisiting, Benchmarking, and
Refining Heterogeneous Graph Neural Networks"
<http://keg.cs.tsinghua.edu.cn/jietang/publications/
KDD21-Lv-et-al-HeterGNN.pdf>`_ paper.

.. note::
    Test labels are randomly given to prevent data leakage issues.
    If you want to obtain final test performance, you will need to submit
    your model predictions to the
    `HGB leaderboard <https://www.biendata.xyz/hgb/>`_.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (one of :obj:`"ACM"`,
        :obj:`"DBLP"`, :obj:`"Freebase"`, :obj:`"IMDB"`)
    transform (callable, optional): A function/transform that takes in an
        :class:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :class:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `HM`

The heterogeneous H&M dataset from the `Kaggle H&M Personalized Fashion
Recommendations
<https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations>`_
challenge.
The task is to develop product recommendations based on data from previous
transactions, as well as from customer and product meta data.

Args:
    root (str): Root directory where the dataset should be saved.
    use_all_tables_as_node_types (bool, optional): If set to :obj:`True`,
        will use the transaction table as a distinct node type.
        (default: :obj:`False`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `HeterophilousGraphDataset`

The heterophilous graphs :obj:`"Roman-empire"`,
:obj:`"Amazon-ratings"`, :obj:`"Minesweeper"`, :obj:`"Tolokers"` and
:obj:`"Questions"` from the `"A Critical Look at the Evaluation of GNNs
under Heterophily: Are We Really Making Progress?"
<https://arxiv.org/abs/2302.11640>`_ paper.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"Roman-empire"`,
        :obj:`"Amazon-ratings"`, :obj:`"Minesweeper"`, :obj:`"Tolokers"`,
        :obj:`"Questions"`).
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
    * - Roman-empire
      - 22,662
      - 32,927
      - 300
      - 18
    * - Amazon-ratings
      - 24,492
      - 93,050
      - 300
      - 5
    * - Minesweeper
      - 10,000
      - 39,402
      - 7
      - 2
    * - Tolokers
      - 11,758
      - 519,000
      - 10
      - 2
    * - Questions
      - 48,921
      - 153,540
      - 301
      - 2

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `HydroNet`

The HydroNet dataest from the
`"HydroNet: Benchmark Tasks for Preserving Intermolecular Interactions and
Structural Motifs in Predictive and Generative Models for Molecular Data"
<https://arxiv.org/abs/2012.00131>`_ paper, consisting of 5 million water
clusters held together by hydrogen bonding networks.  This dataset
provides atomic coordinates and total energy in kcal/mol for the cluster.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str, optional): Name of the subset of the full dataset to use:
        :obj:`"small"` uses 500k graphs sampled from the :obj:`"medium"`
        dataset, :obj:`"medium"` uses 2.7m graphs with maximum size of 75
        nodes.
        Mutually exclusive option with the clusters argument.
        (default :obj:`None`)
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
    num_workers (int): Number of multiprocessing workers to use for
        pre-processing the dataset. (default :obj:`8`)
    clusters (int or List[int], optional): Select a subset of clusters
        from the full dataset. If set to :obj:`None`, will select all.
        (default :obj:`None`)
    use_processed (bool): Option to use a pre-processed version of the
        original :obj:`xyz` dataset. (default: :obj:`True`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

- **`select_clusters(self, clusters: Union[int, List[int], NoneType]) -> None`**

### `ICEWS18`

The Integrated Crisis Early Warning System (ICEWS) dataset used in
the, *e.g.*, `"Recurrent Event Network for Reasoning over Temporal
Knowledge Graphs" <https://arxiv.org/abs/1904.05530>`_ paper, consisting of
events collected from 1/1/2018 to 10/31/2018 (24 hours time granularity).

Args:
    root (str): Root directory where the dataset should be saved.
    split (str, optional): If :obj:`"train"`, loads the training dataset.
        If :obj:`"val"`, loads the validation dataset.
        If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process_events(self) -> torch.Tensor`**

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `IGMCDataset`

The user-item heterogeneous rating datasets :obj:`"Douban"`,
:obj:`"Flixster"` and :obj:`"Yahoo-Music"` from the `"Inductive Matrix
Completion Based on Graph Neural Networks"
<https://arxiv.org/abs/1904.12058>`_ paper.

Nodes represent users and items.
Edges and features between users and items represent a (training) rating of
the item given by the user.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"Douban"`,
        :obj:`"Flixster"`, :obj:`"Yahoo-Music"`).
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`load_matlab_file(path_file: str, name: str) -> torch.Tensor`**

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `IMDB`

A subset of the Internet Movie Database (IMDB), as collected in the
`"MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph
Embedding" <https://arxiv.org/abs/2002.01680>`_ paper.
IMDB is a heterogeneous graph containing three types of entities - movies
(4,278 nodes), actors (5,257 nodes), and directors (2,081 nodes).
The movies are divided into three classes (action, comedy, drama) according
to their genre.
Movie features correspond to elements of a bag-of-words representation of
its plot keywords.

Args:
    root (str): Root directory where the dataset should be saved.
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `InfectionDataset`

Generates a synthetic infection dataset for evaluating explainabilty
algorithms, as described in the `"Explainability Techniques for Graph
Convolutional Networks" <https://arxiv.org/abs/1905.13686>`__ paper.
The :class:`~torch_geometric.datasets.InfectionDataset` creates synthetic
graphs coming from a
:class:`~torch_geometric.datasets.graph_generator.GraphGenerator` with
:obj:`num_infected` randomly assigned infected nodes.
The dataset describes a node classification task of predicting the length
of the shortest path to infected nodes, with corresponding ground-truth
edge-level masks.

For example, to generate a random Erdos-Renyi (ER) infection graph
with :obj:`500` nodes and :obj:`0.004` edge probability, write:

.. code-block:: python

    from torch_geometric.datasets import InfectionDataset
    from torch_geometric.datasets.graph_generator import ERGraph

    dataset = InfectionDataset(
        graph_generator=ERGraph(num_nodes=500, edge_prob=0.004),
        num_infected_nodes=50,
        max_path_length=3,
    )

Args:
    graph_generator (GraphGenerator or str): The graph generator to be
        used, *e.g.*,
        :class:`torch.geometric.datasets.graph_generator.BAGraph`
        (or any string that automatically resolves to it).
    num_infected_nodes (int or List[int]): The number of randomly
        selected infected nodes in the graph.
        If given as a list, will select a different number of infected
        nodes for different graphs.
    max_path_length (int, List[int]): The maximum shortest path length to
        determine whether a node will be infected.
        If given as a list, will apply different shortest path lengths for
        different graphs. (default: :obj:`5`)
    num_graphs (int, optional): The number of graphs to generate.
        The number of graphs will be automatically determined by
        :obj:`len(num_infected_nodes)` or :obj:`len(max_path_length)` in
        case either of them is given as a list, and should only be set in
        case one wants to create multiple graphs while
        :obj:`num_infected_nodes` and :obj:`max_path_length` are given as
        an integer. (default: :obj:`None`)
    graph_generator_kwargs (Dict[str, Any], optional): Arguments passed to
        the respective graph generator module in case it gets automatically
        resolved. (default: :obj:`None`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)

#### Methods

- **`get_graph(self, num_infected_nodes: int, max_path_length: int) -> torch_geometric.explain.explanation.Explanation`**

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

### `KarateClub`

Zachary's karate club network from the `"An Information Flow Model for
Conflict and Fission in Small Groups"
<https://www.journals.uchicago.edu/doi/abs/10.1086/jar.33.4.3629752>`_
paper, containing 34 nodes,
connected by 156 (undirected and unweighted) edges.
Every node is labeled by one of four classes obtained via modularity-based
clustering, following the `"Semi-supervised Classification with Graph
Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_ paper.
Training is based on a single labeled example per class, *i.e.* a total
number of 4 labeled nodes.

Args:
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10
    :header-rows: 1

    * - #nodes
      - #edges
      - #features
      - #classes
    * - 34
      - 156
      - 34
      - 4

### `LINKXDataset`

A variety of non-homophilous graph datasets from the `"Large Scale
Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple
Methods" <https://arxiv.org/abs/2110.14446>`_ paper.

.. note::
    Some of the datasets provided in :class:`LINKXDataset` are from other
    sources, but have been updated with new features and/or labels.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"penn94"`, :obj:`"reed98"`,
        :obj:`"amherst41"`, :obj:`"cornell5"`, :obj:`"johnshopkins55"`,
        :obj:`"genius"`).
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

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `LRGBDataset`

The `"Long Range Graph Benchmark (LRGB)"
<https://arxiv.org/abs/2206.08164>`_
datasets which is a collection of 5 graph learning datasets with tasks
that are based on long-range dependencies in graphs. See the original
`source code <https://github.com/vijaydwivedi75/lrgb>`_ for more details
on the individual datasets.

+------------------------+-------------------+----------------------+
| Dataset                | Domain            | Task                 |
+========================+===================+======================+
| :obj:`PascalVOC-SP`    | Computer Vision   | Node Classification  |
+------------------------+-------------------+----------------------+
| :obj:`COCO-SP`         | Computer Vision   | Node Classification  |
+------------------------+-------------------+----------------------+
| :obj:`PCQM-Contact`    | Quantum Chemistry | Link Prediction      |
+------------------------+-------------------+----------------------+
| :obj:`Peptides-func`   | Chemistry         | Graph Classification |
+------------------------+-------------------+----------------------+
| :obj:`Peptides-struct` | Chemistry         | Graph Regression     |
+------------------------+-------------------+----------------------+

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (one of :obj:`"PascalVOC-SP"`,
        :obj:`"COCO-SP"`, :obj:`"PCQM-Contact"`, :obj:`"Peptides-func"`,
        :obj:`"Peptides-struct"`)
    split (str, optional): If :obj:`"train"`, loads the training dataset.
        If :obj:`"val"`, loads the validation dataset.
        If :obj:`"test"`, loads the test dataset.
        (default: :obj:`"train"`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 15 10 10 10 10
    :header-rows: 1

    * - Name
      - #graphs
      - #nodes
      - #edges
      - #classes
    * - PascalVOC-SP
      - 11,355
      - ~479.40
      - ~2,710.48
      - 21
    * - COCO-SP
      - 123,286
      - ~476.88
      - ~2,693.67
      - 81
    * - PCQM-Contact
      - 529,434
      - ~30.14
      - ~61.09
      - 1
    * - Peptides-func
      - 15,535
      - ~150.94
      - ~307.30
      - 10
    * - Peptides-struct
      - 15,535
      - ~150.94
      - ~307.30
      - 11

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

- **`label_remap_coco(self) -> Dict[int, int]`**

### `LastFM`

A subset of the last.fm music website keeping track of users' listining
information from various sources, as collected in the
`"MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph
Embedding" <https://arxiv.org/abs/2002.01680>`_ paper.
last.fm is a heterogeneous graph containing three types of entities - users
(1,892 nodes), artists (17,632 nodes), and artist tags (1,088 nodes).
This dataset can be used for link prediction, and no labels or features are
provided.

Args:
    root (str): Root directory where the dataset should be saved.
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `LastFMAsia`

The LastFM Asia Network dataset introduced in the `"Characteristic
Functions on Graphs: Birds of a Feather, from Statistical Descriptors to
Parametric Models" <https://arxiv.org/abs/2005.07959>`_ paper.
Nodes represent LastFM users from Asia and edges are friendships.
It contains 7,624 nodes, 55,612 edges, 128 node features and 18 classes.

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

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `MD17`

A variety of ab-initio molecular dynamics trajectories from the authors
of `sGDML <http://quantum-machine.org/gdml>`_.
This class provides access to the original MD17 datasets, their revised
versions, and the CCSD(T) trajectories.

For every trajectory, the dataset contains the Cartesian positions of atoms
(in Angstrom), their atomic numbers, as well as the total energy
(in kcal/mol) and forces (kcal/mol/Angstrom) on each atom.
The latter two are the regression targets for this collection.

.. note::

    Data objects contain no edge indices as these are most commonly
    constructed via the :obj:`torch_geometric.transforms.RadiusGraph`
    transform, with its cut-off being a hyperparameter.

The `original MD17 dataset <https://arxiv.org/abs/1611.04678>`_ contains
ten molecule trajectories.
This version of the dataset was found to suffer from high numerical noise.
The `revised MD17 dataset <https://arxiv.org/abs/2007.09593>`_ contains the
same molecules, but the energies and forces were recalculated at the
PBE/def2-SVP level of theory using very tight SCF convergence and very
dense DFT integration grid.
The third version of the dataset contains fewer molecules, computed at the
CCSD(T) level of theory.
The benzene molecule at the DFT FHI-aims level of theory was
`released separately <https://arxiv.org/abs/1802.09238>`_.

Check the table below for detailed information on the molecule, level of
theory and number of data points contained in each dataset.
Which trajectory is loaded is determined by the :attr:`name` argument.
For the coupled cluster trajectories, the dataset comes with pre-defined
training and testing splits which are loaded separately via the
:attr:`train` argument.

+--------------------+--------------------+-------------------------------+-----------+
| Molecule           | Level of Theory    | Name                          | #Examples |
+====================+====================+===============================+===========+
| Benzene            | DFT                | :obj:`benzene`                | 627,983   |
+--------------------+--------------------+-------------------------------+-----------+
| Uracil             | DFT                | :obj:`uracil`                 | 133,770   |
+--------------------+--------------------+-------------------------------+-----------+
| Naphthalene        | DFT                | :obj:`napthalene`             | 326,250   |
+--------------------+--------------------+-------------------------------+-----------+
| Aspirin            | DFT                | :obj:`aspirin`                | 211,762   |
+--------------------+--------------------+-------------------------------+-----------+
| Salicylic acid     | DFT                | :obj:`salicylic acid`         | 320,231   |
+--------------------+--------------------+-------------------------------+-----------+
| Malonaldehyde      | DFT                | :obj:`malonaldehyde`          | 993,237   |
+--------------------+--------------------+-------------------------------+-----------+
| Ethanol            | DFT                | :obj:`ethanol`                | 555,092   |
+--------------------+--------------------+-------------------------------+-----------+
| Toluene            | DFT                | :obj:`toluene`                | 442,790   |
+--------------------+--------------------+-------------------------------+-----------+
| Paracetamol        | DFT                | :obj:`paracetamol`            | 106,490   |
+--------------------+--------------------+-------------------------------+-----------+
| Azobenzene         | DFT                | :obj:`azobenzene`             | 99,999    |
+--------------------+--------------------+-------------------------------+-----------+
| Benzene (R)        | DFT (PBE/def2-SVP) | :obj:`revised benzene`        | 100,000   |
+--------------------+--------------------+-------------------------------+-----------+
| Uracil (R)         | DFT (PBE/def2-SVP) | :obj:`revised uracil`         | 100,000   |
+--------------------+--------------------+-------------------------------+-----------+
| Naphthalene (R)    | DFT (PBE/def2-SVP) | :obj:`revised napthalene`     | 100,000   |
+--------------------+--------------------+-------------------------------+-----------+
| Aspirin (R)        | DFT (PBE/def2-SVP) | :obj:`revised aspirin`        | 100,000   |
+--------------------+--------------------+-------------------------------+-----------+
| Salicylic acid (R) | DFT (PBE/def2-SVP) | :obj:`revised salicylic acid` | 100,000   |
+--------------------+--------------------+-------------------------------+-----------+
| Malonaldehyde (R)  | DFT (PBE/def2-SVP) | :obj:`revised malonaldehyde`  | 100,000   |
+--------------------+--------------------+-------------------------------+-----------+
| Ethanol (R)        | DFT (PBE/def2-SVP) | :obj:`revised ethanol`        | 100,000   |
+--------------------+--------------------+-------------------------------+-----------+
| Toluene (R)        | DFT (PBE/def2-SVP) | :obj:`revised toluene`        | 100,000   |
+--------------------+--------------------+-------------------------------+-----------+
| Paracetamol (R)    | DFT (PBE/def2-SVP) | :obj:`revised paracetamol`    | 100,000   |
+--------------------+--------------------+-------------------------------+-----------+
| Azobenzene (R)     | DFT (PBE/def2-SVP) | :obj:`revised azobenzene`     | 99,988    |
+--------------------+--------------------+-------------------------------+-----------+
| Benzene            | CCSD(T)            | :obj:`benzene CCSD(T)`        | 1,500     |
+--------------------+--------------------+-------------------------------+-----------+
| Aspirin            | CCSD               | :obj:`aspirin CCSD`           | 1,500     |
+--------------------+--------------------+-------------------------------+-----------+
| Malonaldehyde      | CCSD(T)            | :obj:`malonaldehyde CCSD(T)`  | 1,500     |
+--------------------+--------------------+-------------------------------+-----------+
| Ethanol            | CCSD(T)            | :obj:`ethanol CCSD(T)`        | 2,000     |
+--------------------+--------------------+-------------------------------+-----------+
| Toluene            | CCSD(T)            | :obj:`toluene CCSD(T)`        | 1,501     |
+--------------------+--------------------+-------------------------------+-----------+
| Benzene            | DFT FHI-aims       | :obj:`benzene FHI-aims`       | 49,863    |
+--------------------+--------------------+-------------------------------+-----------+

.. warning::

    It is advised to not train a model on more than 1,000 samples from the
    original or revised MD17 dataset.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): Keyword of the trajectory that should be loaded.
    train (bool, optional): Determines whether the train or test split
        gets loaded for the coupled cluster trajectories.
        (default: :obj:`None`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 20 10 10 10 10 10
    :header-rows: 1

    * - Name
      - #graphs
      - #nodes
      - #edges
      - #features
      - #tasks
    * - Benzene
      - 627,983
      - 12
      - 0
      - 1
      - 2
    * - Uracil
      - 133,770
      - 12
      - 0
      - 1
      - 2
    * - Naphthalene
      - 326,250
      - 10
      - 0
      - 1
      - 2
    * - Aspirin
      - 211,762
      - 21
      - 0
      - 1
      - 2
    * - Salicylic acid
      - 320,231
      - 16
      - 0
      - 1
      - 2
    * - Malonaldehyde
      - 993,237
      - 9
      - 0
      - 1
      - 2
    * - Ethanol
      - 555,092
      - 9
      - 0
      - 1
      - 2
    * - Toluene
      - 442,790
      - 15
      - 0
      - 1
      - 2
    * - Paracetamol
      - 106,490
      - 20
      - 0
      - 1
      - 2
    * - Azobenzene
      - 99,999
      - 24
      - 0
      - 1
      - 2
    * - Benzene (R)
      - 100,000
      - 12
      - 0
      - 1
      - 2
    * - Uracil (R)
      - 100,000
      - 12
      - 0
      - 1
      - 2
    * - Naphthalene (R)
      - 100,000
      - 10
      - 0
      - 1
      - 2
    * - Aspirin (R)
      - 100,000
      - 21
      - 0
      - 1
      - 2
    * - Salicylic acid (R)
      - 100,000
      - 16
      - 0
      - 1
      - 2
    * - Malonaldehyde (R)
      - 100,000
      - 9
      - 0
      - 1
      - 2
    * - Ethanol (R)
      - 100,000
      - 9
      - 0
      - 1
      - 2
    * - Toluene (R)
      - 100,000
      - 15
      - 0
      - 1
      - 2
    * - Paracetamol (R)
      - 100,000
      - 20
      - 0
      - 1
      - 2
    * - Azobenzene (R)
      - 99,988
      - 24
      - 0
      - 1
      - 2
    * - Benzene CCSD-T
      - 1,500
      - 12
      - 0
      - 1
      - 2
    * - Aspirin CCSD-T
      - 1,500
      - 21
      - 0
      - 1
      - 2
    * - Malonaldehyde CCSD-T
      - 1,500
      - 9
      - 0
      - 1
      - 2
    * - Ethanol CCSD-T
      - 2000
      - 9
      - 0
      - 1
      - 2
    * - Toluene CCSD-T
      - 1,501
      - 15
      - 0
      - 1
      - 2
    * - Benzene FHI-aims
      - 49,863
      - 12
      - 0
      - 1
      - 2

#### Methods

- **`mean(self) -> float`**

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `MNISTSuperpixels`

MNIST superpixels dataset from the `"Geometric Deep Learning on
Graphs and Manifolds Using Mixture Model CNNs"
<https://arxiv.org/abs/1611.08402>`_ paper, containing 70,000 graphs with
75 nodes each.
Every graph is labeled by one of 10 classes.

Args:
    root (str): Root directory where the dataset should be saved.
    train (bool, optional): If :obj:`True`, loads the training dataset,
        otherwise the test dataset. (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - #graphs
      - #nodes
      - #edges
      - #features
      - #classes
    * - 70,000
      - 75
      - ~1,393.0
      - 1
      - 10

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `MalNetTiny`

The MalNet Tiny dataset from the
`"A Large-Scale Database for Graph Representation Learning"
<https://openreview.net/pdf?id=1xDTDk3XPW>`_ paper.
:class:`MalNetTiny` contains 5,000 malicious and benign software function
call graphs across 5 different types. Each graph contains at most 5k nodes.

Args:
    root (str): Root directory where the dataset should be saved.
    split (str, optional): If :obj:`"train"`, loads the training dataset.
        If :obj:`"val"`, loads the validation dataset.
        If :obj:`"trainval"`, loads the training and validation dataset.
        If :obj:`"test"`, loads the test dataset.
        If :obj:`None`, loads the entire dataset.
        (default: :obj:`None`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `MixHopSyntheticDataset`

The MixHop synthetic dataset from the `"MixHop: Higher-Order
Graph Convolutional Architectures via Sparsified Neighborhood Mixing"
<https://arxiv.org/abs/1905.00067>`_ paper, containing 10
graphs, each with varying degree of homophily (ranging from 0.0 to 0.9).
All graphs have 5,000 nodes, where each node corresponds to 1 out of 10
classes.
The feature values of the nodes are sampled from a 2D Gaussian
distribution, which are distinct for each class.

Args:
    root (str): Root directory where the dataset should be saved.
    homophily (float): The degree of homophily (one of :obj:`0.0`,
        :obj:`0.1`, ..., :obj:`0.9`).
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

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `ModelNet`

The ModelNet10/40 datasets from the `"3D ShapeNets: A Deep
Representation for Volumetric Shapes"
<https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf>`_ paper,
containing CAD models of 10 and 40 categories, respectively.

.. note::

    Data objects hold mesh faces instead of edge indices.
    To convert the mesh to a graph, use the
    :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
    To convert the mesh to a point cloud, use the
    :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
    sample a fixed number of points on the mesh faces according to their
    face area.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str, optional): The name of the dataset (:obj:`"10"` for
        ModelNet10, :obj:`"40"` for ModelNet40). (default: :obj:`"10"`)
    train (bool, optional): If :obj:`True`, loads the training dataset,
        otherwise the test dataset. (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 20 10 10 10 10 10
    :header-rows: 1

    * - Name
      - #graphs
      - #nodes
      - #edges
      - #features
      - #classes
    * - ModelNet10
      - 4,899
      - ~9,508.2
      - ~37,450.5
      - 3
      - 10
    * - ModelNet40
      - 12,311
      - ~17,744.4
      - ~66,060.9
      - 3
      - 40

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

- **`process_set(self, dataset: str) -> List[torch_geometric.data.data.Data]`**

### `MoleculeNet`

The `MoleculeNet <http://moleculenet.org/datasets-1>`_ benchmark
collection  from the `"MoleculeNet: A Benchmark for Molecular Machine
Learning" <https://arxiv.org/abs/1703.00564>`_ paper, containing datasets
from physical chemistry, biophysics and physiology.
All datasets come with the additional node and edge features introduced by
the :ogb:`null`
`Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`_.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"ESOL"`, :obj:`"FreeSolv"`,
        :obj:`"Lipo"`, :obj:`"PCBA"`, :obj:`"MUV"`, :obj:`"HIV"`,
        :obj:`"BACE"`, :obj:`"BBBP"`, :obj:`"Tox21"`, :obj:`"ToxCast"`,
        :obj:`"SIDER"`, :obj:`"ClinTox"`).
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)
    from_smiles (callable, optional): A custom function that takes a SMILES
        string and outputs a :obj:`~torch_geometric.data.Data` object.
        If not set, defaults to :meth:`~torch_geometric.utils.from_smiles`.
        (default: :obj:`None`)

**STATS:**

.. list-table::
    :widths: 20 10 10 10 10 10
    :header-rows: 1

    * - Name
      - #graphs
      - #nodes
      - #edges
      - #features
      - #classes
    * - ESOL
      - 1,128
      - ~13.3
      - ~27.4
      - 9
      - 1
    * - FreeSolv
      - 642
      - ~8.7
      - ~16.8
      - 9
      - 1
    * - Lipophilicity
      - 4,200
      - ~27.0
      - ~59.0
      - 9
      - 1
    * - PCBA
      - 437,929
      - ~26.0
      - ~56.2
      - 9
      - 128
    * - MUV
      - 93,087
      - ~24.2
      - ~52.6
      - 9
      - 17
    * - HIV
      - 41,127
      - ~25.5
      - ~54.9
      - 9
      - 1
    * - BACE
      - 1513
      - ~34.1
      - ~73.7
      - 9
      - 1
    * - BBBP
      - 2,050
      - ~23.9
      - ~51.6
      - 9
      - 1
    * - Tox21
      - 7,831
      - ~18.6
      - ~38.6
      - 9
      - 12
    * - ToxCast
      - 8,597
      - ~18.7
      - ~38.4
      - 9
      - 617
    * - SIDER
      - 1,427
      - ~33.6
      - ~70.7
      - 9
      - 27
    * - ClinTox
      - 1,484
      - ~26.1
      - ~55.5
      - 9
      - 2

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `MovieLens`

A heterogeneous rating dataset, assembled by GroupLens Research from
the `MovieLens web site <https://movielens.org>`_, consisting of nodes of
type :obj:`"movie"` and :obj:`"user"`.
User ratings for movies are available as ground truth labels for the edges
between the users and the movies :obj:`("user", "rates", "movie")`.

Args:
    root (str): Root directory where the dataset should be saved.
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    model_name (str): Name of model used to transform movie titles to node
        features. The model comes from the`Huggingface SentenceTransformer
        <https://huggingface.co/sentence-transformers>`_.
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `MovieLens100K`

The MovieLens 100K heterogeneous rating dataset, assembled by GroupLens
Research from the `MovieLens web site <https://movielens.org>`__,
consisting of movies (1,682 nodes) and users (943 nodes) with 100K
ratings between them.
User ratings for movies are available as ground truth labels.
Features of users and movies are encoded according to the `"Inductive
Matrix Completion Based on Graph Neural Networks"
<https://arxiv.org/abs/1904.12058>`__ paper.

Args:
    root (str): Root directory where the dataset should be saved.
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 20 10 10 10
    :header-rows: 1

    * - Node/Edge Type
      - #nodes/#edges
      - #features
      - #tasks
    * - Movie
      - 1,682
      - 18
      -
    * - User
      - 943
      - 24
      -
    * - User-Movie
      - 80,000
      - 1
      - 1

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `MovieLens1M`

The MovieLens 1M heterogeneous rating dataset, assembled by GroupLens
Research from the `MovieLens web site <https://movielens.org>`__,
consisting of movies (3,883 nodes) and users (6,040 nodes) with
approximately 1 million ratings between them.
User ratings for movies are available as ground truth labels.
Features of users and movies are encoded according to the `"Inductive
Matrix Completion Based on Graph Neural Networks"
<https://arxiv.org/abs/1904.12058>`__ paper.

Args:
    root (str): Root directory where the dataset should be saved.
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 20 10 10 10
    :header-rows: 1

    * - Node/Edge Type
      - #nodes/#edges
      - #features
      - #tasks
    * - Movie
      - 3,883
      - 18
      -
    * - User
      - 6,040
      - 30
      -
    * - User-Movie
      - 1,000,209
      - 1
      - 1

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `MyketDataset`

The Myket Android Application Install dataset from the
`"Effect of Choosing Loss Function when Using T-Batching for Representation
Learning on Dynamic Networks" <https://arxiv.org/abs/2308.06862>`_ paper.
The dataset contains a temporal graph of application install interactions
in an Android application market.

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
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - Name
      - #nodes
      - #edges
      - #features
      - #classes
    * - Myket
      - 17,988
      - 694,121
      - 33
      - 1

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `NELL`

The NELL dataset, a knowledge graph from the
`"Toward an Architecture for Never-Ending Language Learning"
<https://www.cs.cmu.edu/~acarlson/papers/carlson-aaai10.pdf>`_ paper.
The dataset is processed as in the
`"Revisiting Semi-Supervised Learning with Graph Embeddings"
<https://arxiv.org/abs/1603.08861>`_ paper.

.. note::

    Entity nodes are described by sparse feature vectors of type
    :class:`torch.sparse_csr_tensor`.

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
    * - 65,755
      - 251,550
      - 61,278
      - 186

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `NeuroGraphDataset`

The NeuroGraph benchmark datasets from the
`"NeuroGraph: Benchmarks for Graph Machine Learning in Brain Connectomics"
<https://arxiv.org/abs/2306.06202>`_ paper.
:class:`NeuroGraphDataset` holds a collection of five neuroimaging graph
learning datasets that span multiple categories of demographics, mental
states, and cognitive traits.
See the `documentation
<https://neurograph.readthedocs.io/en/latest/NeuroGraph.html>`_ and the
`Github <https://github.com/Anwar-Said/NeuroGraph>`_ for more details.

+--------------------+---------+----------------------+
| Dataset            | #Graphs | Task                 |
+====================+=========+======================+
| :obj:`HCPTask`     | 7,443   | Graph Classification |
+--------------------+---------+----------------------+
| :obj:`HCPGender`   | 1,078   | Graph Classification |
+--------------------+---------+----------------------+
| :obj:`HCPAge`      | 1,065   | Graph Classification |
+--------------------+---------+----------------------+
| :obj:`HCPFI`       | 1,071   | Graph Regression     |
+--------------------+---------+----------------------+
| :obj:`HCPWM`       | 1,078   | Graph Regression     |
+--------------------+---------+----------------------+

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (one of :obj:`"HCPGender"`,
        :obj:`"HCPTask"`, :obj:`"HCPAge"`, :obj:`"HCPFI"`,
        :obj:`"HCPWM"`).
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `OGB_MAG`

The :obj:`ogbn-mag` dataset from the `"Open Graph Benchmark: Datasets
for Machine Learning on Graphs" <https://arxiv.org/abs/2005.00687>`_ paper.
:obj:`ogbn-mag` is a heterogeneous graph composed of a subset of the
Microsoft Academic Graph (MAG).
It contains four types of entities — papers (736,389 nodes), authors
(1,134,649 nodes), institutions (8,740 nodes), and fields of study
(59,965 nodes) — as well as four types of directed relations connecting two
types of entities.
Each paper is associated with a 128-dimensional :obj:`word2vec` feature
vector, while all other node types are not associated with any input
features.
The task is to predict the venue (conference or journal) of each paper.
In total, there are 349 different venues.

Args:
    root (str): Root directory where the dataset should be saved.
    preprocess (str, optional): Pre-processes the original dataset by
        adding structural features (:obj:`"metapath2vec"`, :obj:`"TransE"`)
        to featureless nodes. (default: :obj:`None`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `OMDB`

The `Organic Materials Database (OMDB)
<https://omdb.mathub.io/dataset>`__ of bulk organic crystals.

Args:
    root (str): Root directory where the dataset should be saved.
    train (bool, optional): If :obj:`True`, loads the training dataset,
        otherwise the test dataset. (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `OPFDataset`

The heterogeneous OPF data from the `"Large-scale Datasets for AC
Optimal Power Flow with Topological Perturbations"
<https://arxiv.org/abs/2406.07234>`_ paper.

:class:`OPFDataset` is a large-scale dataset of solved optimal power flow
problems, derived from the
`pglib-opf <https://github.com/power-grid-lib/pglib-opf>`_ dataset.

The physical topology of the grid is represented by the :obj:`"bus"` node
type, and the connecting AC lines and transformers. Additionally,
:obj:`"generator"`, :obj:`"load"`, and :obj:`"shunt"` nodes are connected
to :obj:`"bus"` nodes using a dedicated edge type each, *e.g.*,
:obj:`"generator_link"`.

Edge direction corresponds to the properties of the line, *e.g.*,
:obj:`b_fr` is the line charging susceptance at the :obj:`from`
(source/sender) bus.

Args:
    root (str): Root directory where the dataset should be saved.
    split (str, optional): If :obj:`"train"`, loads the training dataset.
        If :obj:`"val"`, loads the validation dataset.
        If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
    case_name (str, optional): The name of the original pglib-opf case.
        (default: :obj:`"pglib_opf_case14_ieee"`)
    num_groups (int, optional): The dataset is divided into 20 groups with
        each group containing 15,000 samples.
        For large networks, this amount of data can be overwhelming.
        The :obj:`num_groups` parameters controls the amount of data being
        downloaded. Allowed values are :obj:`[1, 20]`.
        (default: :obj:`20`)
    topological_perturbations (bool, optional): Whether to use the dataset
        with added topological perturbations. (default: :obj:`False`)
    transform (callable, optional): A function/transform that takes in
        a :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes
        in a :obj:`torch_geometric.data.HeteroData` object and returns
        a transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in a
        :obj:`torch_geometric.data.HeteroData` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `OSE_GVCS`

A dataset describing the `Product ecology
<https://wiki.opensourceecology.org/wiki/Product_Ecologies>`_ of the Open
Source Ecology's iconoclastic `Global Village Construction Set
<https://wiki.opensourceecology.org/wiki/
Global_Village_Construction_Set>`_.
GVCS is a modular, DIY, low-cost set of blueprints that enables the
fabrication of the 50 different industrial machines that it takes to
build a small, sustainable civilization with modern comforts.

The dataset contains a heterogenous graphs with 50 :obj:`machine` nodes,
composing the GVCS, and 290 directed edges, each representing one out of
three relationships between machines.

Args:
    root (str): Root directory where the dataset should be saved.
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `PCPNetDataset`

The PCPNet dataset from the `"PCPNet: Learning Local Shape Properties
from Raw Point Clouds" <https://arxiv.org/abs/1710.04954>`_ paper,
consisting of 30 shapes, each given as a point cloud, densely sampled with
100k points.
For each shape, surface normals and local curvatures are given as node
features.

Args:
    root (str): Root directory where the dataset should be saved.
    category (str): The training set category (one of :obj:`"NoNoise"`,
        :obj:`"Noisy"`, :obj:`"VarDensity"`, :obj:`"NoisyAndVarDensity"`
        for :obj:`split="train"` or :obj:`split="val"`,
        or one of :obj:`"All"`, :obj:`"LowNoise"`, :obj:`"MedNoise"`,
        :obj:`"HighNoise", :obj:`"VarDensityStriped",
        :obj:`"VarDensityGradient"` for :obj:`split="test"`).
    split (str, optional): If :obj:`"train"`, loads the training dataset.
        If :obj:`"val"`, loads the validation dataset.
        If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

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

### `PPI`

The protein-protein interaction networks from the `"Predicting
Multicellular Function through Multi-layer Tissue Networks"
<https://arxiv.org/abs/1707.04638>`_ paper, containing positional gene
sets, motif gene sets and immunological signatures as features (50 in
total) and gene ontology sets as labels (121 in total).

Args:
    root (str): Root directory where the dataset should be saved.
    split (str, optional): If :obj:`"train"`, loads the training dataset.
        If :obj:`"val"`, loads the validation dataset.
        If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - #graphs
      - #nodes
      - #edges
      - #features
      - #tasks
    * - 20
      - ~2,245.3
      - ~61,318.4
      - 50
      - 121

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `PascalPF`

The Pascal-PF dataset from the `"Proposal Flow"
<https://arxiv.org/abs/1511.05065>`_ paper, containing 4 to 16 keypoints
per example over 20 categories.

Args:
    root (str): Root directory where the dataset should be saved.
    category (str): The category of the images (one of
        :obj:`"Aeroplane"`, :obj:`"Bicycle"`, :obj:`"Bird"`,
        :obj:`"Boat"`, :obj:`"Bottle"`, :obj:`"Bus"`, :obj:`"Car"`,
        :obj:`"Cat"`, :obj:`"Chair"`, :obj:`"Diningtable"`, :obj:`"Dog"`,
        :obj:`"Horse"`, :obj:`"Motorbike"`, :obj:`"Person"`,
        :obj:`"Pottedplant"`, :obj:`"Sheep"`, :obj:`"Sofa"`,
        :obj:`"Train"`, :obj:`"TVMonitor"`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `PascalVOCKeypoints`

The Pascal VOC 2011 dataset with Berkely annotations of keypoints from
the `"Poselets: Body Part Detectors Trained Using 3D Human Pose
Annotations" <https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/
human/ poselets_iccv09.pdf>`_ paper, containing 0 to 23 keypoints per
example over 20 categories.
The dataset is pre-filtered to exclude difficult, occluded and truncated
objects.
The keypoints contain interpolated features from a pre-trained VGG16 model
on ImageNet (:obj:`relu4_2` and :obj:`relu5_1`).

Args:
    root (str): Root directory where the dataset should be saved.
    category (str): The category of the images (one of
        :obj:`"Aeroplane"`, :obj:`"Bicycle"`, :obj:`"Bird"`,
        :obj:`"Boat"`, :obj:`"Bottle"`, :obj:`"Bus"`, :obj:`"Car"`,
        :obj:`"Cat"`, :obj:`"Chair"`, :obj:`"Diningtable"`, :obj:`"Dog"`,
        :obj:`"Horse"`, :obj:`"Motorbike"`, :obj:`"Person"`,
        :obj:`"Pottedplant"`, :obj:`"Sheep"`, :obj:`"Sofa"`,
        :obj:`"Train"`, :obj:`"TVMonitor"`)
    train (bool, optional): If :obj:`True`, loads the training dataset,
        otherwise the test dataset. (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)
    device (str or torch.device, optional): The device to use for
        processing the raw data. If set to :obj:`None`, will utilize
        GPU-processing if available. (default: :obj:`None`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

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

### `PolBlogs`

The Political Blogs dataset from the `"The Political Blogosphere and
the 2004 US Election: Divided they Blog"
<https://dl.acm.org/doi/10.1145/1134271.1134277>`_ paper.

:class:`Polblogs` is a graph with 1,490 vertices (representing political
blogs) and 19,025 edges (links between blogs).
The links are automatically extracted from a crawl of the front page of the
blog.
Each vertex receives a label indicating the political leaning of the blog:
liberal or conservative.

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
    * - 1,490
      - 19,025
      - 0
      - 2

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `QM7b`

The QM7b dataset from the `"MoleculeNet: A Benchmark for Molecular
Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
7,211 molecules with 14 regression targets.

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
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - #graphs
      - #nodes
      - #edges
      - #features
      - #tasks
    * - 7,211
      - ~15.4
      - ~245.0
      - 0
      - 14

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `QM9`

The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
about 130,000 molecules with 19 regression targets.
Each molecule includes complete spatial information for the single low
energy conformation of the atoms in the molecule.
In addition, we provide the atom features from the `"Neural Message
Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| Target | Property                         | Description                                                                       | Unit                                        |
+========+==================================+===================================================================================+=============================================+
| 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 11     | :math:`c_{\textrm{v}}`           | Heat capavity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

.. note::

    We also provide a pre-processed version of the dataset in case
    :class:`rdkit` is not installed. The pre-processed version matches with
    the manually processed version as outlined in :meth:`process`.

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
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - #graphs
      - #nodes
      - #edges
      - #features
      - #tasks
    * - 130,831
      - ~18.0
      - ~37.3
      - 11
      - 19

#### Methods

- **`mean(self, target: int) -> float`**

- **`std(self, target: int) -> float`**

- **`atomref(self, target: int) -> Optional[torch.Tensor]`**

### `RCDD`

The risk commodity detection dataset (RCDD) from the
`"Datasets and Interfaces for Benchmarking Heterogeneous Graph
Neural Networks" <https://dl.acm.org/doi/10.1145/3583780.3615117>`_ paper.
RCDD is an industrial-scale heterogeneous graph dataset based on a
real risk detection scenario from Alibaba's e-commerce platform.
It consists of 13,806,619 nodes and 157,814,864 edges across 7 node types
and 7 edge types, respectively.

Args:
    root (str): Root directory where the dataset should be saved.
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `RandomPartitionGraphDataset`

The random partition graph dataset from the `"How to Find Your
Friendly Neighborhood: Graph Attention Design with Self-Supervision"
<https://openreview.net/forum?id=Wi5KUNlqWty>`_ paper.
This is a synthetic graph of communities controlled by the node homophily
and the average degree, and each community is considered as a class.
The node features are sampled from normal distributions where the centers
of clusters are vertices of a hypercube, as computed by the
:meth:`sklearn.datasets.make_classification` method.

Args:
    root (str): Root directory where the dataset should be saved.
    num_classes (int): The number of classes.
    num_nodes_per_class (int): The number of nodes per class.
    node_homophily_ratio (float): The degree of node homophily.
    average_degree (float): The average degree of the graph.
    num_graphs (int, optional): The number of graphs. (default: :obj:`1`)
    num_channels (int, optional): The number of node features. If given
        as :obj:`None`, node features are not generated.
        (default: :obj:`None`)
    is_undirected (bool, optional): Whether the graph to generate is
        undirected. (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes
        in an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    **kwargs (optional): The keyword arguments that are passed down
        to :meth:`sklearn.datasets.make_classification` method in
        drawing node features.

#### Methods

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `Reddit`

The Reddit dataset from the `"Inductive Representation Learning on
Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, containing
Reddit posts belonging to different communities.

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
    * - 232,965
      - 114,615,892
      - 602
      - 41

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `Reddit2`

The Reddit dataset from the `"GraphSAINT: Graph Sampling Based
Inductive Learning Method" <https://arxiv.org/abs/1907.04931>`_ paper,
containing Reddit posts belonging to different communities.

.. note::

    This is a sparser version of the original
    :obj:`~torch_geometric.datasets.Reddit` dataset (~23M edges instead of
    ~114M edges), and is used in papers such as
    `SGC <https://arxiv.org/abs/1902.07153>`_ and
    `GraphSAINT <https://arxiv.org/abs/1907.04931>`_.

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
    * - 232,965
      - 23,213,838
      - 602
      - 41

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `RelLinkPredDataset`

The relational link prediction datasets from the
`"Modeling Relational Data with Graph Convolutional Networks"
<https://arxiv.org/abs/1703.06103>`_ paper.
Training and test splits are given by sets of triplets.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"FB15k-237"`).
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
    * - 14,541
      - 544,230
      - 0
      - 0

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `S3DIS`

The (pre-processed) Stanford Large-Scale 3D Indoor Spaces dataset from
the `"3D Semantic Parsing of Large-Scale Indoor Spaces"
<https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf>`_
paper, containing point clouds of six large-scale indoor parts in three
buildings with 12 semantic elements (and one clutter class).

Args:
    root (str): Root directory where the dataset should be saved.
    test_area (int, optional): Which area to use for testing (1-6).
        (default: :obj:`6`)
    train (bool, optional): If :obj:`True`, loads the training dataset,
        otherwise the test dataset. (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `SHREC2016`

The SHREC 2016 partial matching dataset from the `"SHREC'16: Partial
Matching of Deformable Shapes"
<http://www.dais.unive.it/~shrec2016/shrec16-partial.pdf>`_ paper.
The reference shape can be referenced via :obj:`dataset.ref`.

.. note::

    Data objects hold mesh faces instead of edge indices.
    To convert the mesh to a graph, use the
    :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
    To convert the mesh to a point cloud, use the
    :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
    sample a fixed number of points on the mesh faces according to their
    face area.

Args:
    root (str): Root directory where the dataset should be saved.
    partiality (str): The partiality of the dataset (one of :obj:`"Holes"`,
        :obj:`"Cuts"`).
    category (str): The category of the dataset (one of
        :obj:`"Cat"`, :obj:`"Centaur"`, :obj:`"David"`, :obj:`"Dog"`,
        :obj:`"Horse"`, :obj:`"Michael"`, :obj:`"Victoria"`,
        :obj:`"Wolf"`).
    train (bool, optional): If :obj:`True`, loads the training dataset,
        otherwise the test dataset. (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `SNAPDataset`

A variety of graph datasets collected from `SNAP at Stanford University
<https://snap.stanford.edu/data>`_.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset.
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `ShapeNet`

The ShapeNet part level segmentation dataset from the `"A Scalable
Active Framework for Region Annotation in 3D Shape Collections"
<http://web.stanford.edu/~ericyi/papers/part_annotation_16_small.pdf>`_
paper, containing about 17,000 3D shape point clouds from 16 shape
categories.
Each category is annotated with 2 to 6 parts.

Args:
    root (str): Root directory where the dataset should be saved.
    categories (str or [str], optional): The category of the CAD models
        (one or a combination of :obj:`"Airplane"`, :obj:`"Bag"`,
        :obj:`"Cap"`, :obj:`"Car"`, :obj:`"Chair"`, :obj:`"Earphone"`,
        :obj:`"Guitar"`, :obj:`"Knife"`, :obj:`"Lamp"`, :obj:`"Laptop"`,
        :obj:`"Motorbike"`, :obj:`"Mug"`, :obj:`"Pistol"`, :obj:`"Rocket"`,
        :obj:`"Skateboard"`, :obj:`"Table"`).
        Can be explicitly set to :obj:`None` to load all categories.
        (default: :obj:`None`)
    include_normals (bool, optional): If set to :obj:`False`, will not
        include normal vectors as input features to :obj:`data.x`.
        As a result, :obj:`data.x` will be :obj:`None`.
        (default: :obj:`True`)
    split (str, optional): If :obj:`"train"`, loads the training dataset.
        If :obj:`"val"`, loads the validation dataset.
        If :obj:`"trainval"`, loads the training and validation dataset.
        If :obj:`"test"`, loads the test dataset.
        (default: :obj:`"trainval"`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - #graphs
      - #nodes
      - #edges
      - #features
      - #classes
    * - 16,881
      - ~2,616.2
      - 0
      - 3
      - 50

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process_filenames(self, filenames: List[str]) -> List[torch_geometric.data.data.Data]`**

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `StochasticBlockModelDataset`

A synthetic graph dataset generated by the stochastic block model.
The node features of each block are sampled from normal distributions where
the centers of clusters are vertices of a hypercube, as computed by the
:meth:`sklearn.datasets.make_classification` method.

Args:
    root (str): Root directory where the dataset should be saved.
    block_sizes ([int] or LongTensor): The sizes of blocks.
    edge_probs ([[float]] or FloatTensor): The density of edges going from
        each block to each other block. Must be symmetric if the graph is
        undirected.
    num_graphs (int, optional): The number of graphs. (default: :obj:`1`)
    num_channels (int, optional): The number of node features. If given
        as :obj:`None`, node features are not generated.
        (default: :obj:`None`)
    is_undirected (bool, optional): Whether the graph to generate is
        undirected. (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes
        in an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed
        before being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)
    **kwargs (optional): The keyword arguments that are passed down to the
        :meth:`sklearn.datasets.make_classification` method for drawing
        node features.

#### Methods

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `SuiteSparseMatrixCollection`

A suite of sparse matrix benchmarks known as the `Suite Sparse Matrix
Collection <https://sparse.tamu.edu>`_ collected from a wide range of
applications.

Args:
    root (str): Root directory where the dataset should be saved.
    group (str): The group of the sparse matrix.
    name (str): The name of the sparse matrix.
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

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `TOSCA`

The TOSCA dataset from the `"Numerical Geometry of Non-Ridig Shapes"
<https://www.amazon.com/Numerical-Geometry-Non-Rigid-Monographs-Computer/
dp/0387733000>`_ book, containing 80 meshes.
Meshes within the same category have the same triangulation and an equal
number of vertices numbered in a compatible way.

.. note::

    Data objects hold mesh faces instead of edge indices.
    To convert the mesh to a graph, use the
    :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
    To convert the mesh to a point cloud, use the
    :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
    sample a fixed number of points on the mesh faces according to their
    face area.

Args:
    root (str): Root directory where the dataset should be saved.
    categories (list, optional): List of categories to include in the
        dataset. Can include the categories :obj:`"Cat"`, :obj:`"Centaur"`,
        :obj:`"David"`, :obj:`"Dog"`, :obj:`"Gorilla"`, :obj:`"Horse"`,
        :obj:`"Michael"`, :obj:`"Victoria"`, :obj:`"Wolf"`. If set to
        :obj:`None`, the dataset will contain all categories. (default:
        :obj:`None`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `TUDataset`

A variety of graph kernel benchmark datasets, *.e.g.*,
:obj:`"IMDB-BINARY"`, :obj:`"REDDIT-BINARY"` or :obj:`"PROTEINS"`,
collected from the `TU Dortmund University
<https://chrsmrrs.github.io/datasets>`_.
In addition, this dataset wrapper provides `cleaned dataset versions
<https://github.com/nd7141/graph_datasets>`_ as motivated by the
`"Understanding Isomorphism Bias in Graph Data Sets"
<https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
graphs.

.. note::
    Some datasets may not come with any node labels.
    You can then either make use of the argument :obj:`use_node_attr`
    to load additional continuous node attributes (if present) or provide
    synthetic node features using transforms such as
    :class:`torch_geometric.transforms.Constant` or
    :class:`torch_geometric.transforms.OneHotDegree`.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The `name
        <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
        dataset.
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)
    use_node_attr (bool, optional): If :obj:`True`, the dataset will
        contain additional continuous node attributes (if present).
        (default: :obj:`False`)
    use_edge_attr (bool, optional): If :obj:`True`, the dataset will
        contain additional continuous edge attributes (if present).
        (default: :obj:`False`)
    cleaned (bool, optional): If :obj:`True`, the dataset will
        contain only non-isomorphic graphs. (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 20 10 10 10 10 10
    :header-rows: 1

    * - Name
      - #graphs
      - #nodes
      - #edges
      - #features
      - #classes
    * - MUTAG
      - 188
      - ~17.9
      - ~39.6
      - 7
      - 2
    * - ENZYMES
      - 600
      - ~32.6
      - ~124.3
      - 3
      - 6
    * - PROTEINS
      - 1,113
      - ~39.1
      - ~145.6
      - 3
      - 2
    * - COLLAB
      - 5,000
      - ~74.5
      - ~4914.4
      - 0
      - 3
    * - IMDB-BINARY
      - 1,000
      - ~19.8
      - ~193.1
      - 0
      - 2
    * - REDDIT-BINARY
      - 2,000
      - ~429.6
      - ~995.5
      - 0
      - 2
    * - ...
      -
      -
      -
      -
      -

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `Taobao`

Taobao is a dataset of user behaviors from Taobao offered by Alibaba,
provided by the `Tianchi Alicloud platform
<https://tianchi.aliyun.com/dataset/649>`_.

Taobao is a heterogeneous graph for recommendation.
Nodes represent users with user IDs, items with item IDs, and categories
with category ID.
Edges between users and items represent different types of user behaviors
towards items (alongside with timestamps).
Edges between items and categories assign each item to its set of
categories.

Args:
    root (str): Root directory where the dataset should be saved.
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `Twitch`

The Twitch Gamer networks introduced in the
`"Multi-scale Attributed Node Embedding"
<https://arxiv.org/abs/1909.13021>`_ paper.
Nodes represent gamers on Twitch and edges are followerships between them.
Node features represent embeddings of games played by the Twitch users.
The task is to predict whether a user streams mature content.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"DE"`, :obj:`"EN"`,
        :obj:`"ES"`, :obj:`"FR"`, :obj:`"PT"`, :obj:`"RU"`).
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
    * - DE
      - 9,498
      - 315,774
      - 128
      - 2
    * - EN
      - 7,126
      - 77,774
      - 128
      - 2
    * - ES
      - 4,648
      - 123,412
      - 128
      - 2
    * - FR
      - 6,551
      - 231,883
      - 128
      - 2
    * - PT
      - 1,912
      - 64,510
      - 128
      - 2
    * - RU
      - 4,385
      - 78,993
      - 128
      - 2

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `UPFD`

The tree-structured fake news propagation graph classification dataset
from the `"User Preference-aware Fake News Detection"
<https://arxiv.org/abs/2104.12259>`_ paper.
It includes two sets of tree-structured fake & real news propagation graphs
extracted from Twitter.
For a single graph, the root node represents the source news, and leaf
nodes represent Twitter users who retweeted the same root news.
A user node has an edge to the news node if and only if the user retweeted
the root news directly.
Two user nodes have an edge if and only if one user retweeted the root news
from the other user.
Four different node features are encoded using different encoders.
Please refer to `GNN-FakeNews
<https://github.com/safe-graph/GNN-FakeNews>`_ repo for more details.

.. note::

    For an example of using UPFD, see `examples/upfd.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    upfd.py>`_.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the graph set (:obj:`"politifact"`,
        :obj:`"gossipcop"`).
    feature (str): The node feature type (:obj:`"profile"`, :obj:`"spacy"`,
        :obj:`"bert"`, :obj:`"content"`).
        If set to :obj:`"profile"`, the 10-dimensional node feature
        is composed of ten Twitter user profile attributes.
        If set to :obj:`"spacy"`, the 300-dimensional node feature is
        composed of Twitter user historical tweets encoded by
        the `spaCy word2vec encoder
        <https://spacy.io/models/en#en_core_web_lg>`_.
        If set to :obj:`"bert"`, the 768-dimensional node feature is
        composed of Twitter user historical tweets encoded by the
        `bert-as-service <https://github.com/hanxiao/bert-as-service>`_.
        If set to :obj:`"content"`, the 310-dimensional node feature is
        composed of a 300-dimensional "spacy" vector plus a
        10-dimensional "profile" vector.
    split (str, optional): If :obj:`"train"`, loads the training dataset.
        If :obj:`"val"`, loads the validation dataset.
        If :obj:`"test"`, loads the test dataset.
        (default: :obj:`"train"`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `WILLOWObjectClass`

The WILLOW-ObjectClass dataset from the `"Learning Graphs to Match"
<https://www.di.ens.fr/willow/pdfscurrent/cho2013.pdf>`_ paper,
containing 10 equal keypoints of at least 40 images in each category.
The keypoints contain interpolated features from a pre-trained VGG16 model
on ImageNet (:obj:`relu4_2` and :obj:`relu5_1`).

Args:
    root (str): Root directory where the dataset should be saved.
    category (str): The category of the images (one of :obj:`"Car"`,
        :obj:`"Duck"`, :obj:`"Face"`, :obj:`"Motorbike"`,
        :obj:`"Winebottle"`).
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)
    device (str or torch.device, optional): The device to use for
        processing the raw data. If set to :obj:`None`, will utilize
        GPU-processing if available. (default: :obj:`None`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `WebKB`

The WebKB datasets used in the
`"Geom-GCN: Geometric Graph Convolutional Networks"
<https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
Nodes represent web pages and edges represent hyperlinks between them.
Node features are the bag-of-words representation of web pages.
The task is to classify the nodes into one of the five categories, student,
project, course, staff, and faculty.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"Cornell"`, :obj:`"Texas"`,
        :obj:`"Wisconsin"`).
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
    * - Cornell
      - 183
      - 298
      - 1,703
      - 5
    * - Texas
      - 183
      - 325
      - 1,703
      - 5
    * - Wisconsin
      - 251
      - 515
      - 1,703
      - 5

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `WebQSPDataset`

The WebQuestionsSP dataset of the `"The Value of Semantic Parse
Labeling for Knowledge Base Question Answering"
<https://aclanthology.org/P16-2033/>`_ paper.

Args:
    root (str): Root directory where the dataset should be saved.
    split (str, optional): If :obj:`"train"`, loads the training dataset.
        If :obj:`"val"`, loads the validation dataset.
        If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `WikiCS`

The semi-supervised Wikipedia-based dataset from the
`"Wiki-CS: A Wikipedia-Based Benchmark for Graph Neural Networks"
<https://arxiv.org/abs/2007.02901>`_ paper, containing 11,701 nodes,
216,123 edges, 10 classes and 20 different training splits.

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
    is_undirected (bool, optional): Whether the graph is undirected.
        (default: :obj:`True`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `Wikidata5M`

The Wikidata-5M dataset from the `"KEPLER: A Unified Model for
Knowledge Embedding and Pre-trained Language Representation"
<https://arxiv.org/abs/1911.06136>`_ paper,
containing 4,594,485 entities, 822 relations,
20,614,279 train triples, 5,163 validation triples, and 5,133 test triples.

`Wikidata-5M <https://deepgraphlearning.github.io/project/wikidata5m>`_
is a large-scale knowledge graph dataset with aligned corpus
extracted form Wikidata.

Args:
    root (str): Root directory where the dataset should be saved.
    setting (str, optional):
        If :obj:`"transductive"`, loads the transductive dataset.
        If :obj:`"inductive"`, loads the inductive dataset.
        (default: :obj:`"transductive"`)
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

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `WikipediaNetwork`

The Wikipedia networks introduced in the
`"Multi-scale Attributed Node Embedding"
<https://arxiv.org/abs/1909.13021>`_ paper.
Nodes represent web pages and edges represent hyperlinks between them.
Node features represent several informative nouns in the Wikipedia pages.
The task is to predict the average daily traffic of the web page.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"chameleon"`,
        :obj:`"crocodile"`, :obj:`"squirrel"`).
    geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
        pre-processed data as introduced in the `"Geom-GCN: Geometric
        Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
        in which the average monthly traffic of the web page is converted
        into five categories to predict.
        If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
        available.
        If set to :obj:`True`, train/validation/test splits will be
        available as masks for multiple splits with shape
        :obj:`[num_nodes, num_splits]`. (default: :obj:`True`)
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

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `WordNet18`

The WordNet18 dataset from the `"Translating Embeddings for Modeling
Multi-Relational Data"
<https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling
-multi-relational-data>`_ paper,
containing 40,943 entities, 18 relations and 151,442 fact triplets,
*e.g.*, furniture includes bed.

.. note::

    The original :obj:`WordNet18` dataset suffers from test leakage, *i.e.*
    more than 80% of test triplets can be found in the training set with
    another relation type.
    Therefore, it should not be used for research evaluation anymore.
    We recommend to use its cleaned version
    :class:`~torch_geometric.datasets.WordNet18RR` instead.

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

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `WordNet18RR`

The WordNet18RR dataset from the `"Convolutional 2D Knowledge Graph
Embeddings" <https://arxiv.org/abs/1707.01476>`_ paper, containing 40,943
entities, 11 relations and 93,003 fact triplets.

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

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `Yelp`

The Yelp dataset from the `"GraphSAINT: Graph Sampling Based
Inductive Learning Method" <https://arxiv.org/abs/1907.04931>`_ paper,
containing customer reviewers and their friendship.

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
      - #tasks
    * - 716,847
      - 13,954,819
      - 300
      - 100

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `ZINC`

The ZINC dataset from the `ZINC database
<https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559>`_ and the
`"Automatic Chemical Design Using a Data-Driven Continuous Representation
of Molecules" <https://arxiv.org/abs/1610.02415>`_ paper, containing about
250,000 molecular graphs with up to 38 heavy atoms.
The task is to regress the penalized :obj:`logP` (also called constrained
solubility in some works), given by :obj:`y = logP - SAS - cycles`, where
:obj:`logP` is the water-octanol partition coefficient, :obj:`SAS` is the
synthetic accessibility score, and :obj:`cycles` denotes the number of
cycles with more than six atoms.
Penalized :obj:`logP` is a score commonly used for training molecular
generation models, see, *e.g.*, the
`"Junction Tree Variational Autoencoder for Molecular Graph Generation"
<https://proceedings.mlr.press/v80/jin18a.html>`_ and
`"Grammar Variational Autoencoder"
<https://proceedings.mlr.press/v70/kusner17a.html>`_ papers.

Args:
    root (str): Root directory where the dataset should be saved.
    subset (bool, optional): If set to :obj:`True`, will only load a
        subset of the dataset (12,000 molecular graphs), following the
        `"Benchmarking Graph Neural Networks"
        <https://arxiv.org/abs/2003.00982>`_ paper. (default: :obj:`False`)
    split (str, optional): If :obj:`"train"`, loads the training dataset.
        If :obj:`"val"`, loads the validation dataset.
        If :obj:`"test"`, loads the test dataset.
        (default: :obj:`"train"`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 20 10 10 10 10 10
    :header-rows: 1

    * - Name
      - #graphs
      - #nodes
      - #edges
      - #features
      - #classes
    * - ZINC Full
      - 249,456
      - ~23.2
      - ~49.8
      - 1
      - 1
    * - ZINC Subset
      - 12,000
      - ~23.2
      - ~49.8
      - 1
      - 1

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.
