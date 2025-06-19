# explainer_dataset

Part of `torch_geometric.datasets`
Module: `torch_geometric.datasets.explainer_dataset`

## Classes (7)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

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

### `Explanation`

Holds all the obtained explanations of a homogeneous graph.

The explanation object is a :obj:`~torch_geometric.data.Data` object and
can hold node attributions and edge attributions.
It can also hold the original graph if needed.

Args:
    node_mask (Tensor, optional): Node-level mask with shape
        :obj:`[num_nodes, 1]`, :obj:`[1, num_features]` or
        :obj:`[num_nodes, num_features]`. (default: :obj:`None`)
    edge_mask (Tensor, optional): Edge-level mask with shape
        :obj:`[num_edges]`. (default: :obj:`None`)
    **kwargs (optional): Additional attributes.

#### Methods

- **`validate(self, raise_on_error: bool = True) -> bool`**
  Validates the correctness of the :class:`Explanation` object.

- **`get_explanation_subgraph(self) -> 'Explanation'`**
  Returns the induced subgraph, in which all nodes and edges with

- **`get_complement_subgraph(self) -> 'Explanation'`**
  Returns the induced subgraph, in which all nodes and edges with any

### `GraphGenerator`

An abstract base class for generating synthetic graphs.

#### Methods

- **`resolve(query: Any, *args: Any, **kwargs: Any) -> 'GraphGenerator'`**

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

### `MotifGenerator`

An abstract base class for generating a motif.

#### Methods

- **`resolve(query: Any, *args: Any, **kwargs: Any) -> 'MotifGenerator'`**

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
