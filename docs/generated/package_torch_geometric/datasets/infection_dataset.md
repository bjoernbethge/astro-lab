# infection_dataset

Part of `torch_geometric.datasets`
Module: `torch_geometric.datasets.infection_dataset`

## Functions (1)

### `k_hop_subgraph(node_idx: Union[int, List[int], torch.Tensor], num_hops: int, edge_index: torch.Tensor, relabel_nodes: bool = False, num_nodes: Optional[int] = None, flow: str = 'source_to_target', directed: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`

Computes the induced subgraph of :obj:`edge_index` around all nodes in
:attr:`node_idx` reachable within :math:`k` hops.

The :attr:`flow` argument denotes the direction of edges for finding
:math:`k`-hop neighbors. If set to :obj:`"source_to_target"`, then the
method will find all neighbors that point to the initial set of seed nodes
in :attr:`node_idx.`
This mimics the natural flow of message passing in Graph Neural Networks.

The method returns (1) the nodes involved in the subgraph, (2) the filtered
:obj:`edge_index` connectivity, (3) the mapping from node indices in
:obj:`node_idx` to their new location, and (4) the edge mask indicating
which edges were preserved.

Args:
    node_idx (int, list, tuple or :obj:`torch.Tensor`): The central seed
        node(s).
    num_hops (int): The number of hops :math:`k`.
    edge_index (LongTensor): The edge indices.
    relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
        :obj:`edge_index` will be relabeled to hold consecutive indices
        starting from zero. (default: :obj:`False`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    flow (str, optional): The flow direction of :math:`k`-hop aggregation
        (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
        (default: :obj:`"source_to_target"`)
    directed (bool, optional): If set to :obj:`True`, will only include
        directed edges to the seed nodes :obj:`node_idx`.
        (default: :obj:`False`)

:rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
         :class:`BoolTensor`)

Examples:
    >>> edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
    ...                            [2, 2, 4, 4, 6, 6]])

    >>> # Center node 6, 2-hops
    >>> subset, edge_index, mapping, edge_mask = k_hop_subgraph(
    ...     6, 2, edge_index, relabel_nodes=True)
    >>> subset
    tensor([2, 3, 4, 5, 6])
    >>> edge_index
    tensor([[0, 1, 2, 3],
            [2, 2, 4, 4]])
    >>> mapping
    tensor([4])
    >>> edge_mask
    tensor([False, False,  True,  True,  True,  True])
    >>> subset[mapping]
    tensor([6])

    >>> edge_index = torch.tensor([[1, 2, 4, 5],
    ...                            [0, 1, 5, 6]])
    >>> (subset, edge_index,
    ...  mapping, edge_mask) = k_hop_subgraph([0, 6], 2,
    ...                                       edge_index,
    ...                                       relabel_nodes=True)
    >>> subset
    tensor([0, 1, 2, 4, 5, 6])
    >>> edge_index
    tensor([[1, 2, 3, 4],
            [0, 1, 4, 5]])
    >>> mapping
    tensor([0, 5])
    >>> edge_mask
    tensor([True, True, True, True])
    >>> subset[mapping]
    tensor([0, 6])

## Classes (5)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

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
