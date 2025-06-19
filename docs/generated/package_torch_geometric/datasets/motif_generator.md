# motif_generator

Part of `torch_geometric.datasets`
Module: `torch_geometric.datasets.motif_generator`

## Classes (5)

### `CustomMotif`

Generates a motif based on a custom structure coming from a
:class:`torch_geometric.data.Data` or :class:`networkx.Graph` object.

Args:
    structure (torch_geometric.data.Data or networkx.Graph): The structure
        to use as a motif.

### `CycleMotif`

Generates the cycle motif from the `"GNNExplainer:
Generating Explanations for Graph Neural Networks"
<https://arxiv.org/abs/1903.03894>`__ paper.

Args:
    num_nodes (int): The number of nodes in the cycle.

### `GridMotif`

Generates the grid-structured motif from the
`"GNNExplainer: Generating Explanations for Graph Neural Networks"
<https://arxiv.org/abs/1903.03894>`__ paper.

### `HouseMotif`

Generates the house-structured motif from the `"GNNExplainer:
Generating Explanations for Graph Neural Networks"
<https://arxiv.org/abs/1903.03894>`__ paper, containing 5 nodes and 6
undirected edges. Nodes are labeled according to their structural role:
the top, middle and bottom of the house.

### `MotifGenerator`

An abstract base class for generating a motif.

#### Methods

- **`resolve(query: Any, *args: Any, **kwargs: Any) -> 'MotifGenerator'`**
