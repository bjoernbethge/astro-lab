# functional

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.functional`

## Description

Functional operator package.

## Functions (2)

### `bro(x: torch.Tensor, batch: torch.Tensor, p: Union[int, str] = 2) -> torch.Tensor`

The Batch Representation Orthogonality penalty from the `"Improving
Molecular Graph Neural Network Explainability with Orthonormalization
and Induced Sparsity" <https://arxiv.org/abs/2105.04854>`_ paper.

Computes a regularization for each graph representation in a mini-batch
according to

.. math::
    \mathcal{L}_{\textrm{BRO}}^\mathrm{graph} =
      || \mathbf{HH}^T - \mathbf{I}||_p

and returns an average over all graphs in the batch.

Args:
    x (torch.Tensor): The node feature matrix.
    batch (torch.Tensor): The batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
        each node to a specific example.
    p (int or str, optional): The norm order to use. (default: :obj:`2`)

### `gini(w: torch.Tensor) -> torch.Tensor`

The Gini coefficient from the `"Improving Molecular Graph Neural
Network Explainability with Orthonormalization and Induced Sparsity"
<https://arxiv.org/abs/2105.04854>`_ paper.

Computes a regularization penalty :math:`\in [0, 1]` for each row of a
matrix according to

.. math::
    \mathcal{L}_\textrm{Gini}^i = \sum_j^n \sum_{j'}^n \frac{|w_{ij}
     - w_{ij'}|}{2 (n^2 - n)\bar{w_i}}

and returns an average over all rows.

Args:
    w (torch.Tensor): A two-dimensional tensor.
