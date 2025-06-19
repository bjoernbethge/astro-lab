# unpool

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.unpool`

## Description

Unpooling package.

## Functions (1)

### `knn_interpolate(x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor, batch_x: Optional[torch.Tensor] = None, batch_y: Optional[torch.Tensor] = None, k: int = 3, num_workers: int = 1)`

The k-NN interpolation from the `"PointNet++: Deep Hierarchical
Feature Learning on Point Sets in a Metric Space"
<https://arxiv.org/abs/1706.02413>`_ paper.

For each point :math:`y` with position :math:`\mathbf{p}(y)`, its
interpolated features :math:`\mathbf{f}(y)` are given by

.. math::
    \mathbf{f}(y) = \frac{\sum_{i=1}^k w(x_i) \mathbf{f}(x_i)}{\sum_{i=1}^k
    w(x_i)} \textrm{, where } w(x_i) = \frac{1}{d(\mathbf{p}(y),
    \mathbf{p}(x_i))^2}

and :math:`\{ x_1, \ldots, x_k \}` denoting the :math:`k` nearest points
to :math:`y`.

Args:
    x (torch.Tensor): Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
    pos_x (torch.Tensor): Node position matrix
        :math:`\in \mathbb{R}^{N \times d}`.
    pos_y (torch.Tensor): Upsampled node position matrix
        :math:`\in \mathbb{R}^{M \times d}`.
    batch_x (torch.Tensor, optional): Batch vector
        :math:`\mathbf{b_x} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
        each node from :math:`\mathbf{X}` to a specific example.
        (default: :obj:`None`)
    batch_y (torch.Tensor, optional): Batch vector
        :math:`\mathbf{b_y} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
        each node from :math:`\mathbf{Y}` to a specific example.
        (default: :obj:`None`)
    k (int, optional): Number of neighbors. (default: :obj:`3`)
    num_workers (int, optional): Number of workers to use for computation.
        Has no effect in case :obj:`batch_x` or :obj:`batch_y` is not
        :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
