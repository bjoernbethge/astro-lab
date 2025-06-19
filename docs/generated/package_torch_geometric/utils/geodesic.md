# geodesic

Part of `torch_geometric.utils`
Module: `torch_geometric.utils.geodesic`

## Functions (1)

### `geodesic_distance(pos: torch.Tensor, face: torch.Tensor, src: Optional[torch.Tensor] = None, dst: Optional[torch.Tensor] = None, norm: bool = True, max_distance: Optional[float] = None, num_workers: int = 0, **kwargs: Optional[torch.Tensor]) -> torch.Tensor`

Computes (normalized) geodesic distances of a mesh given by :obj:`pos`
and :obj:`face`. If :obj:`src` and :obj:`dst` are given, this method only
computes the geodesic distances for the respective source and target
node-pairs.

.. note::

    This function requires the :obj:`gdist` package.
    To install, run :obj:`pip install cython && pip install gdist`.

Args:
    pos (torch.Tensor): The node positions.
    face (torch.Tensor): The face indices.
    src (torch.Tensor, optional): If given, only compute geodesic distances
        for the specified source indices. (default: :obj:`None`)
    dst (torch.Tensor, optional): If given, only compute geodesic distances
        for the specified target indices. (default: :obj:`None`)
    norm (bool, optional): Normalizes geodesic distances by
        :math:`\sqrt{\textrm{area}(\mathcal{M})}`. (default: :obj:`True`)
    max_distance (float, optional): If given, only yields results for
        geodesic distances less than :obj:`max_distance`. This will speed
        up runtime dramatically. (default: :obj:`None`)
    num_workers (int, optional): How many subprocesses to use for
        calculating geodesic distances.
        :obj:`0` means that computation takes place in the main process.
        :obj:`-1` means that the available amount of CPU cores is used.
        (default: :obj:`0`)

:rtype: :class:`Tensor`

Example:
    >>> pos = torch.tensor([[0.0, 0.0, 0.0],
    ...                     [2.0, 0.0, 0.0],
    ...                     [0.0, 2.0, 0.0],
    ...                     [2.0, 2.0, 0.0]])
    >>> face = torch.tensor([[0, 0],
    ...                      [1, 2],
    ...                      [3, 3]])
    >>> geodesic_distance(pos, face)
    [[0, 1, 1, 1.4142135623730951],
    [1, 0, 1.4142135623730951, 1],
    [1, 1.4142135623730951, 0, 1],
    [1.4142135623730951, 1, 1, 0]]

## Classes (1)

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
