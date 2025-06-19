# nn

Part of `torch_geometric.contrib`
Module: `torch_geometric.contrib.nn`

## Classes (2)

### `GRBCDAttack`

The Greedy Randomized Block Coordinate Descent (GRBCD) adversarial
attack from the `Robustness of Graph Neural Networks at Scale
<https://www.cs.cit.tum.de/daml/robustness-of-gnns-at-scale>`_ paper.

GRBCD shares most of the properties and requirements with
:class:`PRBCDAttack`. It also uses an efficient gradient based approach.
However, it greedily flips edges based on the gradient towards the
adjacency matrix.

.. note::
    For examples of using the GRBCD Attack, see
    `examples/contrib/rbcd_attack.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    contrib/rbcd_attack.py>`_
    for a test time attack (evasion).

Args:
    model (torch.nn.Module): The GNN module to assess.
    block_size (int): Number of randomly selected elements in the
        adjacency matrix to consider.
    epochs (int, optional): Number of epochs (aborts early if
        :obj:`mode='greedy'` and budget is satisfied) (default: :obj:`125`)
    loss (str or callable, optional): A loss to quantify the "strength" of
        an attack. Note that this function must match the output format of
        :attr:`model`. By default, it is assumed that the task is
        classification and that the model returns raw predictions (*i.e.*,
        no output activation) or uses :obj:`logsoftmax`. Moreover, and the
        number of predictions should match the number of labels passed to
        :attr:`attack`. Either pass Callable or one of: :obj:`'masked'`,
        :obj:`'margin'`, :obj:`'prob_margin'`, :obj:`'tanh_margin'`.
        (default: :obj:`'masked'`)
    is_undirected (bool, optional): If :obj:`True` the graph is
        assumed to be undirected. (default: :obj:`True`)
    log (bool, optional): If set to :obj:`False`, will not log any learning
        progress. (default: :obj:`True`)

### `PRBCDAttack`

The Projected Randomized Block Coordinate Descent (PRBCD) adversarial
attack from the `Robustness of Graph Neural Networks at Scale
<https://www.cs.cit.tum.de/daml/robustness-of-gnns-at-scale>`_ paper.

This attack uses an efficient gradient based approach that (during the
attack) relaxes the discrete entries in the adjacency matrix
:math:`\{0, 1\}` to :math:`[0, 1]` and solely perturbs the adjacency matrix
(no feature perturbations). Thus, this attack supports all models that can
handle weighted graphs that are differentiable w.r.t. these edge weights,
*e.g.*, :class:`~torch_geometric.nn.conv.GCNConv` or
:class:`~torch_geometric.nn.conv.GraphConv`. For non-differentiable models
you might need modifications, e.g., see example for
:class:`~torch_geometric.nn.conv.GATConv`.

The memory overhead is driven by the additional edges (at most
:attr:`block_size`). For scalability reasons, the block is drawn with
replacement and then the index is made unique. Thus, the actual block size
is typically slightly smaller than specified.

This attack can be used for both global and local attacks as well as
test-time attacks (evasion) and training-time attacks (poisoning). Please
see the provided examples.

This attack is designed with a focus on node- or graph-classification,
however, to adapt to other tasks you most likely only need to provide an
appropriate loss and model. However, we currently do not support batching
out of the box (sampling needs to be adapted).

.. note::
    For examples of using the PRBCD Attack, see
    `examples/contrib/rbcd_attack.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    contrib/rbcd_attack.py>`_
    for a test time attack (evasion) or
    `examples/contrib/rbcd_attack_poisoning.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    contrib/rbcd_attack_poisoning.py>`_
    for a training time (poisoning) attack.

Args:
    model (torch.nn.Module): The GNN module to assess.
    block_size (int): Number of randomly selected elements in the
        adjacency matrix to consider.
    epochs (int, optional): Number of epochs (aborts early if
        :obj:`mode='greedy'` and budget is satisfied) (default: :obj:`125`)
    epochs_resampling (int, optional): Number of epochs to resample the
        random block. (default: obj:`100`)
    loss (str or callable, optional): A loss to quantify the "strength" of
        an attack. Note that this function must match the output format of
        :attr:`model`. By default, it is assumed that the task is
        classification and that the model returns raw predictions (*i.e.*,
        no output activation) or uses :obj:`logsoftmax`. Moreover, and the
        number of predictions should match the number of labels passed to
        :attr:`attack`. Either pass a callable or one of: :obj:`'masked'`,
        :obj:`'margin'`, :obj:`'prob_margin'`, :obj:`'tanh_margin'`.
        (default: :obj:`'prob_margin'`)
    metric (callable, optional): Second (potentially
        non-differentiable) loss for monitoring or early stopping (if
        :obj:`mode='greedy'`). (default: same as :attr:`loss`)
    lr (float, optional): Learning rate for updating edge weights.
        Additionally, it is heuristically corrected for :attr:`block_size`,
        budget (see :attr:`attack`) and graph size. (default: :obj:`1_000`)
    is_undirected (bool, optional): If :obj:`True` the graph is
        assumed to be undirected. (default: :obj:`True`)
    log (bool, optional): If set to :obj:`False`, will not log any learning
        progress. (default: :obj:`True`)

#### Methods

- **`attack(self, x: torch.Tensor, edge_index: torch.Tensor, labels: torch.Tensor, budget: int, idx_attack: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]`**
  Attack the predictions for the provided model and graph.
