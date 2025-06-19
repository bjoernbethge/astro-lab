# explain

Part of `torch_geometric.contrib`
Module: `torch_geometric.contrib.explain`

## Functions (1)

### `deprecated(details: Optional[str] = None, func_name: Optional[str] = None) -> Callable`

## Classes (3)

### `GraphMaskExplainer`

The GraphMask-Explainer model from the `"Interpreting Graph Neural
Networks for NLP With Differentiable Edge Masking"
<https://arxiv.org/abs/2010.00577>`_ paper for identifying layer-wise
compact subgraph structures and node features that play a crucial role in
the predictions made by a GNN.

.. note::
    For an example of using :class:`GraphMaskExplainer`,
    see `examples/explain/graphmask_explainer.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    /explain/graphmask_explainer.py>`_.

    A working real-time example of :class:`GraphMaskExplainer` in the form
    of a deployed app can be accessed `here
    <https://graph-explainability.streamlit.app/>`_.

Args:
    num_layers (int): The number of layers to use.
    epochs (int, optional): The number of epochs to train.
        (default: :obj:`100`)
    lr (float, optional): The learning rate to apply.
        (default: :obj:`0.01`)
    penalty_scaling (int, optional): Scaling value of penalty term. Value
        must lie between 0 and 10. (default: :obj:`5`)
    lambda_optimizer_lr (float, optional): The learning rate to optimize
        the Lagrange multiplier. (default: :obj:`1e-2`)
    init_lambda (float, optional): The Lagrange multiplier. Value must lie
        between :obj:`0` and `1`. (default: :obj:`0.55`)
    allowance (float, optional): A float value between :obj:`0` and
        :obj:`1` denotes tolerance level. (default: :obj:`0.03`)
    log (bool, optional): If set to :obj:`False`, will not log any
        learning progress. (default: :obj:`True`)
    **kwargs (optional): Additional hyper-parameters to override default
        settings in
        :attr:`~torch_geometric.nn.models.GraphMaskExplainer.coeffs`.

### `NewGraphMaskExplainer`

The GraphMask-Explainer model from the `"Interpreting Graph Neural
Networks for NLP With Differentiable Edge Masking"
<https://arxiv.org/abs/2010.00577>`_ paper for identifying layer-wise
compact subgraph structures and node features that play a crucial role in
the predictions made by a GNN.

.. note::
    For an example of using :class:`GraphMaskExplainer`,
    see `examples/explain/graphmask_explainer.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    /explain/graphmask_explainer.py>`_.

    A working real-time example of :class:`GraphMaskExplainer` in the form
    of a deployed app can be accessed `here
    <https://graph-explainability.streamlit.app/>`_.

Args:
    num_layers (int): The number of layers to use.
    epochs (int, optional): The number of epochs to train.
        (default: :obj:`100`)
    lr (float, optional): The learning rate to apply.
        (default: :obj:`0.01`)
    penalty_scaling (int, optional): Scaling value of penalty term. Value
        must lie between 0 and 10. (default: :obj:`5`)
    lambda_optimizer_lr (float, optional): The learning rate to optimize
        the Lagrange multiplier. (default: :obj:`1e-2`)
    init_lambda (float, optional): The Lagrange multiplier. Value must lie
        between :obj:`0` and `1`. (default: :obj:`0.55`)
    allowance (float, optional): A float value between :obj:`0` and
        :obj:`1` denotes tolerance level. (default: :obj:`0.03`)
    log (bool, optional): If set to :obj:`False`, will not log any
        learning progress. (default: :obj:`True`)
    **kwargs (optional): Additional hyper-parameters to override default
        settings in
        :attr:`~torch_geometric.nn.models.GraphMaskExplainer.coeffs`.

#### Methods

- **`forward(self, model: torch.nn.modules.module.Module, x: torch.Tensor, edge_index: torch.Tensor, *, target: torch.Tensor, index: Union[int, torch.Tensor, NoneType] = None, **kwargs) -> torch_geometric.explain.explanation.Explanation`**
  Computes the explanation.

- **`supports(self) -> bool`**
  Checks if the explainer supports the user-defined settings provided

- **`reset_parameters(self, input_dims: List[int], h_dim: List[int])`**
  Resets all learnable parameters of the module.

### `PGMExplainer`

The PGMExplainer model from the `"PGMExplainer: Probabilistic
Graphical Model Explanations  for Graph Neural Networks"
<https://arxiv.org/abs/1903.03894>`_ paper.

The generated :class:`~torch_geometric.explain.Explanation` provides a
:obj:`node_mask` and a :obj:`pgm_stats` tensor, which stores the
:math:`p`-values of each node as calculated by the Chi-squared test.

Args:
    feature_index (List): The indices of the perturbed features. If set
        to :obj:`None`, all features are perturbed. (default: :obj:`None`)
    perturb_mode (str, optional): The method to generate the variations in
        features. One of :obj:`"randint"`, :obj:`"mean"`, :obj:`"zero"`,
        :obj:`"max"` or :obj:`"uniform"`. (default: :obj:`"randint"`)
    perturbations_is_positive_only (bool, optional): If set to :obj:`True`,
        restrict perturbed values to be positive. (default: :obj:`False`)
    is_perturbation_scaled (bool, optional): If set to :obj:`True`, will
        normalize the range of the perturbed features.
        (default: :obj:`False`)
    num_samples (int, optional): The number of samples of perturbations
        used to test the significance of nodes to the prediction.
        (default: :obj:`100`)
    max_subgraph_size (int, optional): The maximum number of neighbors to
        consider for the explanation. (default: :obj:`None`)
    significance_threshold (float, optional): The statistical threshold
        (:math:`p`-value) for which a node is considered to have an effect
        on the prediction. (default: :obj:`0.05`)
    pred_threshold (float, optional): The buffer value (in range
        :obj:`[0, 1]`) to consider the output from a perturbed data to be
        different from the original. (default: :obj:`0.1`)

#### Methods

- **`forward(self, model: torch.nn.modules.module.Module, x: torch.Tensor, edge_index: torch.Tensor, *, target: torch.Tensor, index: Union[int, torch.Tensor, NoneType] = None, **kwargs) -> torch_geometric.explain.explanation.Explanation`**
  Computes the explanation.

- **`supports(self) -> bool`**
  Checks if the explainer supports the user-defined settings provided
