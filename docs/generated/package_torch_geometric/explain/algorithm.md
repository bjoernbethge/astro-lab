# algorithm

Part of `torch_geometric.explain`
Module: `torch_geometric.explain.algorithm`

## Classes (7)

### `AttentionExplainer`

An explainer that uses the attention coefficients produced by an
attention-based GNN (*e.g.*,
:class:`~torch_geometric.nn.conv.GATConv`,
:class:`~torch_geometric.nn.conv.GATv2Conv`, or
:class:`~torch_geometric.nn.conv.TransformerConv`) as edge explanation.
Attention scores across layers and heads will be aggregated according to
the :obj:`reduce` argument.

Args:
    reduce (str, optional): The method to reduce the attention scores
        across layers and heads. (default: :obj:`"max"`)

#### Methods

- **`forward(self, model: torch.nn.modules.module.Module, x: torch.Tensor, edge_index: torch.Tensor, *, target: torch.Tensor, index: Union[int, torch.Tensor, NoneType] = None, **kwargs) -> torch_geometric.explain.explanation.Explanation`**
  Computes the explanation.

- **`supports(self) -> bool`**
  Checks if the explainer supports the user-defined settings provided

### `CaptumExplainer`

A `Captum <https://captum.ai>`__-based explainer for identifying compact
subgraph structures and node features that play a crucial role in the
predictions made by a GNN.

This explainer algorithm uses :captum:`null` `Captum <https://captum.ai/>`_
to compute attributions.

Currently, the following attribution methods are supported:

* :class:`captum.attr.IntegratedGradients`
* :class:`captum.attr.Saliency`
* :class:`captum.attr.InputXGradient`
* :class:`captum.attr.Deconvolution`
* :class:`captum.attr.ShapleyValueSampling`
* :class:`captum.attr.GuidedBackprop`

Args:
    attribution_method (Attribution or str): The Captum attribution method
        to use. Can be a string or a :class:`captum.attr` method.
    **kwargs: Additional arguments for the Captum attribution method.

#### Methods

- **`forward(self, model: torch.nn.modules.module.Module, x: Union[torch.Tensor, Dict[str, torch.Tensor]], edge_index: Union[torch.Tensor, Dict[Tuple[str, str, str], torch.Tensor]], *, target: torch.Tensor, index: Union[int, torch.Tensor, NoneType] = None, **kwargs) -> Union[torch_geometric.explain.explanation.Explanation, torch_geometric.explain.explanation.HeteroExplanation]`**
  Computes the explanation.

- **`supports(self) -> bool`**
  Checks if the explainer supports the user-defined settings provided

### `DummyExplainer`

A dummy explainer that returns random explanations (useful for testing
purposes).

#### Methods

- **`forward(self, model: torch.nn.modules.module.Module, x: Union[torch.Tensor, Dict[str, torch.Tensor]], edge_index: Union[torch.Tensor, Dict[Tuple[str, str, str], torch.Tensor]], edge_attr: Union[torch.Tensor, Dict[Tuple[str, str, str], torch.Tensor], NoneType] = None, **kwargs) -> Union[torch_geometric.explain.explanation.Explanation, torch_geometric.explain.explanation.HeteroExplanation]`**
  Computes the explanation.

- **`supports(self) -> bool`**
  Checks if the explainer supports the user-defined settings provided

### `ExplainerAlgorithm`

An abstract base class for implementing explainer algorithms.

#### Methods

- **`forward(self, model: torch.nn.modules.module.Module, x: Union[torch.Tensor, Dict[str, torch.Tensor]], edge_index: Union[torch.Tensor, Dict[Tuple[str, str, str], torch.Tensor]], *, target: torch.Tensor, index: Union[int, torch.Tensor, NoneType] = None, **kwargs) -> Union[torch_geometric.explain.explanation.Explanation, torch_geometric.explain.explanation.HeteroExplanation]`**
  Computes the explanation.

- **`supports(self) -> bool`**
  Checks if the explainer supports the user-defined settings provided

- **`connect(self, explainer_config: torch_geometric.explain.config.ExplainerConfig, model_config: torch_geometric.explain.config.ModelConfig)`**
  Connects an explainer and model configuration to the explainer

### `GNNExplainer`

The GNN-Explainer model from the `"GNNExplainer: Generating
Explanations for Graph Neural Networks"
<https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
structures and node features that play a crucial role in the predictions
made by a GNN.

.. note::

    For an example of using :class:`GNNExplainer`, see
    `examples/explain/gnn_explainer.py <https://github.com/pyg-team/
    pytorch_geometric/blob/master/examples/explain/gnn_explainer.py>`_,
    `examples/explain/gnn_explainer_ba_shapes.py <https://github.com/
    pyg-team/pytorch_geometric/blob/master/examples/
    explain/gnn_explainer_ba_shapes.py>`_, and `examples/explain/
    gnn_explainer_link_pred.py <https://github.com/pyg-team/
    pytorch_geometric/blob/master/examples/explain/gnn_explainer_link_pred.py>`_.

.. note::

    The :obj:`edge_size` coefficient is multiplied by the number of nodes
    in the explanation at every iteration, and the resulting value is added
    to the loss as a regularization term, with the goal of producing
    compact explanations.
    A higher value will push the algorithm towards explanations with less
    elements.
    Consider adjusting the :obj:`edge_size` coefficient according to the
    average node degree in the dataset, especially if this value is bigger
    than in the datasets used in the original paper.

Args:
    epochs (int, optional): The number of epochs to train.
        (default: :obj:`100`)
    lr (float, optional): The learning rate to apply.
        (default: :obj:`0.01`)
    **kwargs (optional): Additional hyper-parameters to override default
        settings in
        :attr:`~torch_geometric.explain.algorithm.GNNExplainer.coeffs`.

#### Methods

- **`forward(self, model: torch.nn.modules.module.Module, x: torch.Tensor, edge_index: torch.Tensor, *, target: torch.Tensor, index: Union[int, torch.Tensor, NoneType] = None, **kwargs) -> torch_geometric.explain.explanation.Explanation`**
  Computes the explanation.

- **`supports(self) -> bool`**
  Checks if the explainer supports the user-defined settings provided

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

#### Methods

- **`forward(self, model: torch.nn.modules.module.Module, x: torch.Tensor, edge_index: torch.Tensor, *, target: torch.Tensor, index: Union[int, torch.Tensor, NoneType] = None, **kwargs) -> torch_geometric.explain.explanation.Explanation`**
  Computes the explanation.

- **`supports(self) -> bool`**
  Checks if the explainer supports the user-defined settings provided

- **`reset_parameters(self, input_dims: List[int], h_dim: List[int])`**
  Resets all learnable parameters of the module.

### `PGExplainer`

The PGExplainer model from the `"Parameterized Explainer for Graph
Neural Network" <https://arxiv.org/abs/2011.04573>`_ paper.

Internally, it utilizes a neural network to identify subgraph structures
that play a crucial role in the predictions made by a GNN.
Importantly, the :class:`PGExplainer` needs to be trained via
:meth:`~PGExplainer.train` before being able to generate explanations:

.. code-block:: python

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=30, lr=0.003),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=ModelConfig(...),
    )

    # Train against a variety of node-level or graph-level predictions:
    for epoch in range(30):
        for index in [...]:  # Indices to train against.
            loss = explainer.algorithm.train(epoch, model, x, edge_index,
                                             target=target, index=index)

    # Get the final explanations:
    explanation = explainer(x, edge_index, target=target, index=0)

Args:
    epochs (int): The number of epochs to train.
    lr (float, optional): The learning rate to apply.
        (default: :obj:`0.003`).
    **kwargs (optional): Additional hyper-parameters to override default
        settings in
        :attr:`~torch_geometric.explain.algorithm.PGExplainer.coeffs`.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`train(self, epoch: int, model: torch.nn.modules.module.Module, x: torch.Tensor, edge_index: torch.Tensor, *, target: torch.Tensor, index: Union[int, torch.Tensor, NoneType] = None, **kwargs)`**
  Trains the underlying explainer model.

- **`forward(self, model: torch.nn.modules.module.Module, x: torch.Tensor, edge_index: torch.Tensor, *, target: torch.Tensor, index: Union[int, torch.Tensor, NoneType] = None, **kwargs) -> torch_geometric.explain.explanation.Explanation`**
  Computes the explanation.
