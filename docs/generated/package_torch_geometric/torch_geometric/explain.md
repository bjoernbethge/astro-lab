# explain

Part of `torch_geometric.torch_geometric`
Module: `torch_geometric.explain`

## Functions (5)

### `characterization_score(pos_fidelity: torch.Tensor, neg_fidelity: torch.Tensor, pos_weight: float = 0.5, neg_weight: float = 0.5) -> torch.Tensor`

Returns the componentwise characterization score as described in the
`"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for
Graph Neural Networks" <https://arxiv.org/abs/2206.09677>`_ paper.

..  math::
   \textrm{charact} = \frac{w_{+} + w_{-}}{\frac{w_{+}}{\textrm{fid}_{+}} +
    \frac{w_{-}}{1 - \textrm{fid}_{-}}}

Args:
    pos_fidelity (torch.Tensor): The positive fidelity
        :math:`\textrm{fid}_{+}`.
    neg_fidelity (torch.Tensor): The negative fidelity
        :math:`\textrm{fid}_{-}`.
    pos_weight (float, optional): The weight :math:`w_{+}` for
        :math:`\textrm{fid}_{+}`. (default: :obj:`0.5`)
    neg_weight (float, optional): The weight :math:`w_{-}` for
        :math:`\textrm{fid}_{-}`. (default: :obj:`0.5`)

### `fidelity(explainer: torch_geometric.explain.explainer.Explainer, explanation: torch_geometric.explain.explanation.Explanation) -> Tuple[float, float]`

Evaluates the fidelity of an
:class:`~torch_geometric.explain.Explainer` given an
:class:`~torch_geometric.explain.Explanation`, as described in the
`"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for
Graph Neural Networks" <https://arxiv.org/abs/2206.09677>`_ paper.

Fidelity evaluates the contribution of the produced explanatory subgraph
to the initial prediction, either by giving only the subgraph to the model
(fidelity-) or by removing it from the entire graph (fidelity+).
The fidelity scores capture how good an explainable model reproduces the
natural phenomenon or the GNN model logic.

For **phenomenon** explanations, the fidelity scores are given by:

.. math::
    \textrm{fid}_{+} &= \frac{1}{N} \sum_{i = 1}^N
    \| \mathbb{1}(\hat{y}_i = y_i) -
    \mathbb{1}( \hat{y}_i^{G_{C \setminus S}} = y_i) \|

    \textrm{fid}_{-} &= \frac{1}{N} \sum_{i = 1}^N
    \| \mathbb{1}(\hat{y}_i = y_i) -
    \mathbb{1}( \hat{y}_i^{G_S} = y_i) \|

For **model** explanations, the fidelity scores are given by:

.. math::
    \textrm{fid}_{+} &= 1 - \frac{1}{N} \sum_{i = 1}^N
    \mathbb{1}( \hat{y}_i^{G_{C \setminus S}} = \hat{y}_i)

    \textrm{fid}_{-} &= 1 - \frac{1}{N} \sum_{i = 1}^N
    \mathbb{1}( \hat{y}_i^{G_S} = \hat{y}_i)

Args:
    explainer (Explainer): The explainer to evaluate.
    explanation (Explanation): The explanation to evaluate.

### `fidelity_curve_auc(pos_fidelity: torch.Tensor, neg_fidelity: torch.Tensor, x: torch.Tensor) -> torch.Tensor`

Returns the AUC for the fidelity curve as described in the
`"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for
Graph Neural Networks" <https://arxiv.org/abs/2206.09677>`_ paper.

More precisely, returns the AUC of

.. math::
    f(x) = \frac{\textrm{fid}_{+}}{1 - \textrm{fid}_{-}}

Args:
    pos_fidelity (torch.Tensor): The positive fidelity
        :math:`\textrm{fid}_{+}`.
    neg_fidelity (torch.Tensor): The negative fidelity
        :math:`\textrm{fid}_{-}`.
    x (torch.Tensor): Tensor containing the points on the :math:`x`-axis.
        Needs to be sorted in ascending order.

### `groundtruth_metrics(pred_mask: torch.Tensor, target_mask: torch.Tensor, metrics: Union[str, List[str], NoneType] = None, threshold: float = 0.5) -> Union[float, Tuple[float, ...]]`

Compares and evaluates an explanation mask with the ground-truth
explanation mask.

Args:
    pred_mask (torch.Tensor): The prediction mask to evaluate.
    target_mask (torch.Tensor): The ground-truth target mask.
    metrics (str or List[str], optional): The metrics to return
        (:obj:`"accuracy"`, :obj:`"recall"`, :obj:`"precision"`,
        :obj:`"f1_score"`, :obj:`"auroc"`). (default: :obj:`["accuracy",
        "recall", "precision", "f1_score", "auroc"]`)
    threshold (float, optional): The threshold value to perform hard
        thresholding of :obj:`mask` and :obj:`groundtruth`.
        (default: :obj:`0.5`)

### `unfaithfulness(explainer: torch_geometric.explain.explainer.Explainer, explanation: torch_geometric.explain.explanation.Explanation, top_k: Optional[int] = None) -> float`

Evaluates how faithful an :class:`~torch_geometric.explain.Explanation`
is to an underyling GNN predictor, as described in the
`"Evaluating Explainability for Graph Neural Networks"
<https://arxiv.org/abs/2208.09339>`_ paper.

In particular, the graph explanation unfaithfulness metric is defined as

.. math::
    \textrm{GEF}(y, \hat{y}) = 1 - \exp(- \textrm{KL}(y || \hat{y}))

where :math:`y` refers to the prediction probability vector obtained from
the original graph, and :math:`\hat{y}` refers to the prediction
probability vector obtained from the masked subgraph.
Finally, the Kullback-Leibler (KL) divergence score quantifies the distance
between the two probability distributions.

Args:
    explainer (Explainer): The explainer to evaluate.
    explanation (Explanation): The explanation to evaluate.
    top_k (int, optional): If set, will only keep the original values of
        the top-:math:`k` node features identified by an explanation.
        If set to :obj:`None`, will use :obj:`explanation.node_mask` as it
        is for masking node features. (default: :obj:`None`)

## Classes (13)

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

### `Explainer`

An explainer class for instance-level explanations of Graph Neural
Networks.

Args:
    model (torch.nn.Module): The model to explain.
    algorithm (ExplainerAlgorithm): The explanation algorithm.
    explanation_type (ExplanationType or str): The type of explanation to
        compute. The possible values are:

            - :obj:`"model"`: Explains the model prediction.

            - :obj:`"phenomenon"`: Explains the phenomenon that the model
              is trying to predict.

        In practice, this means that the explanation algorithm will either
        compute their losses with respect to the model output
        (:obj:`"model"`) or the target output (:obj:`"phenomenon"`).
    model_config (ModelConfig): The model configuration.
        See :class:`~torch_geometric.explain.config.ModelConfig` for
        available options. (default: :obj:`None`)
    node_mask_type (MaskType or str, optional): The type of mask to apply
        on nodes. The possible values are (default: :obj:`None`):

            - :obj:`None`: Will not apply any mask on nodes.

            - :obj:`"object"`: Will mask each node.

            - :obj:`"common_attributes"`: Will mask each feature.

            - :obj:`"attributes"`: Will mask each feature across all nodes.

    edge_mask_type (MaskType or str, optional): The type of mask to apply
        on edges. Has the sample possible values as :obj:`node_mask_type`.
        (default: :obj:`None`)
    threshold_config (ThresholdConfig, optional): The threshold
        configuration.
        See :class:`~torch_geometric.explain.config.ThresholdConfig` for
        available options. (default: :obj:`None`)

#### Methods

- **`get_prediction(self, *args, **kwargs) -> torch.Tensor`**
  Returns the prediction of the model on the input graph.

- **`get_masked_prediction(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]], edge_index: Union[torch.Tensor, Dict[Tuple[str, str, str], torch.Tensor]], node_mask: Union[torch.Tensor, Dict[str, torch.Tensor], NoneType] = None, edge_mask: Union[torch.Tensor, Dict[Tuple[str, str, str], torch.Tensor], NoneType] = None, **kwargs) -> torch.Tensor`**
  Returns the prediction of the model on the input graph with node

- **`get_target(self, prediction: torch.Tensor) -> torch.Tensor`**
  Returns the target of the model from a given prediction.

### `ExplainerAlgorithm`

An abstract base class for implementing explainer algorithms.

#### Methods

- **`forward(self, model: torch.nn.modules.module.Module, x: Union[torch.Tensor, Dict[str, torch.Tensor]], edge_index: Union[torch.Tensor, Dict[Tuple[str, str, str], torch.Tensor]], *, target: torch.Tensor, index: Union[int, torch.Tensor, NoneType] = None, **kwargs) -> Union[torch_geometric.explain.explanation.Explanation, torch_geometric.explain.explanation.HeteroExplanation]`**
  Computes the explanation.

- **`supports(self) -> bool`**
  Checks if the explainer supports the user-defined settings provided

- **`connect(self, explainer_config: torch_geometric.explain.config.ExplainerConfig, model_config: torch_geometric.explain.config.ModelConfig)`**
  Connects an explainer and model configuration to the explainer

### `ExplainerConfig`

Configuration class to store and validate high level explanation
parameters.

Args:
    explanation_type (ExplanationType or str): The type of explanation to
        compute. The possible values are:

            - :obj:`"model"`: Explains the model prediction.

            - :obj:`"phenomenon"`: Explains the phenomenon that the model
              is trying to predict.

        In practice, this means that the explanation algorithm will either
        compute their losses with respect to the model output
        (:obj:`"model"`) or the target output (:obj:`"phenomenon"`).

    node_mask_type (MaskType or str, optional): The type of mask to apply
        on nodes. The possible values are (default: :obj:`None`):

            - :obj:`None`: Will not apply any mask on nodes.

            - :obj:`"object"`: Will mask each node.

            - :obj:`"common_attributes"`: Will mask each feature.

            - :obj:`"attributes"`: Will mask each feature across all nodes.

    edge_mask_type (MaskType or str, optional): The type of mask to apply
        on edges. Has the sample possible values as :obj:`node_mask_type`.
        (default: :obj:`None`)

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

### `HeteroExplanation`

Holds all the obtained explanations of a heterogeneous graph.

The explanation object is a :obj:`~torch_geometric.data.HeteroData` object
and can hold node attributions and edge attributions.
It can also hold the original graph if needed.

#### Methods

- **`validate(self, raise_on_error: bool = True) -> bool`**
  Validates the correctness of the :class:`Explanation` object.

- **`get_explanation_subgraph(self) -> 'HeteroExplanation'`**
  Returns the induced subgraph, in which all nodes and edges with

- **`get_complement_subgraph(self) -> 'HeteroExplanation'`**
  Returns the induced subgraph, in which all nodes and edges with any

### `ModelConfig`

Configuration class to store model parameters.

Args:
    mode (ModelMode or str): The mode of the model. The possible values
        are:

            - :obj:`"binary_classification"`: A binary classification
              model.

            - :obj:`"multiclass_classification"`: A multiclass
              classification model.

            - :obj:`"regression"`: A regression model.

    task_level (ModelTaskLevel or str): The task-level of the model.
        The possible values are:

            - :obj:`"node"`: A node-level prediction model.

            - :obj:`"edge"`: An edge-level prediction model.

            - :obj:`"graph"`: A graph-level prediction model.

    return_type (ModelReturnType or str, optional): The return type of the
        model. The possible values are (default: :obj:`None`):

            - :obj:`"raw"`: The model returns raw values.

            - :obj:`"probs"`: The model returns probabilities.

            - :obj:`"log_probs"`: The model returns log-probabilities.

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

### `ThresholdConfig`

Configuration class to store and validate threshold parameters.

Args:
    threshold_type (ThresholdType or str): The type of threshold to apply.
        The possible values are:

            - :obj:`None`: No threshold is applied.

            - :obj:`"hard"`: A hard threshold is applied to each mask.
              The elements of the mask with a value below the :obj:`value`
              are set to :obj:`0`, the others are set to :obj:`1`.

            - :obj:`"topk"`: A soft threshold is applied to each mask.
              The top obj:`value` elements of each mask are kept, the
              others are set to :obj:`0`.

            - :obj:`"topk_hard"`: Same as :obj:`"topk"` but values are set
              to :obj:`1` for all elements which are kept.

    value (int or float, optional): The value to use when thresholding.
        (default: :obj:`None`)
