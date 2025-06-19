# metric

Part of `torch_geometric.explain`
Module: `torch_geometric.explain.metric`

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
