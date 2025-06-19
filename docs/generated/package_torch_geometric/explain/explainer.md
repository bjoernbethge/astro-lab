# explainer

Part of `torch_geometric.explain`
Module: `torch_geometric.explain.explainer`

## Functions (3)

### `clear_masks(model: torch.nn.modules.module.Module)`

Clear all masks from the model.

### `set_hetero_masks(model: torch.nn.modules.module.Module, mask_dict: Dict[Tuple[str, str, str], Union[torch.Tensor, torch.nn.parameter.Parameter]], edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor], apply_sigmoid: bool = True)`

Apply masks to every heterogeneous graph layer in the :obj:`model`
according to edge types.

### `set_masks(model: torch.nn.modules.module.Module, mask: Union[torch.Tensor, torch.nn.parameter.Parameter], edge_index: torch.Tensor, apply_sigmoid: bool = True)`

Apply mask to every graph layer in the :obj:`model`.

## Classes (14)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

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

### `ExplanationType`

Enum class for the explanation type.

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

### `MaskType`

Enum class for the mask type.

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

### `ModelMode`

Enum class for the model return type.

### `ModelReturnType`

Enum class for the model return type.

### `NodeType`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

#### Methods

- **`encode(self, /, encoding='utf-8', errors='strict')`**
  Encode the string using the codec registered for encoding.

- **`replace(self, old, new, count=-1, /)`**
  Return a copy with all occurrences of substring old replaced by new.

- **`split(self, /, sep=None, maxsplit=-1)`**
  Return a list of the substrings in the string, using sep as the separator string.

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.

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
