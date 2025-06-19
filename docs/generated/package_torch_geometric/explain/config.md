# config

Part of `torch_geometric.explain`
Module: `torch_geometric.explain.config`

## Functions (1)

### `dataclass(cls=None, /, *, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False, weakref_slot=False)`

Add dunder methods based on the fields defined in the class.

Examines PEP 526 __annotations__ to determine fields.

If init is true, an __init__() method is added to the class. If repr
is true, a __repr__() method is added. If order is true, rich
comparison dunder methods are added. If unsafe_hash is true, a
__hash__() method is added. If frozen is true, fields may not be
assigned to after instance creation. If match_args is true, the
__match_args__ tuple is added. If kw_only is true, then by default
all fields are keyword-only. If slots is true, a new class with a
__slots__ attribute is returned.

## Classes (11)

### `CastMixin`

### `Enum`

Create a collection of name/value pairs.

Example enumeration:

>>> class Color(Enum):
...     RED = 1
...     BLUE = 2
...     GREEN = 3

Access them by:

- attribute access::

>>> Color.RED
<Color.RED: 1>

- value lookup:

>>> Color(1)
<Color.RED: 1>

- name lookup:

>>> Color['RED']
<Color.RED: 1>

Enumerations can be iterated over, and know how many members they have:

>>> len(Color)
3

>>> list(Color)
[<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]

Methods can be added to enumerations, and members can have their own
attributes -- see the documentation for details.

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

### `ExplanationType`

Enum class for the explanation type.

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

### `ModelTaskLevel`

Enum class for the model task level.

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

### `ThresholdType`

Enum class for the threshold type.
