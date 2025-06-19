# nlp

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.nlp`

## Classes (2)

### `LLM`

A wrapper around a Large Language Model (LLM) from HuggingFace.

model_name (str): The HuggingFace model name, *e.g.*, :obj:`"llama2"` or
    :obj:`"gemma"`.
num_params (int): An integer representing how many parameters the
    HuggingFace model has, in billions. This is used to automatically
    allocate the correct number of GPUs needed, given the available GPU
    memory of your GPUs.
dtype (torch.dtype, optional): The data type to use for the LLM.
    (default :obj: `torch.bloat16`)

#### Methods

- **`forward(self, question: List[str], answer: List[str], context: Optional[List[str]] = None, embedding: Optional[List[torch.Tensor]] = None) -> torch.Tensor`**
  The forward pass.

- **`inference(self, question: List[str], context: Optional[List[str]] = None, embedding: Optional[List[torch.Tensor]] = None, max_tokens: Optional[int] = 32) -> List[str]`**
  The inference pass.

### `SentenceTransformer`

Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the submodules as regular attributes::

    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will also have their
parameters converted when you call :meth:`to`, etc.

.. note::
    As per the example above, an ``__init__()`` call to the parent class
    must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or
                evaluation mode.
:vartype training: bool

#### Methods

- **`forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor`**
  Define the computation performed at every call.

- **`encode(self, text: List[str], batch_size: Optional[int] = None, output_device: Union[str, torch.device, NoneType] = None) -> torch.Tensor`**
