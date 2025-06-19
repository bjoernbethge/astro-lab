# module_dict

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.module_dict`

## Classes (2)

### `Module`

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

- **`forward(self, *input: Any) -> None`**
  Define the computation performed at every call.

- **`register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None`**
  Add a buffer to the module.

- **`register_parameter(self, name: str, param: Optional[torch.nn.parameter.Parameter]) -> None`**
  Add a parameter to the module.

### `ModuleDict`

Holds submodules in a dictionary.

:class:`~torch.nn.ModuleDict` can be indexed like a regular Python dictionary,
but modules it contains are properly registered, and will be visible by all
:class:`~torch.nn.Module` methods.

:class:`~torch.nn.ModuleDict` is an **ordered** dictionary that respects

* the order of insertion, and

* in :meth:`~torch.nn.ModuleDict.update`, the order of the merged
  ``OrderedDict``, ``dict`` (started from Python 3.6) or another
  :class:`~torch.nn.ModuleDict` (the argument to
  :meth:`~torch.nn.ModuleDict.update`).

Note that :meth:`~torch.nn.ModuleDict.update` with other unordered mapping
types (e.g., Python's plain ``dict`` before Python version 3.6) does not
preserve the order of the merged mapping.

Args:
    modules (iterable, optional): a mapping (dictionary) of (string: module)
        or an iterable of key-value pairs of type (string, module)

Example::

    class MyModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.choices = nn.ModuleDict({
                    'conv': nn.Conv2d(10, 10, 3),
                    'pool': nn.MaxPool2d(3)
            })
            self.activations = nn.ModuleDict([
                    ['lrelu', nn.LeakyReLU()],
                    ['prelu', nn.PReLU()]
            ])

        def forward(self, x, choice, act):
            x = self.choices[choice](x)
            x = self.activations[act](x)
            return x

#### Methods

- **`keys(self) -> Iterable[Union[str, Tuple[str, ...]]]`**
  Return an iterable of the ModuleDict keys.

- **`items(self) -> Iterable[Tuple[Union[str, Tuple[str, ...]], torch.nn.modules.module.Module]]`**
  Return an iterable of the ModuleDict key/value pairs.
