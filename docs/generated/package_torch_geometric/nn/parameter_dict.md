# parameter_dict

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.parameter_dict`

## Classes (2)

### `Parameter`

A kind of Tensor that is to be considered a module parameter.

Parameters are :class:`~torch.Tensor` subclasses, that have a
very special property when used with :class:`Module` s - when they're
assigned as Module attributes they are automatically added to the list of
its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
Assigning a Tensor doesn't have such effect. This is because one might
want to cache some temporary state, like last hidden state of the RNN, in
the model. If there was no such class as :class:`Parameter`, these
temporaries would get registered too.

Args:
    data (Tensor): parameter tensor.
    requires_grad (bool, optional): if the parameter requires gradient. Note that
        the torch.no_grad() context does NOT affect the default behavior of
        Parameter creation--the Parameter will still have `requires_grad=True` in
        :class:`~no_grad` mode. See :ref:`locally-disable-grad-doc` for more
        details. Default: `True`

### `ParameterDict`

Holds parameters in a dictionary.

ParameterDict can be indexed like a regular Python dictionary, but Parameters it
contains are properly registered, and will be visible by all Module methods.
Other objects are treated as would be done by a regular Python dictionary

:class:`~torch.nn.ParameterDict` is an **ordered** dictionary.
:meth:`~torch.nn.ParameterDict.update` with other unordered mapping
types (e.g., Python's plain ``dict``) does not preserve the order of the
merged mapping. On the other hand, ``OrderedDict`` or another :class:`~torch.nn.ParameterDict`
will preserve their ordering.

Note that the constructor, assigning an element of the dictionary and the
:meth:`~torch.nn.ParameterDict.update` method will convert any :class:`~torch.Tensor` into
:class:`~torch.nn.Parameter`.

Args:
    values (iterable, optional): a mapping (dictionary) of
        (string : Any) or an iterable of key-value pairs
        of type (string, Any)

Example::

    class MyModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.params = nn.ParameterDict({
                    'left': nn.Parameter(torch.randn(5, 10)),
                    'right': nn.Parameter(torch.randn(5, 10))
            })

        def forward(self, x, choice):
            x = self.params[choice].mm(x)
            return x

#### Methods

- **`keys(self) -> Iterable[Union[str, Tuple[str, ...]]]`**
  Return an iterable of the ParameterDict keys.

- **`items(self) -> Iterable[Tuple[Union[str, Tuple[str, ...]], torch.nn.parameter.Parameter]]`**
  Return an iterable of the ParameterDict key/value pairs.
