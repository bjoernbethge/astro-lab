# map

Part of `torch_geometric.utils`
Module: `torch_geometric.utils.map`

## Functions (2)

### `from_dlpack(ext_tensor: Any) -> 'torch.Tensor'`

from_dlpack(ext_tensor) -> Tensor

Converts a tensor from an external library into a ``torch.Tensor``.

The returned PyTorch tensor will share the memory with the input tensor
(which may have come from another library). Note that in-place operations
will therefore also affect the data of the input tensor. This may lead to
unexpected issues (e.g., other libraries may have read-only flags or
immutable data structures), so the user should only do this if they know
for sure that this is fine.

Args:
    ext_tensor (object with ``__dlpack__`` attribute, or a DLPack capsule):
        The tensor or DLPack capsule to convert.

        If ``ext_tensor`` is a tensor (or ndarray) object, it must support
        the ``__dlpack__`` protocol (i.e., have a ``ext_tensor.__dlpack__``
        method). Otherwise ``ext_tensor`` may be a DLPack capsule, which is
        an opaque ``PyCapsule`` instance, typically produced by a
        ``to_dlpack`` function or method.

Examples::

    >>> import torch.utils.dlpack
    >>> t = torch.arange(4)

    # Convert a tensor directly (supported in PyTorch >= 1.10)
    >>> t2 = torch.from_dlpack(t)
    >>> t2[:2] = -1  # show that memory is shared
    >>> t2
    tensor([-1, -1,  2,  3])
    >>> t
    tensor([-1, -1,  2,  3])

    # The old-style DLPack usage, with an intermediate capsule object
    >>> capsule = torch.utils.dlpack.to_dlpack(t)
    >>> capsule
    <capsule object "dltensor" at ...>
    >>> t3 = torch.from_dlpack(capsule)
    >>> t3
    tensor([-1, -1,  2,  3])
    >>> t3[0] = -9  # now we're sharing memory between 3 tensors
    >>> t3
    tensor([-9, -1,  2,  3])
    >>> t2
    tensor([-9, -1,  2,  3])
    >>> t
    tensor([-9, -1,  2,  3])

### `map_index(src: torch.Tensor, index: torch.Tensor, max_index: Union[int, torch.Tensor, NoneType] = None, inclusive: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]`

Maps indices in :obj:`src` to the positional value of their
corresponding occurrence in :obj:`index`.
Indices must be strictly positive.

Args:
    src (torch.Tensor): The source tensor to map.
    index (torch.Tensor): The index tensor that denotes the new mapping.
    max_index (int, optional): The maximum index value.
        (default :obj:`None`)
    inclusive (bool, optional): If set to :obj:`True`, it is assumed that
        every entry in :obj:`src` has a valid entry in :obj:`index`.
        Can speed-up computation. (default: :obj:`False`)

:rtype: (:class:`torch.Tensor`, :class:`torch.BoolTensor`)

Examples:
    >>> src = torch.tensor([2, 0, 1, 0, 3])
    >>> index = torch.tensor([3, 2, 0, 1])

    >>> map_index(src, index)
    (tensor([1, 2, 3, 2, 0]), tensor([True, True, True, True, True]))

    >>> src = torch.tensor([2, 0, 1, 0, 3])
    >>> index = torch.tensor([3, 2, 0])

    >>> map_index(src, index)
    (tensor([1, 2, 2, 0]), tensor([True, True, False, True, True]))

.. note::

    If inputs are on GPU and :obj:`cudf` is available, consider using RMM
    for significant speed boosts.
    Proceed with caution as RMM may conflict with other allocators or
    fragments.

    .. code-block:: python

        import rmm
        rmm.reinitialize(pool_allocator=True)
        torch.cuda.memory.change_current_allocator(rmm.rmm_torch_allocator)

## Classes (1)

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
