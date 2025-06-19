# base

Part of `torch_geometric.loader`
Module: `torch_geometric.loader.base`

## Classes (2)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `DataLoaderIterator`

A data loader iterator extended by a simple post transformation
function :meth:`transform_fn`. While the iterator may request items from
different sub-processes, :meth:`transform_fn` will always be executed in
the main process.

This iterator is used in PyG's sampler classes, and is responsible for
feature fetching and filtering data objects after sampling has taken place
in a sub-process. This has the following advantages:

* We do not need to share feature matrices across processes which may
  prevent any errors due to too many open file handles.
* We can execute any expensive post-processing commands on the main thread
  with full parallelization power (which usually executes faster).
* It lets us naturally support data already being present on the GPU.
