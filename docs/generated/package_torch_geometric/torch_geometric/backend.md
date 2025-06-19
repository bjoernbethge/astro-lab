# backend

Part of `torch_geometric.torch_geometric`
Module: `torch_geometric.backend`

## Functions (1)

### `use_segment_matmul_heuristic(num_segments: int, max_segment_size: int, in_channels: int, out_channels: int) -> bool`

A heuristic based on input sizes to determine whether the usage of
:meth:`segment_matmul` can speed up computation.
