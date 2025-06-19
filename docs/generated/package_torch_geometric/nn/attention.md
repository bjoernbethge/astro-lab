# attention

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.attention`

## Classes (1)

### `PerformerAttention`

The linear scaled attention mechanism from the
`"Rethinking Attention with Performers"
<https://arxiv.org/abs/2009.14794>`_ paper.

Args:
    channels (int): Size of each input sample.
    heads (int, optional): Number of parallel attention heads.
    head_channels (int, optional): Size of each attention head.
        (default: :obj:`64.`)
    kernel (Callable, optional): Kernels for generalized attention.
        If not specified, `ReLU` kernel will be used.
        (default: :obj:`torch.nn.ReLU()`)
    qkv_bias (bool, optional): If specified, add bias to query, key
        and value in the self attention. (default: :obj:`False`)
    attn_out_bias (bool, optional): If specified, add bias to the
        attention output. (default: :obj:`True`)
    dropout (float, optional): Dropout probability of the final
        attention output. (default: :obj:`0.0`)

#### Methods

- **`forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Forward pass.

- **`redraw_projection_matrix(self)`**
  As described in the paper, periodically redraw
