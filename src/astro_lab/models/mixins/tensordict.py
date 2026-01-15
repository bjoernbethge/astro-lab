"""
TensorDictMixin for AstroLab Models
==================================

Provides native TensorDict support for models.

"""

from typing import List, Optional, Union

import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch_geometric.data import Batch, Data


class TensorDictMixin:
    """Mixin for TensorDict support in models."""

    def to_tensordict_module(
        self, in_keys: Optional[List[str]] = None, out_keys: Optional[List[str]] = None
    ) -> TensorDictModule:
        if in_keys is None:
            in_keys = ["x", "edge_index", "batch"]
        if out_keys is None:
            out_keys = ["logits"]

        class TensorDictForward(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                if len(args) == 1 and isinstance(args[0], TensorDict):
                    td = args[0]
                    batch = Data(
                        x=td.get("x"),
                        edge_index=td.get("edge_index"),
                        edge_attr=td.get("edge_attr", None),
                        batch=td.get("batch", None),
                        y=td.get("y", None),
                    )
                    output = self.model(batch)
                    result_td = TensorDict(
                        {
                            out_keys[0]: output,
                        },
                        batch_size=td.batch_size,
                    )
                    if hasattr(self.model, "get_embeddings") and len(out_keys) > 1:
                        embeddings = self.model.get_embeddings(batch)
                        result_td[out_keys[1]] = embeddings
                    return result_td
                else:
                    return self.model(*args, **kwargs)

        wrapper = TensorDictForward(self)
        return TensorDictModule(
            module=wrapper,
            in_keys=in_keys,
            out_keys=out_keys,
        )

    def consolidate_batch(
        self, batch: Union[Data, Batch, TensorDict]
    ) -> Union[Data, Batch, TensorDict]:
        """Consolidate batch data for faster device transfer."""
        return batch  # Placeholder for actual implementation
