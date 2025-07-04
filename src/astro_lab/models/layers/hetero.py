from torch import nn
from torch_geometric.nn import GCNConv, HeteroConv


class HeteroGNNLayer(nn.Module):
    """
    A flexible heterogeneous GNN layer using PyG's HeteroConv.
    Supports arbitrary node and edge types as defined in the metadata.
    """

    def __init__(self, metadata, hidden_channels, conv_class=GCNConv, aggr="sum"):
        super().__init__()
        # metadata: (node_types, edge_types)
        self.conv = HeteroConv(
            {edge_type: conv_class(-1, hidden_channels) for edge_type in metadata[1]},
            aggr=aggr,
        )

    def forward(self, x_dict, edge_index_dict):
        return self.conv(x_dict, edge_index_dict)
