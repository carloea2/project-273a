import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
from src.models.heads import TypewiseInputProjector, EncounterClassifier

class HGTModel(nn.Module):
    def __init__(self, metadata, config):
        super().__init__()
        self.hidden_dim = config.model.hidden_dim
        # Node type input projector
        # Determine input dims for each node type
        input_dims = {}
        embed_dims = {}
        # We'll initialize the projector lazily, since input dims will come from data
        self.proj = None
        # HGT conv layers
        self.convs = nn.ModuleList()
        for _ in range(config.model.num_layers):
            conv = HGTConv(self.hidden_dim, self.hidden_dim, metadata,
                           heads=config.model.heads, group='sum')
            self.convs.append(conv)
        self.classifier = EncounterClassifier(self.hidden_dim)
    def forward(self, x_dict, edge_index_dict):
        # Lazy initialize projector using the current batch's feature sizes
        if self.proj is None:
            input_dims = {}
            embed_dims = {}
            for node_type, x in x_dict.items():
                if node_type == 'encounter':
                    input_dims[node_type] = x.shape[1]
                else:
                    input_dims[node_type] = int(x.max().item()) + 1 if x.numel()>0 else 0
                    embed_dims[node_type] = self.hidden_dim
            self.proj = TypewiseInputProjector(input_dims, self.hidden_dim, embed_dims)
        # Project input features to hidden dim
        x_dict_proj = self.proj(x_dict)
        # Perform HGT convolution layers
        for conv in self.convs:
            x_dict_proj = conv(x_dict_proj, edge_index_dict)
            # apply activation after each conv
            x_dict_proj = {k: F.relu(v) for k,v in x_dict_proj.items()}
        # Classifier on encounter node embeddings
        enc_out = x_dict_proj['encounter']
        logit = self.classifier(enc_out).view(-1)
        return logit
