import torch
from torch import nn
import torch.nn.functional as F

class TypewiseInputProjector(nn.Module):
    def __init__(self, input_dims: dict, hidden_dim: int, embed_dims: dict = None, unknown_token: str = "UNKNOWN"):
        super().__init__()
        self.embed_layers = nn.ModuleDict()
        self.lin_layers = nn.ModuleDict()
        for node_type, in_dim in input_dims.items():
            if node_type == 'encounter':
                # direct linear for encounter features
                self.lin_layers[node_type] = nn.Linear(in_dim, hidden_dim)
            else:
                # embedding for other types
                emb_dim = embed_dims.get(node_type, hidden_dim) if embed_dims else hidden_dim
                self.embed_layers[node_type] = nn.Embedding(in_dim, emb_dim)
                if emb_dim != hidden_dim:
                    self.lin_layers[node_type] = nn.Linear(emb_dim, hidden_dim)
        self.activation = nn.ReLU()
    def forward(self, x_dict):
        out_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.embed_layers:
                # x may be indices for embedding
                x_emb = self.embed_layers[node_type](x.squeeze(-1).long() if x.dim()>1 else x.long())
                if node_type in self.lin_layers:
                    out = self.lin_layers[node_type](x_emb)
                else:
                    out = x_emb
            else:
                out = self.lin_layers[node_type](x)
            out_dict[node_type] = self.activation(out)
        return out_dict

class EncounterClassifier(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)
    def forward(self, x):
        return self.lin(x)
