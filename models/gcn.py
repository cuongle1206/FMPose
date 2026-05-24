import torch
import torch.nn as nn
import config as c  
from utils.graphs import get_adjacency_matrix

# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return torch.sigmoid(x) * x

class GraphConv(nn.Module):
    def __init__(self, dim: int, A: torch.Tensor):
        super(GraphConv, self).__init__()
        self.linear     = nn.Linear(dim, dim)
        self.A          = nn.Parameter(A, requires_grad=True)
        self.act        = Swish()
    
    def forward(self, in_tensor):
        support         = self.linear(in_tensor)
        out_tensor      = self.act(torch.einsum('jk,nkc->njc', self.A, support))
        return out_tensor
    
    def reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
    
class GCNet(nn.Module):
    def __init__(self, in_dim: int=2, hid_dim: int=64, out_dim: int=64, n_layers: int=1):
        super(GCNet, self).__init__()
        self.cond_in    = nn.Linear(in_dim, hid_dim)
        A, A_norm       = get_adjacency_matrix(num_nodes=c.N_2DJOINTS) # normalized A, A = D^{-0.5} @ A @ D^{-0.5}
        A               = torch.from_numpy(A).to(c.device)
        self.dropout_prob = 0.01
        self.pos_enc    = nn.Embedding(c.N_2DJOINTS, hid_dim)
        self.n_layers   = n_layers
        self.gcns       = nn.ModuleList()
        for _ in range(self.n_layers):
            self.gcns.append(GraphConv(hid_dim, torch.zeros_like(A).clone()))
        self.cond_out   = nn.Linear(c.N_2DJOINTS*hid_dim, out_dim)
        
    def forward(self, x):
        N, J, C         = x.shape
        h               = self.cond_in(x)
        if self.dropout_prob > 0. and self.training:
            dropout_mask    = (torch.rand(N, J, 1, device=c.device) > self.dropout_prob).float()
            h               = h * dropout_mask
        h               = h + self.pos_enc(torch.arange(J, device=c.device)).unsqueeze(0)
        for i in range(self.n_layers): h = self.gcns[i](h)
        y               = self.cond_out(h.reshape(N,-1))
        return y
        
        
