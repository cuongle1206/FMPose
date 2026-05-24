import torch
import torch.nn as nn
import sys, os
parent_dir = os.path.dirname("./")
sys.path.append(parent_dir)
import time
import config as c  
from models.gcn import GCNet
from flow_matching.utils import ModelWrapper

# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return torch.sigmoid(x) * x

# Wrapped model for ODE
class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)

class ResidualBlock(nn.Module):
    def __init__(self, dim: int=1024, use_ln: bool=False, act_fn: str='swish'):
        super(ResidualBlock, self).__init__()
        self.use_ln     = use_ln
        self.l1         = nn.Linear(dim, dim)
        self.ln1        = nn.LayerNorm(dim)
        self.l2         = nn.Linear(dim, dim)
        self.ln2        = nn.LayerNorm(dim)
        act_dict = {'relu': nn.ReLU(), 'lrelu': nn.LeakyReLU(), 'sigmoid': nn.Sigmoid(),
                    'tanh': nn.Tanh(), 'swish': Swish()}
        if act_fn not in act_dict: raise ValueError(f"Unsupported activation function: {act_fn}")
        self.act = act_dict[act_fn]
    def forward(self, x):
        inp             = x
        x               = self.act(self.l1(x))
        if self.use_ln: x = self.ln1(x)
        x               = self.act(self.l2(x))
        if self.use_ln: x = self.ln2(x)
        x               += inp
        return x
        
class FlowNet(nn.Module):
    def __init__(self, hid_dim: int, act_fn: str, k: int, n_layers: int):
        super(FlowNet, self).__init__()
        
        cond_in, cond_hid, cond_out = 2*k, 128, 64
        self.condnet    = GCNet(cond_in, cond_hid, cond_out, n_layers)
        
        in_dim          = c.N_3DJOINTS*3 + 1 + cond_out
        out_dim         = c.N_3DJOINTS*3
        self.embed      = nn.Linear(in_dim, hid_dim)
        self.res1       = ResidualBlock(hid_dim, True, act_fn)
        self.res2       = ResidualBlock(hid_dim, True, act_fn)
        self.outlayer   = nn.Linear(hid_dim, out_dim)
        
    def forward(self, x, t, p2d):
        N, J, C         = x.shape
        cond_inp        = p2d.reshape(N,c.N_2DJOINTS,-1) # N x 16 x 2k
        condition       = self.condnet(cond_inp) # N x 2048
        t               = t.requires_grad_(True)
        net_input       = torch.cat((x.reshape(N,-1),
                                     t.unsqueeze(-1).reshape(-1, 1).expand(N, 1),
                                     condition), dim=-1)
        em              = self.embed(net_input)
        em              = self.res1(em)
        em              = self.res2(em)
        dxdt            = self.outlayer(em)
        return dxdt.reshape(N,J,C)
    
    def save(self, path):
        # do not save unnecessary tmp variables..
        filtered_dict = {k: v for k, v in self.state_dict().items() if 'tmp_var' not in k}
        torch.save({'net': filtered_dict}, path)
        print("weights saved at: ", path)

    def load(self, path, device):
        state_dicts = torch.load(path, map_location=device)
        network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
        self.load_state_dict(network_state_dict)
        print("weights of trained model loaded")
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)

if __name__ == "__main__":
    
    model  = FlowNet(hid_dim=1024, act_fn=c.args.act, k=c.args.topk, n_layers=c.args.num_gcn).to(c.device)
    model.eval()
    
    # move model and input to GPU if available:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ensure model is in evaluation mode:
    X = torch.rand(1, 17, 3) # some random input, B=1
    t = torch.rand(1).to(c.device)
    p2d = torch.rand(1, 16, 2, 48).to(c.device)
    
    model = model.to(device)
    X = X.to(device)
    output = model(X, t, p2d)
    
    n = 10000 # feel free to reduce/increase
    with torch.no_grad():
        # start timer:
        start_time = time.time()
        for _ in range(n):
            # perform forward pass:
            # output = model(X)
            output = model(X, t, p2d)
        # end timer:
        end_time = time.time()
    print("Time for forward pass: {} seconds".format( (end_time - start_time) / n))