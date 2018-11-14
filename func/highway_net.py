import torch.nn as nn
import torch.nn.functional as F


class Highway_Net(nn.Module):
    def __init__(self,args):
        super(Highway_Net, self).__init__()
        self.n_layers=2
        self.in_size=args.embed_size*2
        self.normal_layer = nn.ModuleList([nn.Linear(self.in_size,self.in_size) for _ in range(self.n_layers)])
        self.gate_layer = nn.ModuleList([nn.Linear(self.in_size,self.in_size) for _ in range(self.n_layers)])

    def forward(self, x):
        for i in range(self.n_layers):
            normal_layer_ret = F.relu(self.normal_layer[i](x))
            gate = F.sigmoid(self.gate_layer[i](x))
            x = gate * normal_layer_ret + (1 - gate) * x
        return x
