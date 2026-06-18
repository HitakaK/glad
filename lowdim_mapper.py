# lowdim_mapper.py

import torch
import torch.nn as nn


class LowDimMapper(nn.Module):
    def __init__(self, u_dim=4, num_ws=12, w_dim=512, hidden=256):
        super().__init__()
        self.u_dim = u_dim
        self.num_ws = num_ws
        self.w_dim = w_dim
        self.hidden = hidden

        self.net = nn.Sequential(
            nn.Linear(u_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_ws * w_dim),
        )

    def forward(self, u):
        w = self.net(u)
        return w.view(u.shape[0], self.num_ws, self.w_dim)
