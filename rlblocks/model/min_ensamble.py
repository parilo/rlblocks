from typing import List

import torch as t
import torch.nn as nn


class MinEnsamble(nn.Module):

    def __init__(self, modules: List[nn.Module]):
        super().__init__()
        self._mds = nn.ModuleList(modules)

    def forward(self, inp):
        return t.min(
            t.cat(
                [module(inp) for module in self._mds],
                dim=-1
            ),
            dim=-1
        )[0].unsqueeze(-1)
