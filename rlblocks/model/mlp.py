import torch as t
from torch import nn


class MLP(nn.Module):

    def __init__(self, input_size, layers_num, layer_size, output_size):
        super().__init__()

        self._input_size = input_size
        self._layers_num = layers_num
        self._layer_size = layer_size
        self._output_size = output_size

        layers = [
            nn.Linear(self._input_size, self._layer_size),
            nn.ReLU()
        ]
        for _ in range(self._layers_num):
            layers.extend([
                nn.Linear(self._layer_size, self._layer_size),
                nn.ReLU()
            ])
        ll = nn.Linear(self._layer_size, self._output_size)
        with t.no_grad():
            ll.weight *= 0.01
            ll.bias *= 0.01
        layers.append(ll)

        self.add_module('_model', nn.Sequential(*layers))

    def forward(self, x):
        return self._model(x)
