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
        layers.append(nn.Linear(self._layer_size, self._output_size))

        self.add_module('_model', nn.Sequential(*layers))

    def forward(self, x):
        return self._model(x)
