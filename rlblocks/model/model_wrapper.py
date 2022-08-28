from torch import nn


class ModelWrapper:

    @property
    def model(self) -> nn.Module:
        return self._model
