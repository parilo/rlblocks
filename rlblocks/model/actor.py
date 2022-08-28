from typing import Callable, Any, Optional

import torch as t
from torch import nn

from rlblocks.model.model_wrapper import ModelWrapper


class Actor(ModelWrapper):

    def __init__(
            self,
            model: nn.Module,
    ):
        self._model = model

    def __call__(self, state):
        action = self._model(state)
        return action
