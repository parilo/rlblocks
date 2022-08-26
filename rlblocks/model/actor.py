from typing import Callable, Any, Optional

import torch as t
from torch import nn

from lm_rl.model.model_wrapper import ModelWrapper


class Actor(ModelWrapper):

    def __init__(
            self,
            model: nn.Module,
            state_preproc: Optional[Callable[[Any], t.Tensor]] = None,
    ):
        self._model = model
        self._state_preproc = state_preproc

    def __call__(self, state):
        if self._state_preproc:
            state = self._state_preproc(state)
        action = self._model(state)
        return action
