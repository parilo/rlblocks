from typing import Optional

import torch as t

from rlblocks.model.fixed_normalizer import FixedNormalizer
from rlblocks.model.model_wrapper import ModelWrapper


class QFunc(ModelWrapper):

    def __init__(
            self,
            model,
            state_norm: Optional[FixedNormalizer] = None,
    ):
        self._model = model
        self.state_norm = state_norm

    def __call__(self, state: t.Tensor, action: t.Tensor) -> t.Tensor:
        state = self.state_norm.norm(state)
        return self._model(t.cat([state, action], dim=-1))
