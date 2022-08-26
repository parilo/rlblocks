import torch as t

from lm_rl.model.model_wrapper import ModelWrapper


class QFunc(ModelWrapper):

    def __init__(self, model):
        self._model = model

    def __call__(self, state: t.Tensor, action: t.Tensor) -> t.Tensor:
        return self._model(t.cat([state, action], dim=-1))
