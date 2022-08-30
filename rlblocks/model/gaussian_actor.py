from typing import Callable, Any, Optional, Tuple

import torch as t
from torch import nn
from torch.distributions.normal import Normal

from rlblocks.model.model_wrapper import ModelWrapper
from rlblocks.model.stochastic_actor import StochasticActor


def get_mu_logstd(params: t.Tensor, logstd_range: Optional[float] = None):
    mu_size = params.shape[-1] // 2
    mu = params[..., :mu_size]
    logstd = params[..., mu_size:]
    if logstd_range:
        scale = (logstd_range[0] + logstd_range[1]) / 2
        logstd = scale * t.tanh(logstd) + scale - logstd_range[0]
    return mu, logstd


class GaussianActor(ModelWrapper, StochasticActor):

    def __init__(
            self,
            model: nn.Module,
            action_min: float,
            action_max: float,
            logstd_range: Optional[Tuple[float, float]] = None,
    ):
        self._model = model
        self._action_min = action_min
        self._action_max = action_max
        self._logstd_range = logstd_range

    def __call__(self, state: t.Tensor, deterministic: bool = False) -> t.Tensor:
        params = self._model(state)
        mu, logstd = get_mu_logstd(params, self._logstd_range)
        if deterministic:
            action = mu
        else:
            dist = Normal(mu, t.exp(logstd))
            action = dist.rsample()
        return action.clamp(self._action_min, self._action_max)

    def log_prob(self, state: t.Tensor, action: t.Tensor) -> t.Tensor:
        params = self._model(state)
        mu, logstd = get_mu_logstd(params, self._logstd_range)
        dist = Normal(mu, t.exp(logstd))
        return dist.log_prob(action)
