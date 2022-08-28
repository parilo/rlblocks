from typing import Callable, Any, Optional

import torch as t
from torch import nn
from torch.distributions.normal import Normal

from rlblocks.model.model_wrapper import ModelWrapper


def get_mu_logstd(params: t.Tensor):
    mu_size = params.shape[-1] // 2
    mu = params[:mu_size]
    logstd = params[mu_size:]
    return mu, logstd


class GaussianActor(ModelWrapper):

    def __init__(
            self,
            model: nn.Module,
            action_min: float,
            action_max: float,
            logstd_scale: Optional[float] = None,
    ):
        self._model = model

    def __call__(self, state: t.Tensor, deterministic: bool = False):
        params = self._model(state)
        mu, logstd = get_mu_logstd(params)
        if deterministic:
            return mu

        dist = Normal(mu, t.exp(logstd))
        return dist.rsample()

    def log_prob(self, state: t.Tensor, action: t.Tensor):
        params = self._model(state)
        mu, logstd = get_mu_logstd(params)
        dist = Normal(mu, t.exp(logstd))
        return dist.log_prob(action)
