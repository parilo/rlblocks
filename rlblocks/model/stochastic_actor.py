import torch as t


class StochasticActor:

    def __call__(self, state: t.Tensor, deterministic: bool = False) -> t.Tensor:
        raise NotImplemented()

    def log_prob(self, state: t.Tensor, action: t.Tensor) -> t.Tensor:
        raise NotImplemented()


class DeterministicActorWrapper:

    def __init__(self, actor: StochasticActor):
        self._actor = actor

    def __call__(self, state: t.Tensor) -> t.Tensor:
        return self._actor(state, deterministic=True)
