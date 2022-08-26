import torch as t


class Normalizer:

    def __init__(self, state_len, device):
        self._state_len = state_len
        self._mean = t.zeros((state_len,)).float().to(device)
        self._std = t.ones((state_len,)).float().to(device)
        self._beta = 0.01

    def norm(self, state: t.Tensor):
        assert (
                state.shape[-1] == self._mean.shape[-1] and
                state.shape[-1] == self._std.shape[-1]
        ), f'Last shape of normalization parameters {self._mean.shape} {self._std.shape} should match state last shape {state.shape}'
        return state / self._std - self._mean

    def denorm(self, state: t.Tensor):
        return state * self._std + self._mean

    def update(self, state: t.Tensor):
        self._mean = (1 - self._beta) * self._mean + self._beta * t.mean(state, dim=[0, 1])
        self._std  = (1 - self._beta) * self._std  + self._beta * t.std(state, dim=[0, 1])
