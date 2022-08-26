import numpy as np


class NormalExploration:

    def __init__(self, action_len, std):
        self._action_len = action_len
        self._std = std

    def get_noise(self):
        return np.random.normal(0., self._std, size=self._action_len)

    def __call__(self, x):
        return x + self.get_noise()

    def reset(self):
        pass


class CorrelatedExploration:

    def __init__(self, action_len, std, beta):
        self._action_len = action_len
        self._std = std
        self._beta = beta
        self._prev_noise = None

    def get_noise(self):
        noise = np.random.normal(0., self._std, size=self._action_len)
        if self._prev_noise is not None:
            noise = (1 - self._beta) * noise + self._beta * self._prev_noise
        self._prev_noise = noise
        return noise

    def __call__(self, x):
        return x + self.get_noise()

    def reset(self):
        self._prev_noise = None
