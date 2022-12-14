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


class NormalRandomizedExploration:

    def __init__(self, action_len, min_std, max_std):
        self._action_len = action_len
        self._min_std = min_std
        self._max_std = max_std
        self.reset()

    def get_noise(self):
        return np.random.normal(0., self._std, size=self._action_len)

    def __call__(self, x):
        return x + self.get_noise()

    def reset(self):
        self._std = np.random.uniform(self._min_std, self._max_std, 1)[0]


class NormalStepBasedExploration:

    def __init__(self, action_len, std, switch_steps):
        self._action_len = action_len
        self._std = std
        self._step_ind = 0
        self._switch_steps = switch_steps
        self._do_expl = False
        self.reset()

    def get_noise(self):
        self._step_ind += 1
        if self._step_ind > self._switch_steps:
            self._do_expl = not self._do_expl
            self._step_ind = 0

        if self._do_expl:
            return np.random.normal(0., self._std, size=self._action_len)
        else:
            return np.zeros(shape=(self._action_len,))

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


class RandomActionExploration:

    def __init__(self, action_len, action_min, action_max, gap_steps, action_steps):
        self._action_len = action_len
        self._action_min = action_min
        self._action_max = action_max
        self._gap_steps = gap_steps
        self._action_steps = action_steps
        self._total_steps = gap_steps + action_steps
        self._step_ind = 0
        self._sample_action()

    def _sample_action(self):
        self._action = np.random.uniform(self._action_min, self._action_max, size=self._action_len)

    def _update(self):

        self._step_ind += 1

        if self._step_ind > self._total_steps:
            self._step_ind = 0
            self._sample_action()

    def __call__(self, x):
        self._update()
        if self._step_ind > self._gap_steps:
            # print(f'--- step {self._step_ind} / {self._gap_steps} {self._action}')
            return self._action
        else:
            # print(f'--- step {self._step_ind} / {self._gap_steps}')
            return x

    def reset(self):
        pass
