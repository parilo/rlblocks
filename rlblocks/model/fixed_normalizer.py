import torch as t


class FixedNormalizer:

    def __init__(self, mean, std, epsilon=1e-4):
        self._mean = mean
        self._std = std + epsilon

    def _check(self, tensor: t.Tensor):
        if tensor.shape[-1] != self._mean.shape[-1]:
            raise RuntimeError(
                f'Last shape of mean {self._mean.shape} should match tensor last shape {tensor.shape}')
        if tensor.shape[-1] != self._std.shape[-1]:
            raise RuntimeError(
                f'Last shape of std {self._std.shape} should match tensor last shape {tensor.shape}')

    def norm(self, tensor: t.Tensor):
        self._check(tensor)
        return (tensor - self._mean) / self._std

    def denorm(self, tensor: t.Tensor):
        self._check(tensor)
        return tensor * self._std + self._mean
