import numpy as np


class BaseInitializer:
    def initialize(self, x):
        raise NotImplementedError


class RandomNormal(BaseInitializer):
    def __init__(self, mean=0.0, std=1.0):
        self._mean = mean
        self._std = std

    def initialize(self, x):
        x[:] = np.random.normal(loc=self._mean, scale=self._std, size=x.shape)


class RandomUniform(BaseInitializer):
    def __init__(self, low=0.0, high=1.0):
        self._low = low
        self._high = high

    def initialize(self, x):
        x[:] = np.random.uniform(self._low, self._high, size=x.shape)


class Zeros(BaseInitializer):
    def initialize(self, x):
        x[:] = np.zeros_like(x)


class Ones(BaseInitializer):
    def initialize(self, x):
        x[:] = np.ones_like(x)


class TruncatedNormal(BaseInitializer):
    def __init__(self, mean=0.0, std=1.0):
        self._mean = mean
        self._std = std

    def initialize(self, x):
        x[:] = np.random.normal(loc=self._mean, scale=self._std, size=x.shape)
        truncated = 2 * self._std + self._mean
        x[:] = np.clip(x, -truncated, truncated)


class Constant(BaseInitializer):
    def __init__(self, v):
        self._v = v

    def initialize(self, x):
        x[:] = np.full_like(x, self._v)


def get_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[-1]
    return fan_in, fan_out


class GlorotUniform(BaseInitializer):
    def __init__(self):
        pass

    def initialize(self, x):
        fan_in, fan_out = get_fans(x.shape)
        s = np.sqrt(6.0 / (fan_in + fan_out))
        x[:] = np.random.uniform(-s, s, x.shape)


random_normal = RandomNormal()
random_uniform = RandomUniform()
zeros = Zeros()
ones = Ones()
truncated_normal = TruncatedNormal()
glorot_uniform = GlorotUniform()
