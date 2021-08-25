import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import jit
import math
from functools import partial
from typing import Callable


def leaky_hard_tanh(x, slope):
    return jnp.where(x > 1, slope*(x-1) + 1, jnp.where(x < -1, slope*(x+1)-1, x))

class ConvNet1D(nn.Module):
    features: int
    kernel_size: int
    dtype: jnp.dtype
    initializer: str = None
    first_activation: Callable[[jnp.array], jnp.array] = None
    second_activation: Callable[[jnp.array], jnp.array] = None

    def setup(self):
        if self.features % 2 != 0:
            raise ValueError("Channels must be even, but {} is given.".format(self.features))    

        if self.kernel_size % 2 != 1:
            raise ValueError("Kernel_size must be odd, but {} is given.".format(self.kernel_size))

        if self.initializer is None:
            kernel_init = jax.nn.initializers.lecun_normal()
        else:
            kernel_init = getattr(jax.nn.initializers, self.initializer)()

        if self.first_activation is None:
            self._first_activation = lambda x: jnp.cos(math.pi*x)
        else:
            self._first_activation = self.first_activation

        if self.second_activation is None:
            self._second_activation = lambda x: leaky_hard_tanh(x, 0.01)
        else:
            self._first_activation = self.first_activation

        self.conv1 = nn.Conv(features=self.features//2, kernel_size=self.kernel_size,
                padding='VALID', use_bias=False, dtype=self.dtype, kernel_init = kernel_init)
        self.conv2 = nn.Conv(features=self.features, kernel_size=self.kernel_size,
                padding='VALID', use_bias=False, dtype=self.dtype, kernel_init = kernel_init)
        self.fc = nn.Dense(1, use_bias=False, dtype=self.dtype, kernel_init = kernel_init)

    def __call__(self, inputs):
        out = jnp.expand_dims(inputs, -1) #shape (n_batch, N, 1)
        out = jnp.pad(out, ((0,0), (self.kernel_size//2, self.kernel_size//2), (0,0)), 'wrap')
        out = self.conv1(out)
        out = self._first_activation(out)
        out = jnp.pad(out, ((0,0), (self.kernel_size//2, self.kernel_size//2), (0,0)), 'wrap')
        out = self.conv2(out)
        out = self._second_activation(out)
        out = jnp.mean(out, axis=(-2))
        return self.fc(out)


