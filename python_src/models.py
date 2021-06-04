import flax.linen as nn
import jax.numpy as jnp
import math

def leaky_hard_tanh(x, slope):
    return jnp.where(x > 1, slope*(x-1) + 1, jnp.where(x < -1, slope*(x+1)-1, x))

class ConvNet1D(nn.Module):    
    features: int    
    kernel_size: int    

    def setup(self):    
        if self.features % 2 != 0:    
            raise ValueError("Channels must be even, but {} is given.".format(self.features))    

        if self.kernel_size % 2 != 1:    
            raise ValueError("Kernel_size must be odd, but {} is given.".format(self.kernel_size))    

        self.conv1 = nn.Conv(features=self.features//2, kernel_size=self.kernel_size, padding='VALID', use_bias=False)
        self.conv2 = nn.Conv(features=self.features, kernel_size=self.kernel_size, padding='VALID', use_bias=False)
        self.fc = nn.Dense(1, use_bias=False)    

    def __call__(self, inputs):
        out = jnp.expand_dims(inputs, -1) #shape (n_batch, N, 1)
        out = jnp.pad(out, ((0,0), (self.kernel_size//2, self.kernel_size//2), (0,0)), 'wrap')
        out = self.conv1(out)
        out = jnp.cos(math.pi*out)
        out = jnp.pad(out, ((0,0), (self.kernel_size//2, self.kernel_size//2), (0,0)), 'wrap')
        out = self.conv2(out)
        out = jnp.mean(out, axis=(-2))
        out = leaky_hard_tanh(out, 0.1)    
        return self.fc(out)


