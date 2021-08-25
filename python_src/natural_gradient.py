import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import flax.linen as nn
from tree_utils import to_list, reshape_tree_like
from functools import partial
#from jax.interpreters import xla
import sys

class NGD:
    _model: nn.Module
    _fisher: jnp.array
    _grad: jnp.array

    learning_rate: float
    beta1: float
    beta2: float
    t: int

    def __init__(self, model: nn.Module, learning_rate: float, *,
            beta1: float = 0.9, beta2: float = 0.999):
        self._model = model
        self._fisher = None
        self._grad = None #gradient of the KL divergence

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0

    def update_momentums(self, params, confs_data, confs_model):
        new_fisher, grad_model = self._calc_fisher(params, confs_model)

        if self._fisher is None:
            self._fisher = new_fisher
        else:
            self._fisher = self.beta2*self._fisher + (1-self.beta2)*new_fisher

        output_mean, grad_data = self._calc_grad(params, confs_data)
        self.cross_entropy_unnormalized = -output_mean

        new_grad = grad_model - to_list(grad_data)

        if self._grad is None:
            self._grad = new_grad
        else:
            self._grad = self.beta1*self._grad + (1-self.beta1)*new_grad

    @partial(jax.jit, static_argnums = 0)
    def _calc_fisher(self, params, confs):
        batch_size = confs.shape[0]
        grads = jacrev(self._model.apply)(params, confs)
        grads = jnp.hstack([g.reshape(batch_size, -1) for g in jax.tree_leaves(grads)])

        grad_mean = jnp.mean(grads, axis = 0)
        grads = grads - grad_mean[None,:]

        new_fisher = jnp.matmul(jnp.transpose(grads), grads) / batch_size
        return new_fisher, grad_mean

    @partial(jax.jit, static_argnums = 0)
    def _calc_grad(self, params, confs):
        f_data = lambda p: jnp.mean(self._model.apply(p, confs))
        return jax.value_and_grad(f_data)(params)

    def update_sgd(self, params):
        b = reshape_tree_like(self._grad, params)
        return jax.tree_multimap(lambda x, y: x - self.learning_rate*y, params, b)

    def update(self, params):
        #eps = max(1.0*(0.9**self.t), 1e-3)
        self.t += 1
        eps = 1e-3
        b = jnp.linalg.solve(self._fisher + eps*jnp.identity(self._fisher.shape[0]), self._grad)\
                * ((1. - jnp.power(self.beta2, self.t))/(1. - jnp.power(self.beta1, self.t)))
        b = reshape_tree_like(b, params)
        return jax.tree_multimap(lambda x, y: x - self.learning_rate*y, params, b)

    def state_dict(self):
        return {'fisher': self._fisher, 'grad': self._grad, 't': self.t}

    def load_state_dict(self, state_dict):
        self._fisher = state_dict['fisher']
        self._grad = state_dict['grad']
        self.t = state_dict['t']


