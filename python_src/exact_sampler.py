import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from basis_numpy import Basis1DZ2

from tree_utils import to_list


def generate_full_confs(basis):
    N = basis.N
    bvecs = [basis.basis_vectors(i) for i in range(basis.get_dim())]
    degen = [len(v) for v in bvecs]
    confs = jnp.array([basis.to_bin_array(N, v[0]) for v in bvecs])
    confs = 1-2*confs
    return confs, degen


class ExactSampler:
    _basis: Basis1DZ2
    _model: nn.Module
    _model_output: jnp.array

    def __init__(self, basis: Basis1DZ2, model: nn.Module):
        self._basis = basis
        self.N = basis.N
        self._model = model
        self._symm_confs, basis_degen = generate_full_confs(basis)
        self._basis_degen = jnp.array(basis_degen)
        self._data_symm_probs = jnp.power(basis.coeffs_tensor(), 2)
        self._entropy = None
        self._model_output = None

    def update_model_output(self, params):
        self._model_output = jnp.ravel(jax.jit(self._model.apply)(params, self._symm_confs))

    @property
    def basis_degen(self):
        return self._basis_degen

    @property
    def partition_function(self):
        return self._partition_function
    
    @property
    def entropy_data(self):
        if self._entropy == None:
            log_probs = jnp.log(jnp.clip(self._data_symm_probs, a_min = 1e-7))
            log_probs = log_probs - jnp.log(self._basis_degen)
            self._entropy = -jnp.dot(self._data_symm_probs, log_probs)
        return self._entropy

    def cross_entropy(self, params):
        norm = jnp.dot(jnp.exp(self._model_output), self._basis_degen)
        return -jnp.dot(self._data_full_probs, log_model_probs) + jnp.log(norm)

    def overlap(self, params):
        model_probs = jnp.exp(jnp.ravel(self._model_output)) * self._basis_degen
        return jnp.dot(jnp.sqrt(model_probs), jnp.abs(self._basis.coeffs_tensor())) / jnp.sqrt(jnp.sum(model_probs))

    def sample(self, key, params, batch_size):
        model_symm_log_probs = self._model_output.squeeze() + jnp.log(self._basis_degen)
        self._partition_function = jnp.sum(jnp.exp(model_symm_log_probs))
        key, key_r = jax.random.split(key)
        indices = jax.random.categorical(key_r, model_symm_log_probs, shape=(batch_size,))

        confs = []
        probs = []

        for idx in indices:
            bvecs = self._basis.basis_vectors(idx)
            key, key_r = jax.random.split(key)
            random_v = jax.random.randint(key_r, (1,), 0, len(bvecs))
            confs.append(self._basis.to_bin_array(self.N, bvecs[random_v.item()]))
            probs.append(self._model_output[idx])
        
        confs = jnp.vstack(confs)
        probs = jnp.exp(jnp.array(probs))/self._partition_function
        return probs, confs



