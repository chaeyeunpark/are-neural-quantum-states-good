import math
import sys
import argparse
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from flax import serialization
from flax import optim

from models import ConvNet1D, leaky_hard_tanh
from exact_sampler import generate_full_confs
from natural_gradient import NGD
from utils import get_lattice_1d, get_J2, dataset_total_and_save
from basis_numpy import Basis1DZ2
import json
import pickle

import jax.profiler
from functools import partial

class OverlapCalculator:
    _symm_confs: jnp.array
    _data_symm_probs: jnp.array
    _data_symm_signs: jnp.array
    _basis_degen: jnp.array

    def __init__(self, basis):
        self._symm_confs, basis_degen = generate_full_confs(basis)
        self._basis_degen = jnp.array(basis_degen)
        
        data_coeffs = basis.coeffs_tensor()
        self._data_symm_probs = jnp.power(data_coeffs, 2)
        self._data_symm_signs = jnp.sign(data_coeffs)
    
    @partial(jax.jit, static_argnums=(0,1))
    def overlap(self, model, params):
        model_output = model.apply(params, self._symm_confs)
        signs = jnp.sign(jnp.ravel(model_output)) * self._data_symm_signs
        return jnp.dot(signs, self._data_symm_probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn sign structure of 1d model')
    parser.add_argument('--data-path', type=str, required=True, dest='data_path')
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--minbatch-size', type=int, required=True, dest='minbatch_size')
    parser.add_argument('--learning-rate', type=float, required=True, dest='learning_rate')

    args = parser.parse_args()
    args = vars(args)

    N = get_lattice_1d(args['data_path'])
    if 'J1J2' in args['data_path']:
        basis = Basis1DZ2(N, True)
    elif 'TXYZ' in args['data_path']:
        basis = Basis1DZ2(N, False)
    basis.load_data(args['data_path'])
    J2 = get_J2(args['data_path'])

    model_dtype = jnp.float32
    initializer = 'glorot_normal'

    #Set hyperparameters
    DIM = basis.get_dim()
    learning_rate = float(args['learning_rate'])
    width = int(args['width'])
    MINBATCH_SIZE = int(args['minbatch_size'])
    kernel_size = N//2 + 1

    TOTAL_DATASET, SAVE_DATASET = dataset_total_and_save(N, DIM)
    TOTAL_EPOCHS = int((250/150)*TOTAL_DATASET//MINBATCH_SIZE)

    save_overlap_per = SAVE_DATASET//MINBATCH_SIZE

    model = ConvNet1D(features = width, kernel_size = kernel_size, dtype = model_dtype, 
            initializer = initializer)

    key = random.PRNGKey(1337)
    key, key_init, key_r = random.split(key, 3)
    params = model.init(key_init, 1-2*random.randint(key_r, (MINBATCH_SIZE, N), 0, 2)) # Initialization call
    total_params = sum([p.size for p in jax.tree_leaves(params)])

    param_out = args.copy()
    param_out.update({
        'N': N,
        'J2': J2,
        'total_parameters': total_params,
        'initializer': initializer
    })
    json.dump(param_out, Path('param_out.json').open('w'))

    print(f"#mini batch: {MINBATCH_SIZE}, total epochs: {TOTAL_EPOCHS}, dim: {DIM}")

    optimizer_def = optim.Adam(learning_rate = learning_rate)
    optimizer = optimizer_def.create(params)

    overlap_calculator = OverlapCalculator(basis)
    
    @jax.jit
    def loss_func(p, confs, labels):
        """
        Implemented following BCEWithLogitsLoss() in
        https://github.com/pytorch/pytorch/blob/ac67cda272b0856c1be3ebfebc324e0b3bab286f/aten/src/ATen/native/Loss.cpp#L224
        """
        model_output = model.apply(p, confs).flatten()
        max_val = jnp.clip(-model_output, a_min=0.0)
        t = (1-labels)*model_output + max_val + \
                jnp.log((jnp.exp(-max_val) + jnp.exp(-model_output - max_val)))
        return jnp.mean(t)

    with open("overlap.dat", 'w') as overlap_out:
        for epoch in range(TOTAL_EPOCHS):
            ys, xs = basis.generate_batch(MINBATCH_SIZE)
            xs = 1-2*xs

            loss, grad = jax.value_and_grad(loss_func)(optimizer.target, xs, ys)

            print(f'epoch: {epoch}, loss: {loss}')

            if epoch % save_overlap_per == 0:
                overlap = overlap_calculator.overlap(model, optimizer.target)
                print(f'{epoch}\t{overlap}', file=overlap_out)
                overlap_out.flush()

            optimizer = optimizer.apply_gradient(grad)

        overlap = overlap_calculator.overlap(model, optimizer.target)
        print(f'{epoch}\t{overlap}', file=overlap_out)

    bytes_output = serialization.to_bytes(optimizer.target)
    with open('model_final.fl', 'wb') as f:
        f.write(bytes_output)

    opt_state_dict = optimizer.state_dict()
    with open('adam_state.pk', 'wb') as f:
        pickle.dump(opt_state_dict, f)
