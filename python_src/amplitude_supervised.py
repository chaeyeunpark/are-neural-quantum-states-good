import math
import sys
import argparse
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from flax import serialization


from models import ConvNet1D, leaky_hard_tanh
from exact_sampler import ExactSampler
from natural_gradient import NGD
from utils import get_lattice_1d, get_J2
from basis_numpy import Basis1DZ2
import json

import jax.profiler

def bin_array_to_int(arr):
    s = 0
    for idx, d in enumerate(arr):
        s += d*(1<<idx)
    return s

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


    #Set hyperparameters
    DIM = basis.get_dim()
    learning_rate = float(args['learning_rate'])
    width = int(args['width'])
    MINBATCH_SIZE = int(args['minbatch_size'])
    kernel_size = N//2 + 1

    if N == 24:
        TOTAL_DATASET = 128*DIM
        save_overlap_per = DIM//MINBATCH_SIZE
    elif N == 28:
        TOTAL_DATASET = 16*DIM
        save_overlap_per = DIM//MINBATCH_SIZE//4
    elif N == 32:
        TOTAL_DATASET = 2*DIM
        save_overlap_per = DIM//MINBATCH_SIZE//16
    
    TOTAL_EPOCHS = TOTAL_DATASET//MINBATCH_SIZE

    model = ConvNet1D(features = width, kernel_size = kernel_size)

    key = random.PRNGKey(1337)
    key_init, key = random.split(key)
    params = model.init(key_init, 1-2*random.randint(key, (MINBATCH_SIZE, N), 0, 2)) # Initialization call
    total_params = sum([p.size for p in jax.tree_leaves(params)])


    param_out = args.copy()
    param_out.update({
        'J2': J2,
        'total_parameters': total_params
    })
    json.dump(param_out, Path('param_out.json').open('w'))

    exact_sampler = ExactSampler(basis, model)

    ngd = NGD(model, learning_rate, beta1 = 0.9, beta2 = 0.99)
    
    print(f"#mini batch: {MINBATCH_SIZE}, total epochs: {TOTAL_EPOCHS}, dim: {DIM}, data entropy: {exact_sampler.entropy_data}")

    
    with open("overlap.dat", 'w') as overlap_out:
        for epoch in range(TOTAL_EPOCHS):
            exact_sampler.update_model_output(params)

            coeffs_data, confs_data = basis.generate_batch_coeffs(MINBATCH_SIZE)
            key_sampler, key = random.split(key)
            probs_model, confs_model = exact_sampler.sample(key_sampler, params, MINBATCH_SIZE)
            confs_data = 1-2*confs_data
            confs_model = 1-2*confs_model
            ngd.update_momentums(params, confs_data, confs_model)

            cross_entropy = ngd.cross_entropy_unnormalized + jnp.log(exact_sampler.partition_function)
            print(f'epoch: {epoch}, cross entropy: {cross_entropy}')

            if epoch % save_overlap_per == 0:
                overlap = exact_sampler.overlap(params)
                print(f'{epoch}\t{overlap}', file=overlap_out)
                overlap_out.flush()

            params = ngd.update(params)

        overlap = exact_sampler.overlap(params)
        print(f'{epoch}\t{overlap}', file=overlap_out)

    bytes_output = serialization.to_bytes(params)
    with open('model_final.fl', 'wb') as f:
        f.write(bytes_output)
