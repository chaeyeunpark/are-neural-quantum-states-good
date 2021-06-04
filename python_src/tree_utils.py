import jax
import jax.numpy as jnp
import numpy as np

def to_list(p, axis = None):
    flatten_p = jax.tree_leaves(p)
    return jnp.concatenate(flatten_p, axis = None)

   
def reshape_tree_like(flattend_array, params):
    params_flat, params_tree = jax.tree_flatten(params)
    shapes = [v.shape for v in params_flat]
    sizes = [np.prod(v) for v in shapes]
    splitted = jnp.split(flattend_array, indices_or_sections = np.cumsum(sizes)[:-1])
    reshaped = [v.reshape(shape) for v, shape in zip(splitted, shapes)]
    return jax.tree_unflatten(params_tree, reshaped)


