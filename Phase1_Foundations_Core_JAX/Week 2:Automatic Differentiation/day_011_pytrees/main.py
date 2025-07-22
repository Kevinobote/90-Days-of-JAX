# day_011_pytrees/main.py

import jax
import jax.numpy as jnp

print("--- Day 11: PyTrees and `jax.tree.map` (Advanced) ---")

# --- Part 1: What is a PyTree? ---
# A PyTree (Python Tree) is a nested Python structure (e.g., lists, tuples, dicts,
# dataclasses, custom classes) whose "leaves" are JAX arrays (or other non-container types).
# JAX transformations operate seamlessly on PyTrees.

print("\n--- Part 1: What is a PyTree? ---")

# Example 1: A simple PyTree (tuple of arrays)
pytree1 = (jnp.array([1.0, 2.0]), jnp.array(3.0))
print(f"PyTree 1: {pytree1}")
print(f"Type: {type(pytree1)}")
print(f"Type of leaves: {type(pytree1[0])}, {type(pytree1[1])}")

# Example 2: A more complex PyTree (dict of lists of arrays)
pytree2 = {
    'layer1': [jnp.array([10., 11.]), jnp.array(12.)],
    'layer2': {'weights': jnp.array([[1., 0.], [0., 1.]]), 'bias': jnp.array([0.5, 0.5])}
}
print(f"PyTree 2: {pytree2}")

# Example 3: A PyTree with non-array leaves (still a PyTree, non-array leaves are just passed through)
pytree3 = {'data': jnp.array([1., 2.]), 'metadata': 'some_string', 'config': 123}
print(f"PyTree 3: {pytree3}")


# --- Part 2: `jax.tree.map` (Applying a function to all leaves) ---
# This is the most common PyTree utility. It applies a function element-wise
# to the leaves of one or more PyTrees, maintaining the original structure.

print("\n--- Part 2: `jax.tree.map` ---")

# Example 1: Add a scalar to all leaves
def add_one(x):
    return x + 1.0

pytree1_plus_one = jax.tree.map(add_one, pytree1)
print(f"PyTree 1 + 1: {pytree1_plus_one}")

# Example 2: Scale all weights in pytree2 by 0.1
def scale_by_factor(x, factor):
    # We need to ensure 'factor' is applied only to actual arrays.
    # jax.tree.map implicitly handles non-array leaves by passing them through
    # if the function can't operate on them or if they aren't marked as leaves.
    # For this example, let's assume `x` is always a JAX array leaf.
    return x * factor

# jax.tree.map can take multiple PyTrees and a static argument
scaled_pytree2 = jax.tree.map(lambda x: x * 0.1, pytree2) # Simpler lambda for constant factor
print(f"Scaled PyTree 2 (by 0.1): {scaled_pytree2}")

# Example 3: Add two PyTrees element-wise
pytree_a = {'a': jnp.array([1., 2.]), 'b': {'c': jnp.array(3.), 'd': jnp.array([4., 5.])}}
pytree_b = {'a': jnp.array([10., 20.]), 'b': {'c': jnp.array(30.), 'd': jnp.array([40., 50.])}}

added_pytree = jax.tree.map(lambda x, y: x + y, pytree_a, pytree_b)
print(f"PyTree A:\n{pytree_a}")
print(f"PyTree B:\n{pytree_b}")
print(f"Added PyTree (A + B):\n{added_pytree}")

# Important: PyTrees must have the same structure for tree.map with multiple arguments
# try:
#    jax.tree.map(lambda x, y: x + y, pytree_a, {'a': jnp.array([1.]), 'b': jnp.array(2.)})
# except Exception as e:
#    print(f"\nCaught expected error for structural mismatch: {e}")


# --- Part 3: Other Useful PyTree Utilities ---
# `jax.tree.leaves`, `jax.tree.structure`, `jax.tree.unflatten`, `jax.tree.flatten`

print("\n--- Part 3: Other Useful PyTree Utilities ---")

# `jax.tree.leaves`: Extracts all the leaf values from a PyTree into a flat list.
leaves = jax.tree.leaves(pytree2)
print(f"Leaves of PyTree 2: {leaves}")

# `jax.tree.structure`: Returns an object representing the tree's structure.
# This structure can be used to re-create the original PyTree from a flat list of leaves.
structure = jax.tree.structure(pytree2)
print(f"Structure of PyTree 2: {structure}")

# `jax.tree.flatten`: Returns a tuple `(leaves, structure)`.
flat_leaves, tree_def = jax.tree.flatten(pytree2)
print(f"Flattened leaves: {flat_leaves}")
print(f"Tree definition: {tree_def}")

# `jax.tree.unflatten`: Reconstructs a PyTree from a list of leaves and a structure.
reconstructed_pytree2 = jax.tree.unflatten(tree_def, flat_leaves)
print(f"Reconstructed PyTree 2: {reconstructed_pytree2}")
print(f"Original and reconstructed are equal: {jax.tree.map(jnp.array_equal, pytree2, reconstructed_pytree2)}")


# --- Part 4: Practical Application - Parameter Initialization and Updates ---
# This is how Optax and your own training loops manage model parameters.

print("\n--- Part 4: Practical Application (ML Params) ---")

# Imagine `params` is your neural network's weights and biases
nn_params = {
    'layer1': {'w': jnp.array([[1., 2.], [3., 4.]]), 'b': jnp.array([0.1, 0.2])},
    'layer2': {'w': jnp.array([[5.], [6.]]), 'b': jnp.array([0.3])}
}
print(f"NN Params (initial):\n{nn_params}")

# Zero out all biases using tree.map
def zero_biases(leaf, key):
    if key == 'b': # 'key' here is a flat path segment, not unique
                   # A more robust check might involve full path (later topic)
        return jnp.zeros_like(leaf)
    return leaf

# `jax.tree.map` can work with a `is_leaf` predicate or specific key paths
# For now, let's just make a simple function to zero all 'b' fields IF they exist.
# Note: jax.tree.map applies to *all* leaves. We need to be careful if names can clash.
# A safer way to zero specific types of leaves, if `nn_params` were a custom class
# or if we knew the exact path. For a simple dict like this, direct access is clearer for specific items.

# Simpler: Initialize all params with random values
def init_random_param(key_leaf, shape, scale=0.01):
    # This is a bit contrived for this example, usually you pass a key into init_params.
    # But for applying a "randomizer" to existing shapes:
    return jax.random.normal(key_leaf, shape) * scale

# Let's say we want to apply a small update, like gradient descent
# (Imagine 'grads' is a PyTree of the same structure as 'nn_params')
grads = jax.tree.map(lambda x: x * 0.01, nn_params) # Mock gradients (scaled version of params)

learning_rate = 0.001
updated_params = jax.tree.map(lambda p, g: p - learning_rate * g, nn_params, grads)
print(f"NN Params (updated with mock grads):\n{updated_params}")

# This is exactly what `optax.apply_updates` does under the hood!

# --- Part 5: Registering Custom Types as PyTrees ---

print("\n--- Part 5: Registering Custom Types as PyTrees ---")

from dataclasses import dataclass

@dataclass(frozen=True)
class MyCustomLayer:
    weights: jax.Array
    bias: jax.Array
    name: str = "custom_layer"

custom_layer_instance = MyCustomLayer(
    weights=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
    bias=jnp.array([0.5, 0.5])
)

print(f"Custom Layer Instance: {custom_layer_instance}")

try:
    jax.tree.map(lambda x: x * 2, custom_layer_instance)
except TypeError as e:
    print(f"Expected error before registration: {e}")

def flatten_my_custom_layer(layer):
    children = (layer.weights, layer.bias)
    aux_data = {'name': layer.name}
    return children, aux_data

def unflatten_my_custom_layer(aux_data, children):
    weights, bias = children
    return MyCustomLayer(weights=weights, bias=bias, name=aux_data['name'])

# CORRECTED LINE: Use jax.tree_util.register_pytree_node
jax.tree_util.register_pytree_node(MyCustomLayer, flatten_my_custom_layer, unflatten_my_custom_layer) # <--- CHANGE THIS LINE
print("\n`MyCustomLayer` registered as a PyTree node.")

doubled_layer = jax.tree.map(lambda x: x * 2, custom_layer_instance)
print(f"Doubled Custom Layer (using tree.map): {doubled_layer}")

print(f"Original name: {custom_layer_instance.name}, Doubled name: {doubled_layer.name}")