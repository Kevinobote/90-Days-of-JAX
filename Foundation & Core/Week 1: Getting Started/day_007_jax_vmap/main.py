# day_007_jax_vmap/main.py

import jax
import jax.numpy as jnp
import time

print("--- Day 7: Vectorization with `jax.vmap` ---")

# --- Part 1: Basic Vmap Application ---

# Define a function that operates on a single scalar
def elementwise_multiply_add(x, y):
    return x * 2 + y

print("\n--- Basic Vmap Application ---")
single_x = jnp.array(5.0)
single_y = jnp.array(3.0)
print(f"Single element calculation: {elementwise_multiply_add(single_x, single_y)}")

# Suppose we have batches of x and y, and we want to apply the function element-wise.
# Without vmap, we might loop (slow and not JIT-compatible across loop iterations)
batch_x = jnp.array([1.0, 2.0, 3.0])
batch_y = jnp.array([10.0, 20.0, 30.0])

# Manual loop (inefficient for large arrays)
results_manual_loop = [elementwise_multiply_add(batch_x[i], batch_y[i]) for i in range(len(batch_x))]
print(f"Manual loop results: {jnp.array(results_manual_loop)}")

# With vmap: automatically batches the operation
# `in_axes=0` means the 0-th dimension of the input arrays corresponds to the batch dimension.
# Since x and y are (3,) arrays, their 0-th dim is the batch dim.
vmap_elementwise = jax.vmap(elementwise_multiply_add, in_axes=(0, 0)) # or in_axes=0 if all inputs have batch on 0th axis
                                                                    # in_axes=(0, 0) means map over 0th axis of first arg, 0th axis of second arg
results_vmap = vmap_elementwise(batch_x, batch_y)
print(f"vmap results: {results_vmap}")
print(f"Are results close? {jnp.allclose(jnp.array(results_manual_loop), results_vmap)}")


# --- Part 2: Vmap with different input axes (`in_axes`) ---

print("\n--- Vmap with different input axes (`in_axes`) ---")

# Function: dot product of a vector with a matrix row
# def dot_product(vector, matrix_row):
#     return jnp.dot(vector, matrix_row)

# Let's rephrase to make it clearer for vmap:
# `matrix_vector_product_row(vec, row)`: computes dot product of `vec` with `row`
def matrix_vector_product_row(vec, row):
    return jnp.dot(vec, row)

vector = jnp.array([1.0, 2.0]) # Shape (2,)
matrix = jnp.array([[10.0, 11.0],  # Shape (3, 2)
                    [20.0, 21.0],
                    [30.0, 31.0]])

# We want to compute dot_product(vector, row) for each row in the matrix.
# `vector` should not be batched (None), `matrix`'s 0-th axis is the batch axis (0).
vmap_matrix_vector_product = jax.vmap(matrix_vector_product_row, in_axes=(None, 0))
# Expected output:
# 1*10 + 2*11 = 10 + 22 = 32
# 1*20 + 2*21 = 20 + 42 = 62
# 1*30 + 2*31 = 30 + 62 = 92
results_mvp = vmap_matrix_vector_product(vector, matrix)
print(f"Vector: {vector}")
print(f"Matrix:\n{matrix}")
print(f"vmap (vector, matrix rows) results: {results_mvp}")

# Another example: Batch matrix-vector product
# A batch of matrices, a batch of vectors
# (batch_size, M, N) @ (batch_size, N) -> (batch_size, M)
def matmul_single_vec(matrix_2d, vector_1d):
    return matrix_2d @ vector_1d # Uses @ for matrix-vector product

batch_matrices = jnp.array([[[1., 2.], [3., 4.]],
                             [[5., 6.], [7., 8.]]]) # Shape (2, 2, 2)
batch_vectors = jnp.array([[10., 11.],
                            [12., 13.]])          # Shape (2, 2)

# Both inputs have their batch dimension at axis 0.
vmap_batch_matmul = jax.vmap(matmul_single_vec, in_axes=(0, 0))
results_batch_mv = vmap_batch_matmul(batch_matrices, batch_vectors)
print(f"\nBatch matrices:\n{batch_matrices}")
print(f"Batch vectors:\n{batch_vectors}")
print(f"vmap (batch matrix @ batch vector) results:\n{results_batch_mv}")
# Expected:
# [1*10 + 2*11, 3*10 + 4*11] = [10+22, 30+44] = [32, 74]
# [5*12 + 6*13, 7*12 + 8*13] = [60+78, 84+104] = [138, 188]


# --- Part 3: Vmap and JIT (common and powerful combination) ---

@jax.jit
@jax.vmap
def jit_vmap_multiply_add(x, y):
    return x * 2 + y

print("\n--- Vmap and JIT ---")
large_batch_x = jax.random.normal(jax.random.PRNGKey(0), (1000000,))
large_batch_y = jax.random.normal(jax.random.PRNGKey(1), (1000000,))

start_time = time.time()
# When using decorators, vmap is applied first, then jit.
# So, @vmap then @jit is effectively jax.jit(jax.vmap(...)).
# If the function takes multiple arguments, you need to specify in_axes in vmap.
# Decorator order matters. @vmap @jit is actually jit(vmap(func)).
# So, if `vmap` needs `in_axes`, the decorator form is tricky.
# It's often clearer to apply them explicitly:
# `compiled_function = jax.jit(jax.vmap(my_func, in_axes=(...)))`

# Let's define it explicitly for clarity as the decorator syntax for vmap with args is non-trivial.
vmapped_multiply_add_explicit = jax.vmap(elementwise_multiply_add, in_axes=(0, 0))
jit_vmapped_multiply_add = jax.jit(vmapped_multiply_add_explicit)

result_jit_vmap_first = jit_vmapped_multiply_add(large_batch_x, large_batch_y).block_until_ready()
end_time = time.time()
print(f"JIT + Vmap (First Run on large batch): {end_time - start_time:.6f} seconds")

start_time = time.time()
result_jit_vmap_subsequent = jit_vmapped_multiply_add(large_batch_x, large_batch_y).block_until_ready()
end_time = time.time()
print(f"JIT + Vmap (Subsequent Run on large batch): {end_time - start_time:.6f} seconds")

# For comparison, a simple non-JITted, non-vmapped loop (will be very slow for large data)
# This loop simulates what vmap avoids but should not be directly run for 1M elements.
# Skipping direct execution for speed, but mentally note its inefficiency.
# print("\n--- Manual loop (for conceptual comparison, not to be run for large N) ---")
# if len(large_batch_x) < 1000: # Only run for small sizes to avoid freezing
#    start_time = time.time()
#    manual_results = jnp.array([elementwise_multiply_add(large_batch_x[i], large_batch_y[i]) for i in range(len(large_batch_x))])
#    end_time = time.time()
#    print(f"Manual loop (small batch): {end_time - start_time:.6f} seconds")
# else:
#    print("Manual loop for 1M elements skipped for performance reasons.")