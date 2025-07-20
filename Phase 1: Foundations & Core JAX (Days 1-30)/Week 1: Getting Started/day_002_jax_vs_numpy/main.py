# day_002_jax_vs_numpy

import jax
import jax.numpy as jnp
import numpy as np
import time

print("--- Day 2: JAX vs NumPy ---")

print("\n--- Immutability Demo ---")

# NumpPy array: mutable
np_array = np.array([1, 2, 3])
print(f"Original NumPy array: {np_array} (ID: {id(np_array)})")
np_array[0] = 99
print(f"Modified NumPy array: {np_array} (ID: {id(np_array)}) - Same ID, modified in place")

# JAX array: immutable
jax_array = jax.numpy.array([1, 2, 3])
print(f"\nOriginal JAX array: {jax_array} (ID: {id(jax_array)})")

# Attempting in-place modification will raise an error (or a warning in some versions)
# JAX does not allow direct item assignment like NumPy
try:
    jax_array[0] = 99 # This will raise an error
    print("This line should not be reached if JAX enforces immutability strictly.")
except TypeError as e:
    print(f"Attempted in-place modification on JAX array: {e}")

# Correct way to "modify" a JAX array is to create a new one
new_jax_array = jax_array.at[0].set(99)
print(f"JAX array after .at[0].set(99): {new_jax_array} (ID: {id(new_jax_array)})")
print(f"Original JAX array remains unchanged: {jax_array} (ID: {id(jax_array)})")
print("Notice the new JAX array has a differnt ID.")

# --- Part 2: Device Arrays & Computation ---

print("\n--- Device Arrays & Computation ---")

# NumPy operations are generally executed on the CPU
np_large_array = np.random.rand(5000, 5000)
start_time = time.time()
np_result = np_large_array @ np_large_array
end_time = time.time()
print(f"\nNumPy computation time(CPU): {end_time - start_time:.4f} seconds")

# JAX operations are dispatched to available accelerators (GPU/TPU) if present, or CPU otherwise
# The first execution might include compilation overhead
jax_large_array = jnp.array(np_large_array)  # Copying NumPy array to JAX device array
print(f"JAX array backend: {jax_large_array.device}")

# JIT compile the matrix multiplication for performance
@jax.jit
def matmul(x):
    return x @ x

# First run includes compilation time
start_time = time.time()
jax_result_compiled = matmul(jax_large_array).block_until_ready() # .block_until_ready() waits for computation to finish
end_time = time.time()
print(f"JAX matrix multiplication (First run, includes compile): {end_time - start_time:.4f} seconds")

# Subsequent runs are much faster due to JIT compilation
start_time = time.time()
jax_result_recomplied = matmul(jax_large_array).block_until_ready()
end_time = time.time()
print(f"JAX matrix multiplication (Subsequent run, compiled): {end_time - start_time:.4f} seconds")

# Verify results are the close
print(f"\nAre NumPy and JAX results close? {jnp.allclose(np_result, jax_result_compiled)}")

# --- Part 3: Explicit type casting when needed ---
print("\n--- Explicit Type Casting --- ")
int_jax_array = jnp.array([1, 2, 3])
print(f"Default int JAX array dtype: {int_jax_array.dtype}")

float_jax_array = jnp.array([1, 2, 3], dtype=jnp.float32)
print(f"Explicit float32 JAX array dtype: {float_jax_array.dtype}")