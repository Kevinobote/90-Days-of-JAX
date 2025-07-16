# day_006_prng/main.py

import jax
import jax.numpy as jnp
import time # Make sure time is imported for timing

print("--- Day 6: JAX's Pseudo-Random Number Generation (PRNG) ---")

# --- Part 1: Creating PRNG Keys ---

# JAX's PRNG is based on a PRNGKey.
# You typically start with a single "master" key.
master_key = jax.random.PRNGKey(0) # The '0' is the seed

print("\n--- Creating PRNG Keys ---")
print(f"Master Key: {master_key}")
print(f"Type of Master Key: {type(master_key)}") # It's a JAX Array with dtype uint32

# Keys are typically split to create new, independent keys for different operations.
# This ensures reproducibility and avoids re-using the same key, which can lead to
# correlated random numbers.
key1, key2 = jax.random.split(master_key)
print(f"Split Key 1: {key1}")
print(f"Split Key 2: {key2}")

# You can split a key into N subkeys
key_for_weights, key_for_biases, key_for_data = jax.random.split(master_key, num=3)
print(f"Split for weights: {key_for_weights}")
print(f"Split for biases: {key_for_biases}")
print(f"Split for data: {key_for_data}")

# Re-using a key is generally bad practice, as it can lead to correlated randomness
# and non-reproducible results. Always split!
print("\n--- Illustrating Key Re-use (Bad Practice) ---")
key_reused = jax.random.PRNGKey(10)
rand1 = jax.random.normal(key_reused, (2,)) # Use key_reused
rand2 = jax.random.normal(key_reused, (2,)) # Re-use the SAME key_reused
print(f"First random numbers (key_reused): {rand1}")
print(f"Second random numbers (key_reused - SAME key): {rand2}")
print("Notice they are the same! Avoid this unless intended for specific reasons.")

# Correct way to get independent random numbers
key_good_practice, subkey1_good, subkey2_good = jax.random.split(jax.random.PRNGKey(11), num=3)
rand3 = jax.random.normal(subkey1_good, (2,))
rand4 = jax.random.normal(subkey2_good, (2,))
print(f"\n--- Illustrating Good Practice (Splitting Keys) ---")
print(f"First random numbers (subkey1_good): {rand3}")
print(f"Second random numbers (subkey2_good): {rand4}")
print("Notice they are different, as expected.")


# --- Part 2: Generating Random Numbers ---

print("\n--- Generating Random Numbers ---")

# Normal distribution
normal_samples_key = jax.random.PRNGKey(20)
samples_normal = jax.random.normal(normal_samples_key, (5,)) # (key, shape)
print(f"5 normal samples: {samples_normal}")

# Uniform distribution (between 0 and 1 by default)
uniform_samples_key = jax.random.PRNGKey(21)
samples_uniform = jax.random.uniform(uniform_samples_key, (3, 2)) # (key, shape)
print(f"3x2 uniform samples: \n{samples_uniform}")

# Uniform distribution (with specified min/max)
uniform_range_samples_key = jax.random.PRNGKey(22)
samples_uniform_range = jax.random.uniform(uniform_range_samples_key, (4,), minval=10.0, maxval=20.0)
print(f"4 uniform samples in [10, 20): {samples_uniform_range}")

# Integers (inclusive low, exclusive high)
randint_samples_key = jax.random.PRNGKey(23)
samples_randint = jax.random.randint(randint_samples_key, (5,), minval=0, maxval=10) # (key, shape, minval, maxval)
print(f"5 random integers in [0, 10): {samples_randint}")

# Bernoulli (coin flips)
bernoulli_samples_key = jax.random.PRNGKey(24)
samples_bernoulli = jax.random.bernoulli(bernoulli_samples_key, p=0.7, shape=(6,))
print(f"6 Bernoulli samples (p=0.7): {samples_bernoulli}")


# --- Part 3: PRNG and JIT ---

# Key splitting should happen *inside* the JITted function if the random numbers
# are generated on each call. If the keys are static, split outside.
# Generally, it's better to pass the key in and split inside.

@jax.jit
def generate_random_numbers_jitted(key):
    key1, key2 = jax.random.split(key) # Splitting inside JIT is fine
    rand_a = jax.random.normal(key1, (3,))
    rand_b = jax.random.uniform(key2, (2,))
    return rand_a, rand_b # This returns a tuple of JAX arrays

print("\n--- PRNG and JIT ---")
jit_main_key = jax.random.PRNGKey(30)

# First call (includes compile time)
start_time = time.time()
# Unpack the tuple and then call block_until_ready() on one of them
# This ensures all outputs from the JITted function are ready.
jit_output1_a, jit_output1_b = generate_random_numbers_jitted(jit_main_key)
jit_output1_a.block_until_ready() # Call block_until_ready() on one of the JAX arrays
end_time = time.time()
print(f"JITted random numbers (first run): {jit_output1_a}, {jit_output1_b}, Time: {end_time - start_time:.6f} seconds")

# Subsequent call (pass a NEW subkey to get new random numbers)
# The JITted function's input key needs to be different to get different randomness.
jit_main_key, next_subkey = jax.random.split(jit_main_key) # Update the key for the next call
start_time = time.time()
jit_output2_a, jit_output2_b = generate_random_numbers_jitted(next_subkey)
jit_output2_a.block_until_ready() # Call block_until_ready() on one of the JAX arrays
end_time = time.time()
print(f"JITted random numbers (subsequent run with NEW key): {jit_output2_a}, {jit_output2_b}, Time: {end_time - start_time:.6f} seconds")

# Subsequent call (pass the SAME subkey to illustrate predictability inside JIT)
start_time = time.time()
jit_output3_a, jit_output3_b = generate_random_numbers_jitted(next_subkey)
jit_output3_a.block_until_ready() # Call block_until_ready() on one of the JAX arrays
end_time = time.time()
print(f"JITted random numbers (subsequent run with SAME key): {jit_output3_a}, {jit_output3_b}, Time: {end_time - start_time:.6f} seconds")
print("Notice jit_output2 and jit_output3 are the same because the input key was the same.")