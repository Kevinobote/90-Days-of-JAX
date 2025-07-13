# day_003_intro_jit/main.py

import jax
import jax.numpy as jnp
import time

print("--- Day 3: Just-in-Time (JIT) Compilation ---")

# --- Part 1: Basic JIT application ---

# Define a simple function
def add_and_multiply(x, y):
    return (x + y) * 2.0

print("\n--- Basic JIT Application ---")
x = jnp.array(5.0)
y = jnp.array(3.0)

# Run without JIT
start_time = time.time()
result_nojit = add_and_multiply(x, y)
result_nojit.block_until_ready()
end_time = time.time()
print(f"Result (No JIT): {result_nojit}, Time: {end_time - start_time:.6f} seconds")


# Apply JIT to the function
jit_add_and_multiply = jax.jit(add_and_multiply)

# First run with JIT: includes compilation overhead
start_time = time.time()
result_jit_first = jit_add_and_multiply(x, y)
result_jit_first.block_until_ready()
end_time = time.time()
print(f"Result (JIT First Run): {result_jit_first}, Time: {end_time - start_time:.6f} seconds (includes compilation)")

# Subsequent runs with JIT: much faster
start_time = time.time()
result_jit_subsequent = jit_add_and_multiply(x, y)
result_jit_subsequent.block_until_ready()
end_time = time.time()
print(f"Result (JIT Subsequent Run): {result_jit_subsequent}, Time: {end_time - start_time:.6f} seconds (compiled)")


# --- Part 2: JIT and Python Control Flow ---

print("\n--- JIT and Python Control Flow ---")

# The original function for `conditional_sum_static`
def conditional_sum(a, b, condition):
    # 'condition' here is a Python boolean, treated as static
    if condition:
        return a + b
    else:
        return a - b

# Option 1: Use `static_argnums` for the condition by explicitly wrapping
# This tells JAX that `condition` is a static argument, whose *value* is used for compilation.
# If `condition` changes, JAX will recompile.
jit_conditional_sum_static = jax.jit(conditional_sum, static_argnums=(2,))

print("\n--- Conditional Sum with `static_argnums` for condition ---")
# First call with condition=True
print("Calling with condition=True (static_argnums):")
start_time = time.time()
res1_static = jit_conditional_sum_static(jnp.array(10), jnp.array(5), True).block_until_ready()
end_time = time.time()
print(f"Result: {res1_static}, Time: {end_time - start_time:.6f}s (potentially compiles)")

# Second call with condition=True (should be fast, no recompile)
start_time = time.time()
res2_static = jit_conditional_sum_static(jnp.array(10), jnp.array(5), True).block_until_ready()
end_time = time.time()
print(f"Result: {res2_static}, Time: {end_time - start_time:.6f}s (compiled)")

# Call with condition=False (triggers re-compilation for the new static value)
print("\nCalling with condition=False (static_argnums):")
start_time = time.time()
res3_static = jit_conditional_sum_static(jnp.array(10), jnp.array(5), False).block_until_ready()
end_time = time.time()
print(f"Result: {res3_static}, Time: {end_time - start_time:.6f}s (recompiles for new static value)")

# Option 2: Use `jax.lax.cond` for dynamic conditions
# This is the preferred way when the condition itself might be a JAX array or
# its value isn't known until runtime.
@jax.jit
def conditional_sum_lax(a, b, condition_array):
    # condition_array should be a boolean JAX array (e.g., jnp.array(True))
    return jax.lax.cond(
        condition_array,          # The condition (must be a JAX array with bool dtype)
        lambda: a + b,            # True branch function (no args)
        lambda: a - b             # False branch function (no args)
    )

print("\n--- Conditional Sum with `jax.lax.cond` for dynamic conditions ---")
# Call with condition=True (as a JAX array)
print("Calling with condition=jnp.array(True):")
start_time = time.time()
res1_lax = conditional_sum_lax(jnp.array(10), jnp.array(5), jnp.array(True)).block_until_ready()
end_time = time.time()
print(f"Result: {res1_lax}, Time: {end_time - start_time:.6f}s (potentially compiles)")

# Call with condition=False (as a JAX array) - no recompile with `lax.cond`
# because the *structure* of the computation graph is fixed.
print("\nCalling with condition=jnp.array(False):")
start_time = time.time()
res2_lax = conditional_sum_lax(jnp.array(10), jnp.array(5), jnp.array(False)).block_until_ready()
end_time = time.time()
print(f"Result: {res2_lax}, Time: {end_time - start_time:.6f}s (compiled, no recompile)")


# --- Part 3: What JIT can't (easily) do / Static Arguments (Revised) ---
print("\n--- JIT Limitations / Static Arguments for Loop Bounds ---")

def sum_n_elements(arr, n):
    # This loop depends on 'n', a Python integer.
    # 'n' MUST be a static argument for Python's `range` to work inside JIT.
    total = 0.0
    for i in range(n): # JAX can trace this because `n` is declared static
        total += arr[i]
    return total

# With static_argnums, 'n' is passed to the compiler as a concrete Python integer.
# The 'n' argument is at index 1 (0-indexed).
jit_sum_n_elements_static = jax.jit(sum_n_elements, static_argnums=(1,))

arr_data = jnp.arange(10.0)

print("\nCalling sum_n_elements (n static_argnums=1):")
start_time = time.time()
res_static1 = jit_sum_n_elements_static(arr_data, 3).block_until_ready()
end_time = time.time()
print(f"Sum (n=3): {res_static1}, Time: {end_time - start_time:.6f}s (potentially compiles)")

start_time = time.time()
res_static2 = jit_sum_n_elements_static(arr_data, 5).block_until_ready() # This will still recompile for a *new* value of `n`
                                                                        # because the trace depends on 'n's concrete value.
end_time = time.time()
print(f"Sum (n=5): {res_static2}, Time: {end_time - start_time:.6f}s (compiles for new static value)")

start_time = time.time()
res_static3 = jit_sum_n_elements_static(arr_data, 3).block_until_ready() # Should be fast now for n=3 (re-uses compiled code)
end_time = time.time()
print(f"Sum (n=3): {res_static3}, Time: {end_time - start_time:.6f}s (re-use compiled code)")