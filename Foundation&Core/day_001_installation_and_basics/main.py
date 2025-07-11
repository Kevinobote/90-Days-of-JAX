# day_001_installation_and_basics/main.py

import jax
import jax.numpy as jnp

# 1. Verify JAX installation and backend
print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")

# 2. Create your first JAX array
# Similar to NumPy's array creation
my_jax_array = jnp.array([1.0, 2.0, 3.0, 4.0])
print("\nMy first JAX array:")
print(my_jax_array)
print(f"Type: {type(my_jax_array)}")
print(f"Shape: {my_jax_array.shape}")
print(f"Dtype: {my_jax_array.dtype}")

# 3. Perform basic arithmetic operations
# JAX operations are just-in-time compiled on first execution
print("\nBasic arithmetic operations:")
sum_result = my_jax_array + 10.0
print(f"Array + 10: {sum_result}")

product_result = my_jax_array * 2.5
print(f"Array * 2.5: {product_result}")

division_result = my_jax_array / 2.0
print(f"Array / 2.0: {division_result}")

# Dot product with another array (similar to jnp.dot or @)
another_array = jnp.array([5.0, 6.0, 7.0, 8.0])
dot_product = jnp.dot(my_jax_array, another_array)
print(f"Dot product: {dot_product}")

# Element-wise multiplication
elementwise_mult = my_jax_array * another_array
print(f"Element-wise multiplication: {elementwise_mult}")

# More complex operation: sum of squares
sum_of_squares = jnp.sum(my_jax_array**2)
print(f"Sum of squares: {sum_of_squares}")