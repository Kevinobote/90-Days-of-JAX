# Day 008: Basic Neural Network (MLP) in JAX

## Goal
- Implement a simple Multi-Layer Perceptron (MLP) from scratch using JAX.
- Understand how to define model parameters as a PyTree.
- Learn to use `optax` for optimization in JAX.
- Combine `jax.grad`, `jax.jit`, and `jax.random` within a training loop.

## Key Learnings
- **MLP Architecture:** A basic neural network consists of layers of neurons, typically with an activation function between them (e.g., ReLU). The core operations are matrix multiplications (`jnp.dot`) and additions (biases).
- **Parameters as PyTrees:** JAX functions (like `grad`, `jit`) operate seamlessly on "PyTrees" – nested Python structures (lists, tuples, dicts) containing JAX arrays. This is how model parameters are naturally represented and handled in JAX.
- **`jax.nn`:** JAX provides common neural network activation functions (e.g., `jax.nn.relu`, `jax.nn.sigmoid`, `jax.nn.softmax`).
- **`optax`:** This is the canonical JAX library for optimizers (like Adam, SGD, RMSprop).
    - `optimizer = optax.adam(learning_rate)`: Initializes the optimizer.
    - `opt_state = optimizer.init(params)`: Initializes the optimizer's internal state (e.g., momentum buffers for Adam) based on your model parameters.
    - `updates, new_opt_state = optimizer.update(grads, opt_state, params)`: Computes parameter updates based on gradients and current optimizer state.
    - `new_params = optax.apply_updates(params, updates)`: Applies the computed `updates` to your `params` to get the `new_params`.
- **Training Loop:** A typical JAX training loop involves:
    1.  Defining initial parameters using `jax.random`.
    2.  Initializing an optimizer from `optax`.
    3.  Iterating over epochs:
        a.  Computing the loss and its gradients (`jax.value_and_grad`).
        b.  Updating optimizer state and getting parameter updates (`optimizer.update`).
        c.  Applying updates to parameters (`optax.apply_updates`).
    4.  JIT-compiling the `update_step` function is crucial for performance.
- **`jax.value_and_grad`:** A convenient transformation that returns both the scalar value of the function and its gradient.

## Code Explanation (`main.py`)
- **Data Generation:** Creates a simple 1D linear regression-like dataset with noise.
- **`init_params`:** Function to initialize the weights and biases of the MLP using `jax.random.normal` for weights and `jnp.zeros` for biases.
- **`forward_pass`:** Defines the neural network's architecture, including a hidden layer with ReLU activation and a linear output layer.
- **`loss_fn`:** Calculates the Mean Squared Error (MSE) between predictions and true labels.
- **`update_step`:** This is the core training step function. It uses `jax.value_and_grad` to get loss and gradients, then `optax` to apply the updates, and is `jax.jit`-compiled for efficiency.
- **Training Loop:** Iterates for a set number of epochs, calling the `update_step` to train the model.
- **Visualization:** Plots the training loss over epochs and the final model's predictions against the original data.

## Challenges/Notes
- This is a basic example; real-world NNs involve more complex architectures, data loading, batching, and regularization.
- The `optax` library is a powerful tool for various optimization algorithms.
- Pay attention to the shapes of your arrays, especially in matrix multiplications (`jnp.dot`).
- The `jax.tree_map` utility is very handy for inspecting PyTrees (like your `params` dictionary).