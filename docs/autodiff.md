# Post 2: AI/ML in Rust: Unleashing Autodiff (Gradients Explained\!)

As someone who loves learning new things, I often find myself exploring various fields. Lately, I've been diving into Rust, with AI and machine learning always on my mind. When I discovered a growing Rust AI ecosystem, it clicked – why not combine both? I'm no expert in either Rust or AI, so this is a learning journey. If you have any insights or corrections, please share them here or on my [GitHub](https://github.com/spaghetty/rust_ai_ml_adventure). Let's learn together\!

## What’s for today.

Welcome back to our AI/ML journey with Rust\! In the last post, we explored Tensors – the foundation of our data and computations. Now, let’s dive into **Automatic Differentiation (Autodiff)**, the key to how neural networks learn. This is what allows models to fine-tune their parameters for better performance.

In this post, I'll explain what Autodiff is, why it's essential, and how Burn simplifies this powerful technique. Let's explore gradients\!

# What is Autodiff and Why Does it Matter?

Think of a neural network as a machine that takes input, processes it, and produces output. We want to improve this machine. We compare its output to the correct one, and the difference is our "loss" or "error."

To optimize the network, we need to know how adjusting its internal settings (parameters) affects this error. That’s where **gradients** come in. A gradient tells us the direction and magnitude of change needed to minimize the error.

**Automatic Differentiation** automatically calculates these gradients for every operation in our network's calculations. It saves us from manually working out complex math\!

Why is this important? Without gradients, we can't train our models. Training is about finding the best set of parameters to minimize the loss. Gradient Descent uses these gradients to iteratively refine the parameters.

# Autodiff in Burn: The `AutodiffBackend`

Burn enables Autodiff via the `AutodiffBackend`. Recall `MyBackend` from Post 1:

```rust
type MyBackend = Autodiff<NdArray>;
```

By using `Autodiff`, we instruct Burn to track computations for gradient calculations.

When a tensor is created using `Autodiff`, it becomes part of a computational graph.

# Tracking Gradients: The `.require_grad()` Method

Not all tensors need gradient tracking. We typically track gradients for the learnable parameters (weights and biases).

Use `.require_grad()` to mark a tensor for gradient tracking in Burn.

**Let's see a simple example:**
```rust
use burn::tensor::{Tensor, backend::Backend, Autodiff};
use burn::backend::NdArray;

type MyAutodiffBackend = Autodiff<NdArray>;

pub fn simple_autodiff_example() {
    let device = Default::default();

    // 1. Create Tensors (some require gradients)
    let a = Tensor::<MyAutodiffBackend, 1>::from_data([10.0], &device).require_grad();
    let b = Tensor::<MyAutodiffBackend, 1>::from_data([5.0], &device).require_grad();
    let c = Tensor::<MyAutodiffBackend, 1>::from_data([2.0], &device);

    // 2. Perform operations: y \= (a \* b) \+ c
    let y = (a.clone() * b.clone()) + c.clone(); // -> [52.0]

    // 3. Backward Pass (calculate gradients)
    let gradients = y.backward();

    // 4. Access Gradients
    let grad_a = a.grad(&gradients).unwrap(); // -> [5.0]
    let grad_b = b.grad(&gradients).unwrap(); // -> [10.0]

    println!("Gradient of y w.r.t. 'a': {:?}", grad_a.to_data());
    println!("Gradient of y w.r.t. 'b': {:?}", grad_b.to_data());
}
```
**Math for the example:**

* `y = (a * b) + c`
* `a = 10.0`, `b = 5.0`, `c = 2.0`
* `y = 52.0`

Gradients:

* `dy/da = b = 5.0`
* `dy/db = a = 10.0`

Run the code to see these values\! Burn computes these derivatives automatically.

# Autodiff in Action: Linear Neuron Simulation

Let's simulate a linear neuron, a basic neural network building block.
```rust
use burn::tensor::{Tensor, backend::Backend, Autodiff};
use burn::backend::NdArray;

type MyAutodiffBackend = Autodiff\<NdArray\>;

pub fn linear_neuron_autodiff_example() {
    let device = Default::default();
    let learning_rate = 0.01f32;

    // 1. Define neuron parameters and input
    let x = Tensor::<MyAutodiffBackend, 1>::from_data([2.0], &device);
    let mut w = Tensor::<MyAutodiffBackend, 1>::from_data([3.0], &device).require_grad();
    let mut b = Tensor::<MyAutodiffBackend, 1>::from_data([1.0], &device).require_grad();
    let target = Tensor::<MyAutodiffBackend, 1>::from_data([10.0], &device);

    for _ in 0..100 {
        // 2. Forward pass: y = (x * w) + b
        let y = (x.clone() * w.clone()) + b.clone();
        // 3. Calculate MSE loss
        let loss = (y - target.clone()).powf_scalar(2.0).mean();

        // 4. Backward pass
        let gradients = loss.backward();

        // 5. Access gradients
        let grad_w = w.grad(&gradients).unwrap();
        let grad_b = b.grad(&gradients).unwrap();

        // 6. Apply gradients (gradient descent)
        let grad_w_autodiff = Tensor::<MyAutodiffBackend, 1>::from_data(grad_w.into_data(), &device);
        let grad_b_autodiff = Tensor::<MyAutodiffBackend, 1>::from_data(grad_b.into_data(), &device);

        w = w.sub(grad_w_autodiff * learning_rate).detach().require_grad();
        b = b.sub(grad_b_autodiff * learning_rate).detach().require_grad();
    }

    println!("--- Training Loop Finished ---");
    let final_y = (x * w) + b;
    let final_loss = (final_y - target.clone()).powf_scalar(2.0).mean();
    println!("Final Loss: {:}", final_loss.to_data());
}
```

# The Autodiff Flow (Simplified):

1. **Tensor Creation:** Create tensors with `Autodiff`. Use `.require_grad()` for tracking.
2. **Operations:** Perform operations; Burn tracks the graph.
3. **Backward Pass:** `.backward()` computes gradients.
4. **Gradient Access:** `.grad()` retrieves gradients.

# What's Next?

We’ve covered Tensors and Autodiff. In the next post, we'll use Modules and Configs to automate parameter management and build neural networks.

*Questions about gradients? Let’s discuss\!*
