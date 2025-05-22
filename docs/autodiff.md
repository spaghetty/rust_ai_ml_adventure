# Post 3: AI and ML in Rust: The Magic of Autodiff (Gradients, Baby!) - My Journey

Alright team, welcome back to our wild AI/ML adventure in Rust! In our last post, we wrestled with **Tensors** – the fundamental building blocks of all our data and computations. We learned how to create them, understand their shapes, and do some basic math.

Now, we're about to unlock the real magic behind how neural networks actually *learn*: **Automatic Differentiation**, or **Autodiff** for short. This is the secret sauce that lets our models adjust their internal knobs (parameters) to get better at their tasks.

In this post, I'll share *my* findings on what Autodiff is, why it's so incredibly important, and how Burn makes this seemingly complex process surprisingly straightforward. Get ready for some gradients!

## What is Autodiff and Why Do We Care?

Imagine you have a machine that takes some input, does a bunch of calculations, and spits out an output. For our AI/ML models, this machine is our neural network.

Now, imagine you want to make this machine better at its job. You feed it some input, it gives an output, and you compare that to what the *correct* output should have been. The difference is our "error" or "loss."

To make the machine better, you need to know: "If I tweak *this* internal knob (a parameter), how much will it reduce the error?" This "how much" is what a **gradient** tells you. It's like the slope of a hill – it tells you the direction and steepness to go to reach the bottom (minimum error).

**Automatic Differentiation** is a technique that automatically calculates these gradients for *every single operation* in your computational graph. You don't have to manually derive complex calculus equations; the framework does it for you!

Why do we care? Because without gradients, we can't train our models! Training is essentially an optimization problem: we want to find the set of model parameters that minimize our loss function. Gradient Descent (and its fancier cousins) uses these gradients to iteratively update the parameters, nudging them in the right direction.

## Autodiff in Burn: The `AutodiffBackend`

In Burn, the magic of Autodiff is enabled through the `AutodiffBackend`. Remember our `MyBackend` type alias from Post 1?

type MyBackend = Autodiff<NdArray>;

By wrapping `NdArray` (or `Wgpu`, `Tch`, etc.) with `Autodiff`, we're telling Burn: "Hey, for any tensor operations using this backend, please keep track of the computations so we can calculate gradients later!"

When you create a tensor using an `Autodiff` backend, it implicitly becomes part of a computational graph.

## Tracking Gradients: The `.require_grad()` Method

Not every tensor needs its gradients tracked. For example, your input data (like an image) doesn't usually need gradients computed with respect to it. You only want to compute gradients for the **learnable parameters** of your model (like weights and biases).

In Burn, when you create a tensor that you *do* want to compute gradients for, you use the `.require_grad()` method. This explicitly marks the tensor as needing gradient tracking.

Let's see this in action with a super simple example:

```rust
use burn::tensor::{Tensor, backend::Backend, Autodiff}; // Add Autodiff
use burn::backend::NdArray; // Our chosen backend

// Define our Autodiff backend type
type MyAutodiffBackend = Autodiff<NdArray>;

pub fn simple_autodiff_example() {
    let device = Default::default(); // Get the default device for NdArray

    // --- 1. Create Tensors, some requiring gradients ---
    // Tensor 'a' - we want to track its gradient
    let a = Tensor::<MyAutodiffBackend, 1>::from_data([10.0f32], &device).require_grad();
    println!("Tensor 'a': {:}", a.to_data());

    // Tensor 'b' - also track its gradient
    let b = Tensor::<MyAutodiffBackend, 1>::from_data([5.0], &device).require_grad();
    println!("Tensor 'b': {:}", b.to_data());

    // Tensor 'c' - does NOT require gradient (e.g., input data or constant)
    let c = Tensor::<MyAutodiffBackend, 1>::from_data([2.0], &device);
    println!("Tensor 'c': {:}", c.to_data());

    // --- 2. Perform some operations (build the computational graph) ---
    // Let's calculate: y = (a * b) + c
    let y = (a.clone() * b.clone()) + c.clone(); // Clone if you need original tensors later
    println!("\nCalculated 'y' = (a * b) + c: {:}", y.to_data());

    // --- 3. Perform the Backward Pass ---
    // This is where the magic happens! We call .backward() on the final tensor
    // we want to compute gradients with respect to (usually the loss).
    // This traverses the computational graph backwards and computes gradients for
    // all tensors that required gradients.
    let gradients = y.backward();

    // --- 4. Access the Gradients ---
    // Now we can get the gradients for 'a' and 'b' using the 'gradients' object.
    // .grad() returns an Option<Tensor>, so we use .unwrap() for simplicity here.
    let grad_a = a.grad(&gradients).unwrap();
    let grad_b = b.grad(&gradients).unwrap();

    println!("\nGradient of y with respect to 'a': {:}", grad_a.to_data());
    println!("Gradient of y with respect to 'b': {:}", grad_b.to_data());
    // Try to get gradient of 'c' (will panic if unwrapped, as it wasn't required_grad)
    // let grad_c = c.grad(&gradients).unwrap(); // This would panic!
    // println!("Gradient of y with respect to 'c': {:?}", grad_c.to_data());
}

// To run this example:
/*
fn main() {
    simple_autodiff_example();
}
*/
```

**Let's break down the math for our simple example:**

* `y = (a * b) + c`

* `a = 10.0`, `b = 5.0`, `c = 2.0`

* So, `y = (10.0 * 5.0) + 2.0 = 50.0 + 2.0 = 52.0`

Now, for the gradients:

* **Gradient of `y` with respect to `a` (`dy/da`):**

  * If `y = a * b + c`, then `dy/da = b`.

  * Since `b = 5.0`, we expect `dy/da = 5.0`.

* **Gradient of `y` with respect to `b` (`dy/db`):**

  * If `y = a * b + c`, then `dy/db = a`.

  * Since `a = 10.0`, we expect `dy/db = 10.0`.

Run the code, and you should see exactly these gradient values! Pretty cool, right? Burn automatically calculated these derivatives for us.

## Autodiff in Action: Simulating a Linear Neuron

Let's make this even more relevant to neural networks by simulating a single "linear neuron." This is the most basic building block of a neural network!

A linear neuron takes some inputs, multiplies them by weights, adds a bias, and produces an output. The goal is to learn the right weights and bias to minimize an error.

Here's the math: `output = (input * weight) + bias` (for a single input/output) or `output = input @ weights_matrix + bias_vector` (for multiple inputs/outputs).

For our example, let's keep it simple with a single input feature and a single output, and we'll define a `target` value to calculate a basic squared error loss.

```rust
use burn::tensor::{Tensor, backend::Backend, Autodiff}; // Removed TensorData as it's not directly used in this example
use burn::backend::NdArray;

type MyAutodiffBackend = Autodiff<NdArray>;

pub fn linear_neuron_autodiff_example() {
    let device = Default::default();

    // Update rule: new_param = old_param - learning_rate * gradient
    let learning_rate = 0.01f32; // A small step size

    // --- 1. Define our neuron's parameters (weights and bias) and input ---
    // Input feature (e.g., a single data point with one feature)
    // This is our 'x'
    let x = Tensor::<MyAutodiffBackend, 1>::from_data([2.0f32], &device);
    println!("Input 'x': {:}", x.to_data());

    // Weight (our learnable parameter)
    // This is our 'w'. We want to track its gradient!
    let mut w = Tensor::<MyAutodiffBackend, 1>::from_data([3.0f32], &device).require_grad();
    println!("Weight 'w': {:}", w.to_data());

    // Bias (another learnable parameter)
    // This is our 'b'. We also want to track its gradient!
    let mut b = Tensor::<MyAutodiffBackend, 1>::from_data([1.0], &device).require_grad();
    println!("Bias 'b': {:}", b.to_data());

    // Our target value (what we *want* the output to be)
    let target = Tensor::<MyAutodiffBackend, 1>::from_data([10.0], &device);
    println!("Target: {:}", target.to_data());

    for _ in 0..100 { // Loop for 100 optimization steps
        // --- 2. Perform the linear operation (forward pass) ---
        // y = (x * w) + b
        let y = (x.clone() * w.clone()) + b.clone();
        // println!("\nNeuron Output 'y': {:}", y.to_data()); // Commented out for cleaner loop output

        // --- 3. Calculate a simple Mean Squared Error (MSE) loss ---
        // Loss = (y - target)^2
        // Note: .mean() on a single-element tensor will still return a Rank 1 tensor with shape [1] in Burn 0.17.0
        let loss = (y - target.clone()).powf_scalar(2.0).mean();
        println!("Calculated Loss: {:}", loss.to_data());

        // --- 4. Perform the Backward Pass ---
        // Calculate gradients of the loss with respect to 'w' and 'b'
        let gradients = loss.backward();

        // --- 5. Access the Gradients ---
        let grad_w = w.grad(&gradients).unwrap();
        let grad_b = b.grad(&gradients).unwrap();

        // println!(
        //     "\nGradient of Loss with respect to 'w': {:}",
        //     grad_w.to_data()
        // ); // Commented out for cleaner loop output
        // println!(
        //     "Gradient of Loss with respect to 'b': {:}",
        //     grad_b.to_data()
        // ); // Commented out for cleaner loop output

        // --- 6. Apply Gradients (Simulating one step of Gradient Descent) ---
        // This is where we use the gradients to update our parameters!
        // Update rule: new_param = old_param - learning_rate * gradient
        //
        // Convert gradients from base backend (NdArray) to Autodiff backend
        let grad_w_autodiff =
            Tensor::<MyAutodiffBackend, 1>::from_data(grad_w.into_data(), &device);
        let grad_b_autodiff =
            Tensor::<MyAutodiffBackend, 1>::from_data(grad_b.into_data(), &device);

        // Update weight 'w'
        w = w
            .sub(grad_w_autodiff * learning_rate)
            .detach()
            .require_grad();
        // println!("\nUpdated Weight 'w': {:}", w.to_data()); // Commented out for cleaner loop output

        // Update bias 'b'
        b = b
            .sub(grad_b_autodiff * learning_rate)
            .detach()
            .require_grad();
        // println!("Updated Bias 'b': {:}", b.to_data()); // Commented out for cleaner loop output
    }
    println!("\n--- Training Loop Finished ---");
    println!("Final Weight 'w': {:}", w.to_data());
    println!("Final Bias 'b': {:}", b.to_data());
    // Recalculate final loss to show convergence
    let final_y = (x * w) + b;
    let final_loss = (final_y - target.clone()).powf_scalar(2.0).mean();
    println!("Final Loss: {:}", final_loss.to_data());

    // You should observe that the loss decreases with each step,
    // indicating that our parameters are converging towards the target!
}

// To run this example:
/*
fn main() {
    // simple_autodiff_example(); // You can uncomment this to run the first example
    linear_neuron_autodiff_example();
}
*/
```

## The Autodiff Flow: A Quick Mental Model

1.  **Tensor Creation:** When you create a `Tensor` using an `Autodiff` backend, it's like creating a node in a graph. If you call `.require_grad()`, that node gets a special flag.

2.  **Operations:** Each operation (like `*` or `+`) creates new tensors and new nodes in the graph, linking them to the tensors they were computed from. Burn remembers how to compute the derivative for each of these operations.

3.  **Backward Pass (`.backward()`):** When you call `.backward()` on a final tensor (e.g., your loss), Burn traverses this graph in reverse. It applies the chain rule of calculus at each node, accumulating the gradients until it reaches the tensors that were marked with `.require_grad()`.

4.  **Gradient Access (`.grad()`):** The `backward()` call returns a `Gradients` object. You use this object with `.grad()` on your original `required_grad()` tensors to retrieve their computed gradients.

This entire process is what allows optimizers to figure out how to adjust your model's weights and biases during training.

## What's Next?

We've now got our Tensors, and we understand the magic of Autodiff that lets us calculate gradients. These are the two foundational pillars!

You've seen how we manually set up `w` and `b` to `require_grad()` and then accessed their gradients. In a real neural network, you'd have many layers and thousands or millions of parameters! Manually managing all of them would be a nightmare.

In the next post, we'll see how Burn automates this process using **Modules** and **Configs** – Burn's way of organizing neural network layers and their parameters. This is where we start building actual network architectures efficiently!

Stay tuned for modular fun!

---

*Did Autodiff blow your mind a little? Got questions about gradients? Share your thoughts below! Let's learn and build AI/ML in Rust together!*
