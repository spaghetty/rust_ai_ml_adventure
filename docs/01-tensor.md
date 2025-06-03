# Title: Rust Meets AI: My Wild Ride with Burn and Tensors

![project image](./rust-burn-wide.jpg?raw=true)

As someone who loves learning new things, I often find myself exploring various fields. Lately, I've been diving into Rust, with AI and machine learning always on my mind. When I discovered a growing Rust AI ecosystem, it clicked – why not combine both? I'm no expert in either Rust or AI, so this is a learning journey. If you have any insights or corrections, please share them here or on my [GitHub](https://github.com/spaghetty/rust_ai_ml_adventure). Let's learn together\!

# Discovering the Booming Rust AI Ecosystem

While exploring Rust, I stumbled upon something fascinating: a rapidly expanding ecosystem of AI and machine learning crates. It’s more than just a few niche projects – there’s real momentum.

* **Polars**: Gaining significant traction as a high-performance DataFrame library, often cited as a potential Rust alternative to Pandas. (Polars today is often found alongside pandas in many big tech).
* **NdArray & Nalgebra**: Providing robust numerical computing foundations. These libraries are critical for handling the heavy lifting of tensor operations and linear algebra in AI.
* **RIG**: Emerging for building language model (LLM) integrations and agents. This points to Rust's growing capability in cutting-edge AI applications.
* **Burn**: A powerful, flexible deep learning framework built entirely in Rust. Its ability to swap backends without major code changes is a major advantage. (looking at github statistics it seems well active and well received).

This isn't just a personal Rust phase; it's recognizing a movement. This ecosystem’s growth made me think: “Why not dive in and contribute, while documenting the journey?” That's what I'm here to do.

# Burn: My Neural Network Playground

For AI, especially neural nets, **Burn** is key. It's a powerful Rust-native deep learning framework, offering flexibility and performance. Python's easy, but Burn? That's the wilder, more rewarding path.

Setting it up in your current dev env is quite easy at the start but you need to pay attention to features to add to your Cargo.toml

```bash
cargo add burn --features ndarray
```

# Diving into Tensors: The Core of It All

Okay, let's get technical for a sec. At the heart of AI/ML are **Tensors**. Think of them as super-powered arrays that hold all the data, images, model parameters, everything.

Before we dive into creating tensors, it's important to understand two key concepts: **Backends** and the **Device**.

* **Backend:** The backend determines how the tensor will be stored and how operations will be executed. You choose a backend (like \`NdArray\` for CPU operations) when you start. You can change it, but you need to select one initially.
* **Device:** The backend gives you a default device. The \`device\` tells Burn where your tensors should live and where computations happen (e.g., which CPU core or GPU).

```rust
// function receiving the backend selected
pub fn create_some_tensors<B: Backend>(device: &B::Device) {...}

fn main() {
    let device = Default::default(); // default device for NdArrayBackend
    println!("Using device: {:?}", device);
    create_some_tensors::<NdArray>(&device);
}
```

*For reference, you can check “`examples/basic_tensor.rs`”*

## Basic Tensor

In Burn, a tensor is Tensor\<B, D\>, where B is the Backend (like \`ndarray\`) and D is the dimension (1D for a vector, 2D for a matrix, etc.).

Creating tensors in Burn is where the magic happens. We can build them from Rust arrays and do all sorts of operations:

```rust
// 1D Tensor (Vector)
let vector_data = [1.0, 2.0, 3.0, 4.0];
let vector_tensor = Tensor::<B, 1>::from_data(vector_data, device);

// Example: Creating a 2D Tensor (Matrix)
let matrix_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
let matrix_tensor = Tensor::<B, 2>::from_data(matrix_data, device);

// 3D Tensor (Example)
let cube_data = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
let cube_tensor = Tensor::<B, 3>::from_data(cube_data, device);
```

Theoretically, tensors can have any dimension from 0 to N, but in reality, 0 dim tensors are not supported in Burn (yet). You can use scalars, and sometimes Burn tends to return size 1 tensor with a single element instead of a 0 dimension tensor.

## Shape vs. Rank

It's important to distinguish between the **shape** and the **rank** (or dimension) of a tensor:

```rust
println!("\n--- Demonstrating Tensor Shape ---\n");
let matrix_tensor = Tensor::<B, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device);
let matrix_shape = matrix_tensor.shape();

println!("Matrix Shape: {:?}", matrix_shape);  // Prints something like Shape { dims: [2, 3] }
```

Something important to focus here is that in the operation there is no dimension reduction. Burn keeps all the initial dimensions you need to squeeze or remove useless dimensions manually after operations.

# What's Next? Autodiff Magic!

We've set up our Rust environment, learned about Burn, and wrestled with tensors. This is the foundation. Next up, we're diving into **Automatic Differentiation (Autodiff)** – the secret sauce that makes neural networks learn.

Stay tuned for gradients, and let’s keep this Rust & AI adventure going! Got questions? Share your thoughts below – let's learn together.
