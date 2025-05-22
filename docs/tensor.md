# Post 2: AI and ML in Rust: Wrestling with Tensors (It's Not That Scary!) - My Journey
Alright team, welcome back to the adventure! In our first post, we got our Rust environment set up and talked about why we're brave (or maybe just a little bit crazy!) enough to tackle AI/ML in __Rust__ with __Burn__.

Now that we've got our base camp ready, it's time to meet the absolute core concept we'll be working with: the Tensor. If you've seen any deep learning code before, you've seen tensors. They're everywhere!

Think of a tensor as a fancy container for numbers. It's like an array, but super-powered for doing all the mathematical heavy lifting that AI/ML requires.

## What Exactly IS a Tensor? (Beyond Just a Bunch of Numbers)
At its heart, a tensor is a mathematical object that can be represented as a multi-dimensional array.

* A number is a 0-dimensional tensor (a scalar).
* A list of numbers is a 1-dimensional tensor (a vector).
* A grid of numbers is a 2-dimensional tensor (a matrix).

And it keeps going! A cube of numbers is a 3-dimensional tensor, and so on.

In AI/ML, tensors are how we represent everything:

* Input data (like images, text, audio).
* Model parameters (like the weights and biases in a neural network).
* The intermediate results of calculations as data flows through a model.

Tensors are designed to be efficient for the types of operations common in AI/ML, especially when running on specialized hardware like GPUs.

## Tensors in Burn: Tensor<B, D>
In __Burn__, the tensor is represented by the ```Tensor<B, D>``` type. Let's break down that funky looking syntax:

* ```Tensor```: Okay, that part's easy!
* ```<B: Backend>```: This tells us that the tensor is tied to a specific __Backend__ (```B```). Remember backends from Post 1? This is where that comes into play. The backend determines how the tensor's data is stored and how the operations on it are performed (e.g., on the CPU using ```ndarray```, or on the GPU using ```wgpu```). The ```B``` is a generic type parameter that implements the ```Backend``` trait.
* ```<D: Dimension>```: This tells us the rank (number of dimensions) of the tensor. ```D``` is a generic const parameter (just use numbers like ```1```, ```2```, ```3```, etc.). ```Tensor<B, 1>``` is a vector, ```Tensor<B, 2>``` is a matrix, and so on.

So, ```Tensor<MyBackend, 2>``` would be a 2-dimensional tensor using your chosen ```MyBackend```.

## Creating Tensors: Bringing Data to Life!
Okay, now we can start building tensors and operating with them using ```Tensor::from_data```.

Important Note for Scalars (0D Tensors): Creating a 0-dimensional tensor (a scalar) directly from a raw Rust number literal in Burn 0.17.0 seems impossble. My understanding is that the problem got solved in two different way:

* you can use scalar directly in operations
* you get 1D Tensor as result for operation that are suppose to return a scalar value

In practice, you often get scalar values as the result of operations that reduce a tensor down to a single value (like taking the mean or sum of all its elements). However, even these operations return a Rank 1 tensor with a single element ([1]) in Burn, rather than a true Rank 0 scalar ([]). This is a subtle detail to be aware of!

Let's try creating a few different tensors using the ndarray backend we set up in Post 1, focusing on how to get scalar values from operations:

```rust filename="examples/base_tensor.rs"
use burn::backend::NdArray;
use burn::tensor::{Tensor, backend::Backend};

pub fn create_some_tensors<B: Backend>(device: &B::Device) {
    // --- Creating Higher Rank Tensors (Works directly with arrays) ---
    // 1D Tensor (Vector)
    let vector_data = [1.0, 2.0, 3.0, 4.0];
    let vector_tensor = Tensor::<B, 1>::from_data(vector_data, device);
    println!(
        "Vector Tensor ({:?}) {}",
        vector_tensor.shape(),
        vector_tensor.to_data()
    );

    // 2D Tensor (Matrix)
    let matrix_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let matrix_tensor = Tensor::<B, 2>::from_data(matrix_data, device);
    println!(
        "Matrix Tensor ({:?}): {}",
        matrix_tensor.shape(),
        matrix_tensor.to_data()
    );

    // 3D Tensor (Example)
    let cube_data = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    let cube_tensor = Tensor::<B, 3>::from_data(cube_data, device);
    println!(
        "Cube Tensor ({:?}): {}",
        cube_tensor.shape(),
        cube_tensor.to_data()
    );

    // --- Getting Scalar Values from Operations (Often as Rank 1 Tensors) ---
    // Example: Taking the mean of a tensor
    // Note: In Burn 0.17.0, mean() without an axis returns a Rank 1 tensor with shape [1]
    let mean_scalar_result: Tensor<B, 1> = matrix_tensor.mean();
    println!(
        "\nMean of Matrix Tensor (Result Rank 1, {:?}): {}",
        mean_scalar_result.shape(),
        mean_scalar_result.to_data()
    );

    // Example: Taking the sum of a tensor
    // Note: Sum() without an axis also returns a Rank 1 tensor with shape [1]
    // consider that within Burn give the restriction of rust with the variable ownership
    // is very common to use clone function to reuse the Tensor many time,
    // there is no memory pressure with this approach
    // ([Reference]<https://burn.dev/burn-book/building-blocks/tensor.html#ownership-and-cloning>)
    let sum_scalar_result: Tensor<B, 1> = vector_tensor.clone().sum();
    println!(
        "Sum of Vector Tensor (Result Rank 1, {:?}): {}",
        sum_scalar_result.shape(),
        sum_scalar_result.to_data()
    );

    // Example: A single value resulting from a calculation
    // The result of operations on Rank 1 tensors with shape [1] is also a Rank 1 tensor [1]
    let single_value_result: Tensor<B, 1> =
        (mean_scalar_result.clone() + sum_scalar_result.clone()).sqrt();
    println!(
        "Result of Calculation (Result Rank 1, {:?}): {}",
        single_value_result.shape(),
        single_value_result.to_data()
    );

    // --- Using Raw Scalar Values in Operations ---
    // You CAN use raw Rust scalar values directly in some operations (like element-wise)
    let raw_scalar_value = 2.0;
    let scaled_vector = vector_tensor * raw_scalar_value; // Scalar multiplication
    println!(
        "\nVector Tensor * 2.0 (Using raw scalar):\n{}",
        scaled_vector.to_data()
    );

}

fn main() {
    let device = Default::default(); // Get the default device for NdArrayBackend
    println!("Using device: {:?}", device);
    create_some_tensors::<NdArray>(&device);
}

```


## Understanding Shape and Dimensions
Every tensor has a shape. The shape is a list (or array) of integers that tells you the size of the tensor along each dimension.

A scalar (Rank 0) has an empty shape [].

A vector of size 4 (Rank 1) has a shape [4].

A matrix with 2 rows and 3 columns (Rank 2) has a shape [2, 3].

A 3D tensor with shape [2, 2, 2] means it has size 2 along the first dimension, size 2 along the second, and size 2 along the third.

```rust
// ... (previous use statements)

pub fn explore_shapes<B: Backend>(device: &B::Device) {
    let matrix_tensor = Tensor::<B, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device);
    let matrix_shape = matrix_tensor.shape();

    println!("Matrix Tensor: {:}", matrix_tensor.to_data());
    println!("Matrix Shape: {:?}", matrix_shape); // Prints something like Shape { dims: [2, 3] }
    println!("Number of dimensions (Rank): {}", matrix_shape.num_dims()); // Prints 2
    println!("Size of dimension 0: {}", matrix_shape.dims[0]); // Prints 2
    println!("Size of dimension 1: {}", matrix_shape.dims[1]); // Prints 3
}

```

Understanding tensor shapes is absolutely crucial in AI/ML. Mismatched shapes are a common source of errors!

## Basic Tensor Operations: Doing Math!
Tensors wouldn't be very useful if you couldn't do math with them! Burn's Tensor type provides methods for common operations. These operations are typically element-wise or linear algebra operations.

```rust
// ... (previous use statements)

pub fn basic_tensor_ops<B: Backend>(device: &B::Device) {
    println!("\n--- Demonstrating Basic Tensor Functions ---\n");
    let tensor1 = Tensor::<B, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], device);
    let tensor2 = Tensor::<B, 2>::from_data([[5.0, 6.0], [7.0, 8.0]], device);

    // --- Element-wise Addition ---
    let sum_tensor = tensor1.clone() + tensor2.clone(); // Use clone if you need the original tensors later
    println!("\nTensor 1:\n{:}", tensor1.to_data());
    println!("Tensor 2:\n{:}", tensor2.to_data());
    println!(
        "Tensor 1 + Tensor 2 (Element-wise Addition):\n{:}",
        sum_tensor.to_data()
    );

    // --- Element-wise Multiplication ---
    let product_tensor = tensor1.clone() * tensor2.clone();
    println!(
        "Tensor 1 * Tensor 2 (Element-wise Multiplication):\n{:}",
        product_tensor.to_data()
    );

    // --- Matrix Multiplication (Requires compatible shapes!) ---
    // For matmul(A, B), the number of columns in A must equal the number of rows in B.
    // If tensor1 is [2, 2] and tensor2 is [2, 2], matmul is possible.
    let matmul_result = tensor1.clone().matmul(tensor2);
    println!(
        "Tensor 1 @ Tensor 2 (Matrix Multiplication):\n{:}",
        matmul_result.to_data()
    );

    // --- Scalar Operations ---
    // Ensure the scalar literal type matches the backend's float type (e.g., f32)
    let scalar = 2.0f32;
    let scaled_tensor = tensor1 * scalar; // Element-wise multiplication by a scalar
    println!(
        "Tensor 1 * 2.0 (Scalar Multiplication):\n{:}",
        scaled_tensor.to_data()
    );
}

*/
```

Burn provides a rich set of methods for various tensor operations (addition, subtraction, multiplication, division, matrix multiplication, transpose, reshape, activation functions like sigmoid, relu, etc.). We'll explore more as we build models.

Backends and Devices (A Quick Recap)
Remember how Tensor is Tensor<B, D>? This B (the Backend) is important because it determines where the tensor's data lives and how the operations are actually computed.

When you create a tensor using Tensor::from_data(data, device), you're telling Burn which device (provided by the Backend) the tensor should be placed on. Operations between tensors usually require them to be on the same device.

For now, we're using NdArray (CPU), so our tensors live in CPU memory. Later, when we use GPU backends, our tensors will live in GPU memory, and operations will be executed by the GPU.

## What's Next?
We've met the Tensor, learned how to create them, understand their shape, and do some basic math. This is the absolute bedrock!

In the next post, we'll start putting these tensors to work in a more structured way by introducing Automatic Differentiation (Autodiff) – the magic sauce that allows neural networks to learn!

Stay tuned for gradients!

Got questions about tensors? Finding their shapes confusing? Share your thoughts below! Let's learn and build AI/ML in Rust together!
