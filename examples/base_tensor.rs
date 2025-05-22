use burn::backend::NdArray;
use burn::prelude::Int;
use burn::tensor::{Tensor, backend::Backend}; // Our chosen backend

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
    // is very common to use clone function to reuse the Tensor more time,
    // there are no memory pressure with this approach
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
    let raw_scalar_value = 2.0; // Ensure the type matches the backend's FloatElem
    let scaled_vector = vector_tensor * raw_scalar_value; // Scalar multiplication
    println!(
        "\nVector Tensor * 2.0 (Using raw scalar):\n{}",
        scaled_vector.to_data()
    );
}

pub fn explore_shapes<B: Backend>(device: &B::Device) {
    println!("\n--- Demonstrating Tensor Shape ---\n");
    let matrix_tensor = Tensor::<B, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device);
    let matrix_shape = matrix_tensor.shape();

    println!("Matrix Tensor: {:}", matrix_tensor.to_data());
    println!("Matrix Shape: {:?}", matrix_shape); // Prints something like Shape { dims: [2, 3] }
    println!("Number of dimensions (Rank): {}", matrix_shape.num_dims()); // Prints 2
    println!("Size of dimension 0: {}", matrix_shape.dims[0]); // Prints 2
    println!("Size of dimension 1: {}", matrix_shape.dims[1]); // Prints 3
}

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

pub fn demonstrate_tricky_tensor_functions<B: Backend>(device: &B::Device) {
    println!("\n--- Demonstrating Tricky Tensor Functions ---\n");

    // --- Broadcasting ---
    // Adding tensors with compatible shapes
    // Burn automatically handles broadcasting rules similar to NumPy/PyTorch.
    // A smaller tensor's dimensions can be "stretched" to match a larger tensor's shape,
    // provided they are compatible (either the dimension size is 1 or the sizes match).

    let a = Tensor::<B, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], device); // Shape [2, 2]
    let b = Tensor::<B, 2>::from_data([[10.0, 20.0]], device); // Shape [1, 2] - Can broadcast across dim 0
    let c = a.clone() + b.clone();
    println!("Broadcasting A ([2, 2]) + B ([1, 2]) = C ({:?})", c.shape());
    println!("{}", c.to_data());
    // Expected output shape: [2, 2]
    // Internally, B [1, 2] is treated as [[10.0, 20.0], [10.0, 20.0]] for the operation.

    let d = Tensor::<B, 2>::from_data([[100.0], [200.0]], device); // Shape [2, 1] - Can broadcast across dim 1
    let e = a.clone() + d.clone();
    println!("Broadcasting A ([2, 2]) + D ([2, 1]) = E ({:?})", e.shape());
    println!("{}", e.to_data());
    // Expected output shape: [2, 2]
    // Internally, D [2, 1] is treated as [[100.0, 100.0], [200.0, 200.0]] for the operation.

    // --- Reductions with keepdim ---
    // Reduction operations like sum() or mean() collapse dimensions.
    // 'keepdim' determines if the collapsed dimension is kept with size 1 (true)
    // or completely removed (false).

    let matrix = Tensor::<B, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device); // Shape [2, 3]
    println!(
        "\nOriginal matrix for reductions (Shape {:?}):",
        matrix.shape()
    );
    println!("{}", matrix.to_data());

    // Sum along dimension 1 (columns: 2.0, 5.0), keeping dimension (AI is wrong here)
    let sum_dim1_keepdim = matrix.clone().sum_dim(1); // Result shape [2, 1]
    println!(
        "\nSum along dim 1 with keepdim (Shape {:?}):",
        sum_dim1_keepdim.shape()
    );
    println!("{}", sum_dim1_keepdim.to_data());
    // Expected output shape: [2, 1] -> [[6.0], [15.0]]

    // Sum along dimension 1 (columns: 2.0, 5.0), without keeping dimension (AI is wrong here)
    let sum_dim1_no_keepdim: Tensor<B, 1> = matrix.clone().sum_dim(1).squeeze(1); // Result shape [2]
    println!(
        "Sum along dim 1 without keepdim (Shape {:?}):",
        sum_dim1_no_keepdim.shape()
    );
    println!("{}", sum_dim1_no_keepdim.to_data());
    // Expected output shape: [2] -> [6.0, 15.0]

    // Mean along dimension 0 (rows: 1.0, 4.0), keeping dimension
    let mean_dim0_keepdim = matrix.clone().mean_dim(0); // Result shape [1, 3]
    println!(
        "\nMean along dim 0 with keepdim (Shape {:?}):",
        mean_dim0_keepdim.shape()
    );
    println!("{}", mean_dim0_keepdim.to_data());
    // Expected output shape: [1, 3] -> [[2.5, 3.5, 4.5]]

    // --- Indexing/Slicing ---
    // `slice()` allows selecting a sub-tensor using ranges for specific dimensions.
    // `select()` selects a single index along a specific dimension, reducing its rank.

    let large_tensor = Tensor::<B, 3>::from_data(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ], // Shape [3, 2, 2]
        device,
    );
    println!(
        "\nOriginal large tensor for slicing (Shape {:?}):",
        large_tensor.shape()
    );
    println!("{}", large_tensor.to_data());

    // Slicing along the first dimension (get the middle slice - index 1)
    // slice([dim_index, start..end])
    let sliced_tensor = large_tensor.clone().slice([1..2]); // Shape [1, 2, 2]
    println!(
        "Sliced tensor (slice([1..2]), Shape {:?}):",
        sliced_tensor.shape()
    );
    println!("{}", sliced_tensor.to_data());
    // Expected output shape: [1, 2, 2] -> [[[5.0, 6.0], [7.0, 8.0]]]

    // Slicing along the second dimension (get the first element along dim 1 for all batches and dim 2)
    let sliced_rows = large_tensor.clone().slice([0..3, 0..1, 0..2]); // Shape [3, 1, 2]
    println!(
        "Sliced rows (slice([0..3, 0..1, 0..2]), Shape {:?}):",
        sliced_rows.shape()
    );
    println!("{}", sliced_rows.to_data());
    // Expected output shape: [3, 1, 2] -> [[[1.0, 2.0]], [[5.0, 6.0]], [[9.0, 10.0]]]

    // Using select to get a single element along a dimension, reducing rank
    let indexes = Tensor::<B, 1, Int>::from_data([1], device);
    // Select index 1 along dimension 0. Shape becomes [2, 2]
    let selected_slice: Tensor<B, 2> = large_tensor.clone().select(0, indexes).squeeze(0);
    println!(
        "\nSelected slice (select(0, 1), Shape {:?}):",
        selected_slice.shape()
    );
    println!("{}", selected_slice.to_data());
    // Expected output shape: [2, 2] -> [[5.0, 6.0], [7.0, 8.0]]

    // --- Reshaping/Flattening/Unflattening ---
    // These operations change the shape of the tensor without changing its data.

    let reshape_tensor = Tensor::<B, 3>::from_data(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], // Shape [2, 2, 2]
        device,
    );
    println!("\nOriginal reshape tensor ({:?}):", reshape_tensor.shape());
    println!("{}", reshape_tensor.to_data());

    // Flatten the tensor: combines a range of dimensions into a single dimension.
    // Flatten dimensions 0, 1, and 2 into one. New dim will be 2*2*2=8
    let flattened_tensor: Tensor<B, 1> = reshape_tensor.clone().flatten(0, 2);
    println!(
        "Flattened tensor (flatten(0, 2), {:?}):",
        flattened_tensor.shape()
    ); // Shape [8]
    println!("{}", flattened_tensor.to_data());
    // Expected output shape: [8] -> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    // Reshape to a different shape: must contain the same total number of elements.
    let reshaped_tensor = reshape_tensor.clone().reshape([4, 2]); // Shape [4, 2]. Total elements 4*2=8 (matches 2*2*2)
    println!(
        "Reshaped tensor (reshape([4, 2]),  {:?}):",
        reshaped_tensor.shape()
    );
    println!("{}", reshaped_tensor.to_data());
    // Expected output shape: [4, 2] -> [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]

    // Unflatten: opposite of flatten, splits a dimension into multiple dimensions.
    // Unflatten dimension 0 (size 4) into two dimensions [2, 2]. Shape becomes [2, 2, 2]
    let unflattened_tensor = reshaped_tensor.reshape([2, 2, 2]);
    println!(
        "Reshaped tensor (reshape([2, 2, 2]), {:?}):",
        unflattened_tensor.shape()
    );
    println!("{}", unflattened_tensor.to_data());
    // Expected output shape: [2, 2, 2] -> [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
}

fn main() {
    let device = Default::default(); // Get the default device for NdArrayBackend
    println!("Using device: {:?}", device);
    create_some_tensors::<NdArray>(&device);
    explore_shapes::<NdArray>(&device);
    basic_tensor_ops::<NdArray>(&device);
    demonstrate_tricky_tensor_functions::<NdArray>(&device);
}
