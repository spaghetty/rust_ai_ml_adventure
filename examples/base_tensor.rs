use burn::backend::NdArray;
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
    // there are no memory pressure with this approch ([Reference]<https://burn.dev/burn-book/building-blocks/tensor.html#ownership-and-cloning>)
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
    let raw_scalar_value = 2.0f32; // Ensure the type matches the backend's FloatElem
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
