use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData, DType,Shape, Distribution};
use burn::prelude::Int;
type MyBackend = NdArray;
const DEFAULT_TOLERANCE: f32 = 1e-6;

// 1. Basics of Tensors
#[test]
fn test_tensor_creation() {
    let device = Default::default();
    
    // Test 1D tensor creation
    let vector_data = [1.0, 2.0, 3.0, 4.0];
    let vector_tensor = Tensor::<MyBackend, 1>::from_data(vector_data, &device);
    assert_eq!(vector_tensor.shape().dims, [4]);
    assert_eq!(vector_tensor.to_data(), TensorData::from(vector_data).convert_dtype(DType::F32));

    // Test 2D tensor creation
    let matrix_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let matrix_tensor = Tensor::<MyBackend, 2>::from_data(matrix_data, &device);
    assert_eq!(matrix_tensor.shape().dims, [2, 3]);
    assert_eq!(matrix_tensor.to_data(), TensorData::from(matrix_data).convert_dtype(DType::F32));

    // Test 3D tensor creation
    let cube_data = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    let cube_tensor = Tensor::<MyBackend, 3>::from_data(cube_data, &device);
    assert_eq!(cube_tensor.shape().dims, [2, 2, 2]);
    assert_eq!(cube_tensor.to_data(), TensorData::from(cube_data).convert_dtype(DType::F32));
}

#[test]
fn test_tensor_shapes() {
    let device = Default::default();
    
    let matrix_tensor = Tensor::<MyBackend, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let shape = matrix_tensor.shape();
    
    assert_eq!(shape.num_dims(), 2);
    assert_eq!(shape.dims[0], 2);
    assert_eq!(shape.dims[1], 3);
}

#[test]
fn test_tensor_type_conversion() {
    let device =  Default::default();

    let float_tensor = Tensor::<MyBackend, 1>::from_data([1.0, 2.0, 3.0, 4.0], &device);
    let int_tensor = float_tensor.clone().int();
    
    assert_eq!(int_tensor.to_data(), TensorData::from([1, 2, 3, 4]).convert_dtype(DType::I64));
}

// 2. Basic Operations
#[test]
fn test_tensor_operations() {
    let device = Default::default();
    
    // Test element-wise addition
    let tensor1 = Tensor::<MyBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let tensor2 = Tensor::<MyBackend, 2>::from_data([[5.0, 6.0], [7.0, 8.0]], &device);
    let result = tensor1.clone() + tensor2.clone();
    assert_eq!(result.to_data(), TensorData::from([[6.0, 8.0], [10.0, 12.0]]).convert_dtype(DType::F32));

    // Test scalar multiplication
    let scalar = 2.0;
    let scaled = tensor1.clone() * scalar;
    assert_eq!(scaled.to_data(), TensorData::from([[2.0, 4.0], [6.0, 8.0]]).convert_dtype(DType::F32));
}

#[test]
fn test_tensor_reduction() {
    let device = Default::default();
    
    let matrix_tensor = Tensor::<MyBackend, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    
    // Test mean expected to be 3.5
    let mean = matrix_tensor.clone().mean();
    assert!((mean.into_scalar() - 3.5).abs() <  DEFAULT_TOLERANCE);

    // Test sum expected to be 21.0
    let sum = matrix_tensor.clone().sum();
    assert_eq!(sum.into_scalar(), 21.0);
}



// 3. Data Manipulation
#[test]
fn test_tensor_slicing() {
    let device = Default::default();
    
    // Test 2D tensor slicing
    let tensor = Tensor::<MyBackend, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    
    // Test row slicing
    // Test row slicing
    let first_row = tensor.clone().slice([0..1, 0..3]);
    assert_eq!(first_row.to_data(), TensorData::from([[1.0, 2.0, 3.0]]).convert_dtype(DType::F32));
    
    // Test column slicing
    let first_two_cols = tensor.clone().slice([0..2, 0..2]);
    assert_eq!(first_two_cols.to_data(), TensorData::from([[1.0, 2.0], [4.0, 5.0]]).convert_dtype(DType::F32));
    
    // Test full row selection
    let all_rows = tensor.slice([0..2, 1..2]);
    assert_eq!(all_rows.to_data(), TensorData::from([[2.0], [5.0]]).convert_dtype(DType::F32));
}

#[test]
fn test_tensor_concatenation() {
    let device = Default::default();
    
    // Test 2D tensor concatenation along different dimensions
    let tensor1 = Tensor::<MyBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let tensor2 = Tensor::<MyBackend, 2>::from_data([[5.0, 6.0], [7.0, 8.0]], &device);
    
    // Concatenate along dimension 0 (rows)
    let concat_rows = Tensor::cat(vec![tensor1.clone(), tensor2.clone()], 0);
    assert_eq!(concat_rows.to_data(), TensorData::from([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]).convert_dtype(DType::F32));
    
    // Concatenate along dimension 1 (columns)
    let concat_cols = Tensor::cat(vec![tensor1.clone(), tensor2.clone()], 1);
    assert_eq!(concat_cols.to_data(), TensorData::from([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]]).convert_dtype(DType::F32));
    
    // Test 1D tensor concatenation
    let tensor3 = Tensor::<MyBackend, 1>::from_data([9.0, 10.0], &device);
    let tensor4 = Tensor::<MyBackend, 1>::from_data([11.0, 12.0], &device);
    let concat_1d = Tensor::cat(vec![tensor3, tensor4], 0);
    assert_eq!(concat_1d.to_data(), TensorData::from([9.0, 10.0, 11.0, 12.0]).convert_dtype(DType::F32));
}

#[test]
fn test_tensor_elementwise_operations() {
    let device = Default::default();
    
    let tensor1 = Tensor::<MyBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let tensor2 = Tensor::<MyBackend, 2>::from_data([[5.0, 6.0], [7.0, 8.0]], &device);
    let tensor_sub = Tensor::<MyBackend, 2>::from_data([[-4.0, -4.0], [-4.0, -4.0]], &device);
    let tensor_div = Tensor::<MyBackend, 2>::from_data([[0.2, 0.33333334], [0.42857143, 0.5]], &device);
    let tensor_pow = Tensor::<MyBackend, 2>::from_data([[1.0, 4.0], [27.0, 256.0]], &device);
    let tensor_abs = Tensor::<MyBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    
    
    // Test subtraction
    let sub = tensor1.clone() - tensor2.clone();

    assert_eq!((sub - tensor_sub).abs().sum().into_scalar(), 0.0);
    
    // Test division
    let div = tensor1.clone() / tensor2.clone();
    assert_eq!((div - tensor_div).abs().sum().into_scalar(), 0.0);
    
    // Test power operation [[1^1, 2^2], [3^3, 4^4]]
    let pow = tensor1.clone().powf(tensor1.clone());
    assert_eq!((pow - tensor_pow).abs().sum().into_scalar(), 0.0);
    
    // Test absolute value
    let neg_tensor = Tensor::<MyBackend, 2>::from_data([[-1.0, -2.0], [-3.0, -4.0]], &device);
    let abs_tensor = neg_tensor.abs();
    assert_eq!((abs_tensor - tensor_abs).abs().sum().into_scalar(), 0.0);
}

#[test]
fn test_tensor_reduction_operations() {
    let device = Default::default();
    
    let tensor = Tensor::<MyBackend, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    
    // Test min
    let min = tensor.clone().min();
    assert_eq!(min.into_scalar(), 1.0);
    
    // Test max
    let max = tensor.clone().max();
    assert_eq!(max.into_scalar(), 6.0);
    
    // Test argmax keep in mind that does not reduce dimensions
    let argmax = tensor.clone().argmax(0);
    println!("argmax: {:}", argmax.to_data());
    assert_eq!(argmax.to_data(), TensorData::from([[1, 1, 1]]).convert_dtype(DType::I64));


    // Here an example in out to reduce dimensions after operation that should reduce dimensions
    let reduced_argmax: Tensor<MyBackend, 1, Int> = argmax.squeeze(0);
    assert_eq!(reduced_argmax.to_data(), TensorData::from([1, 1, 1]).convert_dtype(DType::I64));
    
    // Test argmin
    let argmin = tensor.argmin(0);
    assert_eq!(argmin.to_data(), TensorData::from([[0, 0, 0]]).convert_dtype(DType::I64));
    
    // Here an example in out to reduce dimensions after operation that should reduce dimensions
    let reduced_argmin: Tensor<MyBackend, 1, Int> = argmin.squeeze(0);
    assert_eq!(reduced_argmin.to_data(), TensorData::from([0, 0, 0]).convert_dtype(DType::I64));    
}

#[test]
fn test_tensor_broadcasting() {
    let device = Default::default();
    
    // Test broadcasting addition
    let tensor1 = Tensor::<MyBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let tensor2 = Tensor::<MyBackend, 2>::from_data([[5.0, 6.0]], &device);
    let expected = Tensor::<MyBackend, 2>::from_data([[6.0, 8.0], [8.0, 10.0]], &device);
    
    let result = tensor1.clone() + tensor2;
    assert_eq!((result - expected).abs().sum().into_scalar(), 0.0);
    
    // Test broadcasting multiplication
    let tensor2 = Tensor::<MyBackend, 2>::from_data([[2.0, 3.0]], &device);
    let expected = Tensor::<MyBackend, 2>::from_data([[2.0, 6.0], [6.0, 12.0]], &device);
    
    let result = tensor1.clone() * tensor2;
    assert_eq!((result - expected).abs().sum().into_scalar(), 0.0);
    
    // Test broadcasting with scalar
    let result = tensor1.clone() * 2.0;
    let expected = Tensor::<MyBackend, 2>::from_data([[2.0, 4.0], [6.0, 8.0]], &device);
    assert_eq!((result - expected).abs().sum().into_scalar(), 0.0);
}

#[test]
fn test_tensor_permutations() {
    let device = Default::default();
    
    // Test transpose
    let tensor = Tensor::<MyBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let transposed = tensor.transpose();
    assert_eq!(transposed.to_data(), TensorData::from([[1.0, 3.0], [2.0, 4.0]]).convert_dtype(DType::F32));
    
    // Test permute
    let tensor3d = Tensor::<MyBackend, 3>::from_data(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        &device
    );
    let permuted = tensor3d.permute([2, 0, 1]);
    assert_eq!(permuted.to_data(), TensorData::from(
        [[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]]
    ).convert_dtype(DType::F32));
    
    // Test reshape
    let tensor = Tensor::<MyBackend, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let reshaped = tensor.reshape([6, 1]);
    assert_eq!(reshaped.to_data(), TensorData::from(
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    ).convert_dtype(DType::F32));
}

#[test]
fn test_tensor_comparison() {
    let device = Default::default();
    
    let tensor1 = Tensor::<MyBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let tensor2 = Tensor::<MyBackend, 2>::from_data([[2.0, 2.0], [3.0, 5.0]], &device);
    
    // Test equality comparison
    let eq = tensor1.clone().equal(tensor2.clone());
    assert_eq!(eq.clone().to_data(), TensorData::from([[false, true], [true, false]]).convert_dtype(DType::Bool));
    
    // Test greater than comparison
    let gt = tensor1.clone().greater(tensor2.clone());
    assert_eq!(gt.to_data(), TensorData::from([[false, false], [false, false]]).convert_dtype(DType::Bool));
    
    // Test less than comparison
    let lt = tensor1.clone().lower(tensor2.clone());
    assert_eq!(lt.clone().to_data(), TensorData::from([[true, false], [false, true]]).convert_dtype(DType::Bool));
    
    // Test all() reduction
    let all = eq.clone().all();
    assert_eq!(all.clone().into_scalar(), false);
    
    // Test any() reduction
    let any = eq.clone().any();
    assert_eq!(any.clone().into_scalar(), true);
}

#[test]
fn test_tensor_properties() {
    let device = Default::default();
    
    let tensor = Tensor::<MyBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    
    // Test dtype
    assert_eq!(tensor.dtype(), DType::F32);
    
    // Test device
    assert_eq!(tensor.device(), device);
    
    // Test shape
    assert_eq!(tensor.shape().dims, [2, 2]);
    
    // Test num_elements
    assert_eq!(tensor.shape().num_elements(), 4);
    
    // Test is_contiguous concept not found in burn
    // assert!(tensor.is_contiguous());
}

#[test]
fn test_tensor_initialization() {
    let device = Default::default();
    
    // Test zeros initialization
    let zeros = Tensor::<MyBackend, 2>::zeros([2, 3], &device);
    assert_eq!(zeros.to_data(), TensorData::from([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).convert_dtype(DType::F32));
    
    // Test ones initialization
    let ones = Tensor::<MyBackend, 2>::ones([2, 3], &device);
    assert_eq!(ones.to_data(), TensorData::from([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).convert_dtype(DType::F32));
    
    // Test random initialization
    let distribution = Distribution::Uniform(0.0, 1.0);
    let random = Tensor::<MyBackend, 2>::random(Shape::new([2, 3]), distribution, &device);
    assert_eq!(random.shape().dims, [2, 3]);
    
    // Test tensor cloning
    let cloned = random.clone();
    assert_eq!(cloned.to_data(), random.to_data());
    
    // Test tensor copying: TODO
    //let copied = random.clone();
    //assert_eq!(copied.to_data(), random.to_data());
}

#[test]
fn test_tensor_math_operations() {
    let device = Default::default();
    
    let tensor = Tensor::<MyBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    
    // Test trigonometric functions
    let sin = tensor.clone().sin();
    let expected_sin = Tensor::<MyBackend, 2>::from_data([
        [0.84147098, 0.90929743],
        [0.14112001, -0.7568025]
    ], &device);
    assert_eq!((sin - expected_sin).abs().sum().into_scalar() < DEFAULT_TOLERANCE, true);
    
    // Test exponential function
    let exp = tensor.clone().exp();
    let expected_exp = Tensor::<MyBackend, 2>::from_data([
        [2.71828183, 7.3890561],
        [20.08553692, 54.59815003]
    ], &device);
    assert_eq!((exp - expected_exp).abs().sum().into_scalar() < DEFAULT_TOLERANCE, true);
    
    // Test logarithm
    let log = tensor.clone().log();
    let expected_log = Tensor::<MyBackend, 2>::from_data([
        [0.0, 0.69314718],
        [1.09861229, 1.38629436]
    ], &device);
    assert_eq!((log - expected_log).abs().sum().into_scalar() < DEFAULT_TOLERANCE, true);
    
    // Test rounding operations
    let tensor_round = Tensor::<MyBackend, 2>::from_data([
        [1.2, 2.8],
        [3.5, 4.3]
    ], &device);
    
    let rounded = tensor_round.round();
    assert_eq!(rounded.to_data(), TensorData::from([
        [1.0, 3.0],
        [4.0, 4.0]
    ]).convert_dtype(DType::F32));
}
