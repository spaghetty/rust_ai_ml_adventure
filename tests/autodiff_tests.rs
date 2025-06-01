use burn::backend::{Autodiff, NdArray};
use burn::tensor::{Tensor, TensorData, DType};
type MyAutodiffBackend = Autodiff<NdArray>;
const DEFAULT_TOLERANCE: f32 = 1e-6;

// 1. Basic Autodiff Concepts
#[test]
fn test_autodiff_tensor_creation() {
    let device = Default::default();
    
    // Test tensor requiring gradients
    let a = Tensor::<MyAutodiffBackend, 1>::from_data([10.0], &device).require_grad();
    assert!(a.is_require_grad());
    
    // Test tensor without gradients
    let b = Tensor::<MyAutodiffBackend, 1>::from_data([5.0], &device);
    assert!(!b.is_require_grad());
    
    // Test converting non-grad tensor to require gradients
    let c = b.require_grad();
    assert!(c.is_require_grad());
}

#[test]
fn test_autodiff_computation_graph() {
    let device = Default::default();
    
    // Create tensors
    let a = Tensor::<MyAutodiffBackend, 1>::from_data([2.0], &device).require_grad();
    let b = Tensor::<MyAutodiffBackend, 1>::from_data([3.0], &device).require_grad();
    
    // Simple computation: y = a * b
    let y = a.clone() * b.clone();
    
    // Test that y is part of the computation graph
    assert!(!y.is_require_grad(), "y shouldn't be part of the computation graph");
    
    // Get gradients
    let gradients = y.backward();
    
    // Test gradients
    let grad_a = a.grad(&gradients).unwrap();
    let grad_b = b.grad(&gradients).unwrap();
    
    // Since y = a * b, gradients should be:
    // da/dy = b = 3.0
    // db/dy = a = 2.0
    assert_eq!(grad_a.to_data(), TensorData::from([3.0]).convert_dtype(DType::F32));
    assert_eq!(grad_b.to_data(), TensorData::from([2.0]).convert_dtype(DType::F32));
}

// 2. Autodiff with Multiple Operations
#[test]
fn test_autodiff_multiple_operations() {
    let device = Default::default();
    
    // Create tensors
    let a = Tensor::<MyAutodiffBackend, 1>::from_data([2.0], &device).require_grad();
    let b = Tensor::<MyAutodiffBackend, 1>::from_data([3.0], &device).require_grad();
    let c = Tensor::<MyAutodiffBackend, 1>::from_data([4.0], &device).require_grad();
    
    // More complex computation: y = (a * b) + (a * c)
    let y = (a.clone() * b.clone()) + (a.clone() * c.clone());
    
    // Get gradients
    let gradients = y.backward();
    
    // Test gradients
    let grad_a = a.grad(&gradients).unwrap();
    let grad_b = b.grad(&gradients).unwrap();
    let grad_c = c.grad(&gradients).unwrap();
    
    // Since y = (a * b) + (a * c), gradients should be:
    // da/dy = b + c = 3.0 + 4.0 = 7.0
    // db/dy = a = 2.0
    // dc/dy = a = 2.0
    assert_eq!(grad_a.to_data(), TensorData::from([7.0]).convert_dtype(DType::F32));
    assert_eq!(grad_b.to_data(), TensorData::from([2.0]).convert_dtype(DType::F32));
    assert_eq!(grad_c.to_data(), TensorData::from([2.0]).convert_dtype(DType::F32));
}

// 3. Autodiff with Reduction Operations
#[test]
fn test_autodiff_reduction_operations() {
    let device = Default::default();
    
    // Create tensors
    let a = Tensor::<MyAutodiffBackend, 1>::from_data([1.0, 2.0, 3.0], &device).require_grad();
    
    // Mean reduction operation
    let mean = a.clone().mean();
    
    // Get gradients
    let gradients = mean.backward();
    
    // Test gradients
    let grad_a = a.grad(&gradients).unwrap();
    
    // Since mean = (a1 + a2 + a3) / 3, gradient should be 1/3 for each element
    assert_eq!(grad_a.to_data(), TensorData::from([1.0/3.0, 1.0/3.0, 1.0/3.0]).convert_dtype(DType::F32));
}

// 4. Autodiff with Matrix Operations
#[test]
fn test_autodiff_matrix_operations() {
    let device = Default::default();
    
    // Create matrices
    let a = Tensor::<MyAutodiffBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device).require_grad();
    let b = Tensor::<MyAutodiffBackend, 2>::from_data([[2.0, 0.0], [1.0, 2.0]], &device).require_grad();
    
    // Matrix multiplication: y = a * b
    let y = a.clone() * b.clone();
    
    // Get gradients
    let gradients = y.backward();
    
    // Test gradients
    let grad_a = a.grad(&gradients).unwrap();
    let grad_b = b.grad(&gradients).unwrap();
    
    // Test gradient shapes
    assert_eq!(grad_a.shape().dims, [2, 2]);
    assert_eq!(grad_b.shape().dims, [2, 2]);
    
    // Test gradient values
    assert_eq!(grad_a.to_data(), TensorData::from([[2.0, 0.0], [1.0, 2.0]]).convert_dtype(DType::F32));
    assert_eq!(grad_b.to_data(), TensorData::from([[1.0, 2.0], [3.0, 4.0]]).convert_dtype(DType::F32));
}

// 5. Autodiff with Gradient Updates
#[test]
fn test_autodiff_gradient_updates() {
    let device = Default::default();
    let learning_rate = 0.1;
    
    // Create parameter
    let mut w = Tensor::<MyAutodiffBackend, 1>::from_data([2.0], &device).require_grad();
    
    // Create input and target
    let x = Tensor::<MyAutodiffBackend, 1>::from_data([3.0], &device);
    let target = Tensor::<MyAutodiffBackend, 1>::from_data([10.0], &device);
    
    // Perform one gradient update
    {
        // Forward pass: y = w * x
        let y = w.clone() * x.clone();
        
        // Calculate loss: (y - target)^2
        let loss = (y - target.clone()).powf_scalar(2.0).mean();
        
        // Backward pass
        let gradients = loss.backward();
        
        // Get gradient
        // Note: grad_w has lost the autodiff backend in order to use in operation with others Autodiff tensor
        //       so we need to convert it back to an Autodiff tensor
        let grad_w = w.grad(&gradients).unwrap();
        
        // Update parameter: w = w - learning_rate * gradient
        let grand_w_autodiff = Tensor::<MyAutodiffBackend, 1>::from_data(grad_w.into_data(), &device);
        w = w - grand_w_autodiff * learning_rate;
    }
    
    // Test updated parameter
    // Initial w = 2.0
    // Gradient = 2 * (w * x - target) * x = 2 * (2 * 3 - 10) * 3 = -24
    // New w = 2.0 - 0.1 * (-24) = 4.4
    assert!((w.clone().into_scalar() - 4.4).abs() < DEFAULT_TOLERANCE, "w should be updated to 4.4, instead we have {}", w.into_scalar());
}

// 6. Autodiff with Multiple Backward Passes
#[test]
fn test_autodiff_multiple_backward_passes() {
    let device = Default::default();
    
    // Create tensor
    let a = Tensor::<MyAutodiffBackend, 1>::from_data([1.0], &device).require_grad();
    
    // First computation: y1 = a * 2
    let y1 = a.clone() * 2.0;
    let gradients1 = y1.backward();
    
    // Second computation: y2 = a * 3
    let y2 = a.clone() * 3.0;
    let gradients2 = y2.backward();
    
    // Test gradients from both backward passes
    let grad1 = a.grad(&gradients1).unwrap();
    let grad2 = a.grad(&gradients2).unwrap();
    
    // First gradient should be 2.0 (from y1 = a * 2)
    assert_eq!(grad1.to_data(), TensorData::from([2.0]).convert_dtype(DType::F32));
    
    // Second gradient should be 3.0 (from y2 = a * 3)
    assert_eq!(grad2.to_data(), TensorData::from([3.0]).convert_dtype(DType::F32));
}
