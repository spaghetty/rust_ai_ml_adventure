use burn::backend::{Autodiff, NdArray};

//you need train to use autodiff
use burn::tensor::Tensor;

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
fn main() {
    simple_autodiff_example();
}
