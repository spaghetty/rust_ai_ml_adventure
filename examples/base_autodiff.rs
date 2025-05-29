use burn::backend::{Autodiff, NdArray};

//you need train to use autodiff
use burn::tensor::Tensor;

// Define our Autodiff backend type
type MyAutodiffBackend = Autodiff<NdArray>;

pub fn simple_autodiff_example() {
    let device = Default::default(); // Get the default device for NdArray

    // --- 1. Create Tensors, some requiring gradients ---
    // Tensor 'a' - we want to track its gradient
    let a = Tensor::<MyAutodiffBackend, 1>::from_data([10.0], &device).require_grad();
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

    for _ in 0..100 {
        // --- 2. Perform the linear operation (forward pass) ---
        // y = (x * w) + b
        let y = (x.clone() * w.clone()) + b.clone();
        println!("\nNeuron Output 'y': {:}", y.to_data());

        // --- 3. Calculate a simple Mean Squared Error (MSE) loss ---
        // Loss = (y - target)^2
        let loss = (y - target.clone()).powf_scalar(2.0).mean();
        println!("Calculated Loss: {:}", loss.to_data());

        // --- 4. Perform the Backward Pass ---
        // Calculate gradients of the loss with respect to 'w' and 'b'
        let gradients = loss.backward();

        // --- 5. Access the Gradients ---
        let grad_w = w.grad(&gradients).unwrap();
        let grad_b = b.grad(&gradients).unwrap();

        println!(
            "\nGradient of Loss with respect to 'w': {:}",
            grad_w.to_data()
        );
        println!(
            "Gradient of Loss with respect to 'b': {:}",
            grad_b.to_data()
        );

        // --- 6. Apply Gradients (Simulating one step of Gradient Descent) ---
        // This is where we use the gradients to update our parameters!
        // Update rule: new_param = old_param - learning_rate * gradient
        //
        let grad_w_autodiff =
            Tensor::<MyAutodiffBackend, 1>::from_data(grad_w.into_data(), &device); // Convert to Autodiff backend
        let grad_b_autodiff =
            Tensor::<MyAutodiffBackend, 1>::from_data(grad_b.into_data(), &device); // Convert to Autodiff backend

        // Update weight 'w'
        w = w
            .sub(grad_w_autodiff * learning_rate)
            .detach()
            .require_grad();
        println!("\nUpdated Weight 'w': {:}", w.to_data());

        // Update bias 'b'
        b = b
            .sub(grad_b_autodiff * learning_rate)
            .detach()
            .require_grad();
        println!("Updated Bias 'b': {:}", b.to_data());
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
fn main() {
    simple_autodiff_example();
    linear_neuron_autodiff_example();
}
