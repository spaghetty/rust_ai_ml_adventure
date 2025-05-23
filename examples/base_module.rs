use burn::backend::{Autodiff, NdArray};
use burn::config::Config; // For #[derive(Config)]
use burn::module::Module; // For #[derive(Module)]
use burn::nn; // For nn::Linear
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::{Tensor, backend::Backend}; // For Tensor operations // For the optimizer

// Define our Autodiff backend type, as established in previous posts
type MyAutodiffBackend = Autodiff<NdArray>;

// --- 1. Define our Linear Neuron as a Burn Module ---
// This module encapsulates the linear layer and its parameters (weights and bias).
#[derive(Module, Debug)]
pub struct LinearNeuronModule<B: Backend> {
    linear: nn::Linear<B>,
}

// --- 2. Define the Configuration for our Linear Neuron Module ---
// This config will hold the dimensions for the linear layer.
#[derive(Config, Debug)]
pub struct LinearNeuronConfig {
    input_features: usize,
    output_features: usize,
}

impl LinearNeuronConfig {
    // The `init` method creates an instance of our module from the config.
    pub fn init<B: Backend>(&self, device: &B::Device) -> LinearNeuronModule<B> {
        LinearNeuronModule {
            // Initialize the nn::Linear layer with the specified dimensions.
            // This layer automatically creates its weights and biases.
            linear: nn::LinearConfig::new(self.input_features, self.output_features).init(device),
        }
    }
}

impl<B: Backend> LinearNeuronModule<B> {
    // The `forward` method defines the computation of our module.
    // It takes an input tensor and returns the output tensor.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // Pass the input through the linear layer
        self.linear.forward(input)
    }
}

// --- 3. Evolved Example: Linear Neuron Training with Module and Optimizer ---
pub fn linear_neuron_module_example() -> LinearNeuronModule<Autodiff<NdArray>> {
    let device = Default::default(); // Get the default device for NdArray backend
    let learning_rate = 0.06; // A small step size
    // Define our problem: single input feature, single output feature, target 10.0
    let input_x = Tensor::<MyAutodiffBackend, 2>::from_data([[2.0], [5.0]], &device); // Input X (batch size 2, 1 feature)
    let target_y = Tensor::<MyAutodiffBackend, 2>::from_data([[10.0], [10.0]], &device); // Target Y (batch size 2, 1 output)

    // --- 4. Configure and Initialize the Model (Our Linear Neuron Module) ---
    let model_config = LinearNeuronConfig::new(1, 1); // 1 input feature, 1 output feature
    let mut model = model_config.init(&device); // Initialize the model on the device

    // --- 5. Configure and Initialize the Optimizer ---
    // The optimizer (Stochastic Gradient Descent in this case) will manage parameter updates.
    let optimizer_config = SgdConfig::new(); // Basic SGD optimizer
    let mut optimizer = optimizer_config.init(); // Initialize the optimizer

    println!("\n--- Starting Training Loop with Module (1000 steps) ---");

    for _ in 0..1000 {
        // --- Forward Pass ---
        // The model's forward method handles the computation.
        // It automatically uses the parameters stored within the model.
        let output_y = model.forward(input_x.clone());

        // --- Calculate Loss (Mean Squared Error) ---
        // Loss = (output - target)^2
        // We use .powf_scalar(2.0) for squaring and .mean() to get a single loss value.
        let loss = (output_y - target_y.clone()).powf_scalar(2.0).mean();

        // --- Backward Pass ---
        // This calculates gradients for ALL learnable parameters within the model.
        let gradients = loss.backward();

        // --- Optimization Step ---
        // The optimizer updates the model's parameters using the calculated gradients.
        // This replaces the manual detach().require_grad() and from_data(into_data()) steps!
        model = optimizer.step(
            learning_rate.into(),
            model.clone(),
            GradientsParams::from_grads(gradients, &model),
        );

        // Print loss to observe convergence
        //println!("Step {}: Loss: {:}", i + 1, loss.to_data());
    }

    println!("\n--- Training Loop Finished ---");
    // Perform a final forward pass to see the model's output after training
    let final_output = model.forward(input_x.clone());
    let final_loss = (final_output.clone() - target_y).powf_scalar(2.0).mean();

    println!("Final Model Output: {:}", final_output.to_data());
    println!("Final Loss: {:}", final_loss.to_data());

    // You should observe that the final loss is very small,
    // and the final output is very close to the target (10.0)!
    return model.clone();
}

pub fn run_inference<B: Backend>(model: LinearNeuronModule<B>) {
    println!("\n--- Inference Example ---");

    let device = Default::default();
    // Create a new input tensor for inference (e.g., a single data point)
    let new_input = Tensor::<B, 2>::from_data([[8.0f32]], &device);
    println!("New Input: {:}", new_input.to_data());

    // Perform the forward pass to get the model's prediction
    let prediction = model.forward(new_input);
    println!("Model Prediction: {:}", prediction.to_data());

    println!("--- Inference Example Finished ---");
}

fn main() {
    let mymodel = linear_neuron_module_example();
    run_inference(mymodel);
}
