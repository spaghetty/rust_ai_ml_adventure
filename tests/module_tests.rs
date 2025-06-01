use burn::backend::{Autodiff, NdArray};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::prelude::{Config, Module};
use burn::tensor::{Tensor, backend::Backend};

type MyAutodiffBackend = Autodiff<NdArray>;

// Define our Linear Neuron Module
#[derive(Module, Debug)]
pub struct LinearNeuronModule<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> LinearNeuronModule<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(input)
    }
}

// Define the Configuration for our Linear Neuron Module
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
            linear: LinearConfig::new(self.input_features, self.output_features).init(device),
        }
    }
}

// 1. Basic Module Concepts
#[test]
fn test_module_creation() {
    let device = Default::default();

    // Test module creation with configuration
    let config = LinearNeuronConfig::new(2, 1);
    let module = config.init::<MyAutodiffBackend>(&device);

    // Test module properties
    assert_eq!(module.linear.weight.shape().dims, [2, 1]);
    assert_eq!(
        module.linear.bias.expect("Bias should exist").shape().dims,
        [1]
    );
}

// 2. Module Forward Pass
#[test]
fn test_module_forward_pass() {
    let device = Default::default();

    // Create a simple module
    let config = LinearNeuronConfig::new(2, 1);
    let module = config.init::<MyAutodiffBackend>(&device);

    // Create input tensor
    let input = Tensor::<MyAutodiffBackend, 2>::from_data([[1.0, 2.0]], &device);

    // Forward pass
    let output = module.forward(input.clone());

    // Test output shape
    assert_eq!(output.shape().dims, [1, 1]);

    // Test that output is correct (weight and bias initialized randomly)
    assert!(output.to_data().to_vec::<f32>().unwrap().len() == 1);
}

// 3. Module Training Loop
#[test]
fn test_module_training() {
    let device = Default::default();
    let learning_rate = 0.01;

    // Create module
    let config = LinearNeuronConfig::new(1, 1);
    let mut module = config.init::<MyAutodiffBackend>(&device);

    // Create optimizer
    let optimizer_config = SgdConfig::new();
    let mut optimizer = optimizer_config.init();

    // Create data
    let input = Tensor::<MyAutodiffBackend, 2>::from_data([[1.0], [2.0]], &device);
    let target = Tensor::<MyAutodiffBackend, 2>::from_data([[2.0], [4.0]], &device);
    let initial_output = module.forward(input.clone());
    let initial_loss = (initial_output.clone() - target.clone())
        .powf_scalar(2.0)
        .mean();
    println!("Initial loss: {}", initial_loss.clone().into_scalar());
    // Train for a few steps
    for _ in 0..10 {
        // Forward pass
        let output = module.forward(input.clone());

        // Calculate loss (MSE)
        let loss = (output - target.clone()).powf_scalar(2.0).mean();

        // Backward pass
        let gradients = loss.backward();

        // Optimization step
        module = optimizer.step(
            learning_rate.into(),
            module.clone(),
            GradientsParams::from_grads(gradients, &module),
        );
    }

    // Test final output
    let final_output = module.forward(input.clone());
    let final_loss = (final_output.clone() - target).powf_scalar(2.0).mean();

    // Loss should be way smaller after training - this shows that the module is converging
    println!("Final loss: {}", final_loss.clone().into_scalar());
    assert!(final_loss.clone().into_scalar() < initial_loss.clone().into_scalar() * 0.5);
}

// 4. Module Parameter Updates
#[test]
fn test_module_parameter_updates() {
    let device = Default::default();

    // Create module
    let config = LinearNeuronConfig::new(1, 1);
    let mut module = config.init::<MyAutodiffBackend>(&device);

    // Save initial parameters
    let initial_weight = module.linear.weight.val().clone();
    let initial_bias = module
        .linear
        .bias
        .clone()
        .expect("Bias should exist")
        .val()
        .clone();

    // Create optimizer
    let optimizer_config = SgdConfig::new();
    let mut optimizer = optimizer_config.init();

    // Create data
    let input = Tensor::<MyAutodiffBackend, 2>::from_data([[1.0]], &device);
    let target = Tensor::<MyAutodiffBackend, 2>::from_data([[2.0]], &device);

    // Perform one training step
    {
        let output = module.forward(input.clone());
        let loss = (output - target.clone()).powf_scalar(2.0).mean();
        let gradients = loss.backward();
        module = optimizer.step(
            0.1.into(),
            module.clone(),
            GradientsParams::from_grads(gradients, &module),
        );
    }

    // Test that parameters were updated
    assert!(module.linear.weight.val().clone().into_data() != initial_weight.clone().into_data());
    assert!(
        module
            .linear
            .bias
            .expect("Bias should exist")
            .val()
            .clone()
            .into_data()
            != initial_bias.clone().into_data()
    );
}

// 5. Module Inference
#[test]
fn test_module_inference() {
    let device = Default::default();

    // Create module
    let config = LinearNeuronConfig::new(1, 1);
    let module = config.init::<MyAutodiffBackend>(&device);

    // Create input for inference
    let input = Tensor::<MyAutodiffBackend, 2>::from_data([[5.0]], &device);

    // Forward pass
    let output = module.forward(input.clone());

    // Test output
    assert_eq!(output.shape().dims, [1, 1]);

    // Test that output is not NaN
    assert!(
        !output
            .into_data()
            .to_vec::<f32>()
            .expect("Output should be a vector")
            .contains(&f32::NAN)
    );
}

// 6. Module Configuration
#[test]
fn test_module_configuration() {
    // Test configuration creation
    let config = LinearNeuronConfig::new(10, 5);

    // Test configuration values
    assert_eq!(config.input_features, 10);
    assert_eq!(config.output_features, 5);

    // Test module initialization from config
    let device = Default::default();
    let module = config.init::<MyAutodiffBackend>(&device);

    // Test module parameters
    assert_eq!(module.linear.weight.shape().dims, [10, 5]);
    assert_eq!(
        module.linear.bias.expect("Bias should exist").shape().dims,
        [5]
    );
}

// 7. Module with Multiple Layers
#[test]
fn test_module_multiple_layers() {
    let device = Default::default();

    // Create module with multiple layers
    let config = LinearNeuronConfig::new(2, 3);
    let module = config.init::<MyAutodiffBackend>(&device);

    // Create input
    let input = Tensor::<MyAutodiffBackend, 2>::from_data([[1.0, 2.0]], &device);

    // Forward pass
    let output = module.forward(input.clone());

    // Test output shape
    assert_eq!(output.shape().dims, [1, 3]);

    // Test parameter shapes
    assert_eq!(module.linear.weight.shape().dims, [2, 3]);
    assert_eq!(
        module.linear.bias.expect("Bias should exist").shape().dims,
        [3]
    );
}

// 8. Module Gradient Flow
#[test]
fn test_module_gradient_flow() {
    let device = Default::default();

    // Create module
    let config = LinearNeuronConfig::new(1, 1);
    let module = config.init::<MyAutodiffBackend>(&device);

    // Create input and target
    let input = Tensor::<MyAutodiffBackend, 2>::from_data([[1.0]], &device);
    let target = Tensor::<MyAutodiffBackend, 2>::from_data([[2.0]], &device);

    // Forward pass
    let output = module.forward(input.clone());

    // Calculate loss
    let loss = (output - target.clone()).powf_scalar(2.0).mean();

    // Backward pass
    let gradients = loss.backward();

    // Test gradients exist
    assert!(module.linear.weight.grad(&gradients).is_some());
    assert!(
        module
            .linear
            .bias
            .expect("Bias should exist")
            .grad(&gradients)
            .is_some()
    );
}
