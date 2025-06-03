# **3: Rust Meets AI: Modular Magic, Organising Neural Nets with Burn**

![project image](./rust-burn.jpg?raw=true)

Welcome back to our Rust AI journey! In the last post, we conquered **Autodiff**, the magic behind neural network learning. But manually tracking every weight and bias? No thanks! Today, we’ll discover how Burn’s **Modules** make building neural networks a breeze.

## **Why Modules? Think LEGOs!**

Modules are like LEGO bricks for neural networks:

- **Self-contained Units:** They hold learnable _parameters_ (weights, biases) and their _computation_ (how input becomes output).
- **Organisation:** Keep your code clean and manageable.
- **Reusability:** Build a module once, use it many times.
- **Automatic Parameter Management:** Burn handles gradients, so you don’t have to.
- **Composability:** Combine modules to build complex networks.

## **Building Our First Module: The Linear Neuron**

Let’s create a module for a simple linear neuron, similar to what we did manually before.

Let’s first focus on import and model setup:

```rust
use burn::backend::{Autodiff, NdArray};
use burn::module::Module; // our module. Exposes #[derive(Module)]
use burn::nn; // our NeuralNetwork building blocks like nn::Linear
use burn::config::Config; // configuration for our model. Exposes #[derive(Config)]
use burn::tensor::{Tensor, backend::Backend};
use burn::optim::{GradientsParams, Optimizer, Sgd, SgdConfig};
// For the optimizer the element that updates the NN params for us.
// Define our Autodiff backend type, as established in previous posts
type MyAutodiffBackend = Autodiff<NdArray>;

Now let’s see the actual Module definition

// 1. Define our Linear Neuron as a Burn Module
// This module encapsulates the linear layer and its parameters (weights and bias).
#[derive(Module, Debug)] // Debug is handy for printing our module
pub struct LinearNeuronModule<B: Backend> {
 linear: nn::Linear<B>, // This holds the weights and bias!
}
// 2. Define the Configuration for our Linear Neuron Module
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
 // The input rank is 2 (e.g., [Batch, Features]) and output rank is 2.
 pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
   // Pass the input through the linear layer
   self.linear.forward(input)
 }
}
```

A few things to note:

**#[derive(Module)]**: does the magic! It automatically implements Burn’s Module trait, ensuring Burn finds all parameters (like linear).”

**#[derive(Config)]**: This macro helps define a configuration struct for your module, which is used to initialise it.

**nn::Linear<B>**: This is Burn’s pre-built linear layer. It’s a module itself! It handles creating its own weights and biases. We just declare it as a field.

**forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2>**: This is the heart of any module. It takes input tensor, then returns the output tensor after performing its computations. It is where you define your ‘recipe’ for the NN computation.

## **Lifecycle of a Neural Network**

Before starting we need to clarify a basic concept related to the lifecycle of a NN.

A NN has two important phases:

* **Training**: is when you tune the parameter learning from labeled examples: *in our example you will see: (input: 2.0, label: 10.0), (input: 5.0, label: 10.0)*
* **Inference**: is when you use the trained NN to guess result for other input: *in our example you will see: (input: 8.0)*

This is just part of the real process, with a real dataset you will split the Training phase in two different sub-phases “Training” and “Validation” in order to prevent overfitting. But this requires bigger datasets and we will discover it later.

## **Linear Neuron Training and Inference**

Now, let’s put it all together! We define a dummy problem here: we’ll train the Linear Model to guess ‘10.0’ for any input, and we will explicitly separate **Training** and **Inference** steps.

This time, Burn’s ‘Module’ and ‘Optimizer’ will handle the heavy lifting of parameter management.

```rust
// -- 3. Linear Neuron Training with Module and Optimizer --
pub fn linear_neuron_module_example() -> LinearNeuronModule<MyAutodiffBackend> {
 let device = Default::default();
 let learning_rate = 0.06; // A small step size
 // Problem: single input feature, single output feature, target 10.0
 let input_x = Tensor::<MyAutodiffBackend, 2>::from_data(
                             [[2.0], [5.0]],
                             &device); // Input X (batch size 2, 1 feature)
 let target_y = Tensor::<MyAutodiffBackend, 2>::from_data(
                             [[10.0], [10.0]],
                             &device); // Target Y (batch size 2, 1 output)
 // -- 4. Configure and Initialize the Model--
 // 1 input feature, 1 output feature
 let model_config = LinearNeuronConfig::new(1, 1);
 // Initialize the model on the device
 let mut model = model_config.init(&device);
 // -- 5. Configure and Initialize the Optimizer --
 // The optimizer (Stochastic Gradient Descent in this case)
 // will manage parameter updates.
 let optimizer_config = SgdConfig::new();
 let mut optimizer = optimizer_config.init();
 println!("\\n --- Starting Training Loop with Module (1000 steps) ---");
 for i in 0..1000 {
   // -- Forward Pass --
   // The model's forward method handles the computation.
   // It automatically uses the parameters stored within the model.
   let output_y = model.forward(input_x.clone());
   // -- Calculate Loss (Mean Squared Error) --
   // Loss = (output - target)²
   // We use .powf_scalar(2.0) for squaring and .mean() to get a single loss value.
   let loss = (output_y - target_y.clone()).powf_scalar(2.0).mean();
   // -- Backward Pass --
   // This calculates gradients for ALL learnable parameters within the model.
   let gradients = loss.backward();
   // -- Optimization Step --
   // The optimizer updates the model's parameters using the calculated
   // gradients. This replaces any other manual steps!
   model = optimizer.step(
       learning_rate.into(),
       model.clone(),
       GradientsParams::from_grads(gradients, &model),
     );
   // Print loss to observe convergence
   println!("Step {}: Loss: {:}", i \+ 1, loss.to_data());
 }
 println!("\\n --- Training Loop Finished ---");
 // Perform a final forward pass to see the model's output after training
 let final_output = model.forward(input_x.clone());
 let final_loss = (final_output.clone() - target_y).powf_scalar(2.0).mean();
 println!("Final Model Output: {:}", final_output.to_data());
 println!("Final Loss: {:}", final_loss.to_data());
 // You should observe that the loss decreases with each step,
 // indicating that our parameters are converging towards the target!
 return model.clone(); // Return the trained model
}
// -- Inference Function --
// This function takes a trained model and to makes a prediction.
pub fn run_inference<B: Backend>(model: LinearNeuronModule<B>) {
 let device = Default::default();
 println!("\\n --- Inference Example ---");
 // Create a new input tensor for inference (e.g., a single data point)
 let new_input = Tensor::<B, 2>::from_data([[8.0]], &device);
 println!("New Input: {:}", new_input.to_data());
 // Perform the forward pass to get the model's prediction
 let prediction = model.forward(new_input);
 println!("Model Prediction: {:}", prediction.to_data());
 println!(" --- Inference Example Finished ---");
}

fn main() {
 // Train the model and get the trained instance
 let trained_model = linear_neuron_module_example();
 // Call inference with the trained model
 run_inference(trained_model);
}
```

See the difference? The Model and Optimizer are doing most of the heavy lifting for us.

Inside the training loop, loss.backward() automatically finds all the parameters in our model and calculates their gradients then optimizer.step(…) takes care of updating all those parameters based on the gradients and the learning rate. Cool 🙌

## **Building Bigger LEGOs: Composing Modules**

The real power comes when you start putting these LEGO bricks together. Let’s build something slightly more complex: a simple two-layer network. We will base our example on a more subtle problem: “***Numbers below 6 should yield 5, while numbers 6 and above should yield 15***” this introduces some non-linearity in the problem.

It will have:

* A **linear** layer.
* A **Sigmoid** activation function.
* Another **linear** layer.

What’s a Sigmoid? It’s an activation function that squashes values between 0 and 1. Imagine it as a smooth “S” shaped curve. It’s often used to introduce non-linearity into neural networks and is especially useful in scenarios where you want to predict probabilities.

```rust
use burn::{
 backend::{Autodiff, NdArray},
 config::Config,
 module::Module,
 nn::{self, LinearConfig, Sigmoid},
 optim::{GradientsParams, Optimizer, SgdConfig},
 tensor::{Tensor, backend::Backend},
};
// Define our Autodiff backend type
type MyAutodiffBackend = Autodiff<NdArray>;
// -- 1. Two-Layer Network Module Definition --
#[derive(Module, Debug)]
pub struct TwoLayerNet<B: Backend> {
 linear1: nn::Linear<B>,
 activation: Sigmoid,
 linear2: nn::Linear<B>,
}
#[derive(Config, Debug)]
pub struct TwoLayerNetConfig {
 input_features: usize,
 hidden_features: usize,
 output_features: usize,
}
impl TwoLayerNetConfig {
 pub fn init<B: Backend>(&self, device: &B::Device) -> TwoLayerNet<B> {
   TwoLayerNet {
     linear1: LinearConfig::new(self.input_features, self.hidden_features).init(device),
     activation: Sigmoid::new(),
     linear2: LinearConfig::new(self.hidden_features, self.output_features).init(device),
   }
 }
}
impl<B: Backend> TwoLayerNet<B> {
 pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
   let x = self.linear1.forward(input);
   let x = self.activation.forward(x);
   let x = self.linear2.forward(x);
   x
 }
}
// -- 5. Training Function for Two-Layer Network --
pub fn train_two_layer_net() {
 let device = Default::default();
     let learning_rate = 0.03;
     let hidden_size = 10;
     let input_x = Tensor::<MyAutodiffBackend, 2>::from_data(
     [[1.0], [2.0], [5.0], [6.0], [7.0], [22.0]],
     &device,
   );
 let target_y = Tensor::<MyAutodiffBackend, 2>::from_data(
     [[5.0], [5.0], [5.0], [15.0], [15.0], [15.0]],
     &device,
   );
 let config = TwoLayerNetConfig::new(1, hidden_size, 1);
 let mut model = config.init(&device);
 let optimizer_config = SgdConfig::new();
 let mut optimizer = optimizer_config.init();
 println!("\\n --- Training the Two-Layer Network (2000 steps) ---");
 for i in 0..50000 {
   let output_y = model.forward(input_x.clone());
   let loss = (output_y.clone() - target_y.clone())
       .powf_scalar(2.0)
       .mean();
   let gradients = loss.backward();
   model = optimizer.step(
       learning_rate.into(),
       model.clone(),
       GradientsParams::from_grads(gradients, &model),
     );
   println!("Step {}: Loss: {:.4}", i \+ 1, loss.to_data());
 }
 println!(" --- Training Finished ---");
 let final_output = model.forward(input_x.clone());
 println!(
     "Final Model Output:\\nInput: {:}\\nTarget: {:}\\nOutput: {:}",
     input_x.into_data(),
     target_y.into_data(),
     final_output.into_data()
 );
 println!("\\n --- Two-Layer Net Quick Inference Test ---");
 let test_input_1 = Tensor::<MyAutodiffBackend, 2>::from_data(
                                                     [[3.0]], &device);
 let test_input_2 = Tensor::<MyAutodiffBackend, 2>::from_data(
                                                     [[8.0]], &device);
 let test_input_3 = Tensor::<MyAutodiffBackend, 2>::from_data(
                                                     [[5.0]], &device);
 let test_input_4 = Tensor::<MyAutodiffBackend, 2>::from_data(
                                                     [[22.0]], &device);
 println!(
     "Test [3.0] -> Pred: {:}",
     model.forward(test_input_1).into_data()
   );
 println!(
     "Test [8.0] -> Pred: {:}",
     model.forward(test_input_2).into_data()
   );
 println!(
     "Test [5.0] -> Pred: {:}",
     model.forward(test_input_3).into_data()
   );
 println!(
     "Test [22.0] -> Pred: {:}",
     model.forward(test_input_4).into_data()
   );
 println!(" - - Inference Test Finished - -");
}
```

See how easy that was? We just declared our layers and activation function as fields in the TwoLayerNet struct, and in the forward function, we defined the order they should be applied. This is the power of composability. You can build incredibly complex networks by combining simpler modules.

### **What’s Next?**

We’ve built modules and trained a network. But where does the data come from? In the next post, we’ll cover **Data Loading** with Burn.

Found Modules intriguing? Got ideas for your own custom LEGO bricks? Share your thoughts below! Let’s learn and build AI/ML in Rust together!
