# Post 4: AI and ML in Rust: Modules - Building Our Neural Network LEGOs! - My Journey

Alright team, welcome back to the AI/ML adventure in Rust! In our last post, we wrestled with **Automatic Differentiation (Autodiff)**. We saw how Burn magically calculates gradients, and we even manually updated a tiny neuron's weights and bias in a loop. Pretty cool, right?

But let's be real: manually managing `w` and `b` and their `require_grad()` calls, especially if we had hundreds or thousands of them, would be a total nightmare. Imagine doing that for a complex neural network with many layers! My brain hurts just thinking about it.

This is where Burn's **Modules** come to the rescue! Think of Modules as the LEGO bricks of neural networks. They're self-contained units that hold their own learnable parameters (like weights and biases) and define their own forward computation.

In this post, I'll share *my* findings on what Modules are, how to build them, and how they help us structure our models so we don't lose our minds. Get ready to build some AI LEGOs!

## What's a Module, Anyway? (And Why Do We Need Them?)

At its core, a **Module** in Burn is a struct that represents a part of your neural network. It could be:

* A single layer (like a Linear layer, a Convolutional layer).

* A whole block of layers (like a "ResNet block").

* Your entire neural network model itself.

The key idea is **encapsulation**. A Module bundles together:

1.  **Parameters:** The learnable weights and biases that the module uses.

2.  **Computation:** The `forward` function that defines how input data flows through this module to produce an output.

Why do we need them?

* **Organization:** Keeps our code clean and manageable.

* **Reusability:** Build a module once, use it multiple times (e.g., the same layer type in different parts of a network).

* **Automatic Parameter Management:** This is the big one! Burn's `Module` trait handles finding all the learnable parameters within your module and automatically setting them up for gradient tracking and optimization. No more manual `require_grad()` on every single weight!

* **Composability:** Modules can contain other modules, allowing you to build complex architectures by nesting these LEGO bricks.

## Building Our First Module: The `Module` Trait and Its `Config`

In Burn, you define a Module by creating a struct and deriving the `Module` trait for it using `#[derive(Module)]`. Modules usually come hand-in-hand with a `Config` struct that defines how to initialize them.

Let's build a simple module that acts like our linear neuron from the last post, but properly encapsulated. This module will contain a `nn::Linear` layer, which is Burn's built-in linear (or fully connected) layer.


```rust
use burn::backend::{Autodiff, NdArray};
use burn::module::Module; // For #[derive(Module)]
use burn::nn; // For nn::Linear
use burn::config::Config; // For #[derive(Config)]
use burn::tensor::{Tensor, backend::Backend}; // For Tensor operations
use burn::optim::{GradientsParams, Optimizer, Sgd, SgdConfig};
// For the optimizer

// Define our Autodiff backend type, as established in previous posts
type MyAutodiffBackend = Autodiff<NdArray>;

// --- 1. Define our Linear Neuron as a Burn Module ---
// This module encapsulates the linear layer and its parameters (weights and bias).
#[derive(Module, Debug)] // Debug is handy for printing our module
pub struct LinearNeuronModule<B: Backend> {
    linear: nn::Linear<B>, // This holds the weights and bias!
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
    // The input rank is 2 (e.g., [Batch, Features]) and output rank is 2.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // Pass the input through the linear layer
        self.linear.forward(input)
    }
}
```

A few things to note:

* `#[derive(Module)]`: This macro does a lot of heavy lifting behind the scenes, implementing the `Module` trait for your struct. It makes sure Burn can find all the parameters nested inside (like `linear`).

* `#[derive(Config)]`: This macro helps define a configuration struct for your module, which is used to initialize it.

* `nn::Linear<B>`: This is Burn's pre-built linear layer. It's a module itself! It handles creating its own weights and biases. We just declare it as a field.

* `Config::init()`: This method is the standard way to create an instance of your module, passing in the necessary hyperparameters from the config and the device.

* `forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2>`: This is the heart of any module. It takes `&self` (so it can access the module's parameters) and an input tensor, then returns the output tensor after performing its computations.

## The `forward` Function: Defining the Data Flow

The `forward` function within your `Module` is where you define the "recipe" for its computation. It's a pure function that takes an input tensor and returns an output tensor. Inside it, you call the `forward` methods of any sub-modules or perform direct tensor operations.

* **No `require_grad()` here!** You don't call `.require_grad()` on parameters inside `forward`. Burn handles that automatically when you initialize your module with an `AutodiffBackend`.

* **Tensor Operations:** You use the tensor methods we learned in Post 2 (e.g., `+`, `*`, `matmul`, `reshape`) to define the computations.

## Evolved Example: Linear Neuron Training and Inference with a Module

Now, let's put it all together! We'll use our `LinearNeuronModule` to train it to guess `10.0` for any input, just like we did with our manual neuron. This time, Burn's `Module` and `Optimizer` will handle the heavy lifting of parameter management. We'll also add an inference step to use the trained model.

```rust
use burn::backend::{Autodiff, NdArray};
use burn::config::Config; // For #[derive(Config)]
use burn::module::Module; // For #[derive(Module)]
use burn::nn; // For nn::Linear
use burn::optim::{GradientsParams, Optimizer, Sgd, SgdConfig};
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
    // The input rank is 2 (e.g., [Batch, Features]) and output rank is 2.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // Pass the input through the linear layer
        self.linear.forward(input)
    }
}

// --- 3. Evolved Example: Linear Neuron Training with Module and Optimizer ---
// Modified to return the trained model
pub fn linear_neuron_module_example() -> LinearNeuronModule<MyAutodiffBackend> {
    let device = Default::default(); // Get the default device for NdArray backend
    let learning_rate = 0.06; // A small step size
    // Define our problem: single input feature, single output feature, target 10.0
    let input_x = Tensor::<MyAutodiffBackend, 2>::from_data([[2.0f32], [5.0f32]], &device); // Input X (batch size 2, 1 feature)
    let target_y = Tensor::<MyAutodiffBackend, 2>::from_data([[10.0f32], [10.0f32]], &device); // Target Y (batch size 2, 1 output)

    // --- 4. Configure and Initialize the Model (Our Linear Neuron Module) ---
    let model_config = LinearNeuronConfig::new(1, 1); // 1 input feature, 1 output feature
    let mut model = model_config.init(&device); // Initialize the model on the device

    // --- 5. Configure and Initialize the Optimizer ---
    // The optimizer (Stochastic Gradient Descent in this case) will manage parameter updates.
    let optimizer_config = SgdConfig::new(); // Basic SGD optimizer
    let mut optimizer = optimizer_config.init(); // Initialize the optimizer

    println!("\n--- Starting Training Loop with Module (1000 steps) ---");

    for i in 0..1000 {
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
        println!("Step {}: Loss: {:}", i + 1, loss.to_data());
    }

    println!("\n--- Training Loop Finished ---");
    // Perform a final forward pass to see the model's output after training
    let final_output = model.forward(input_x.clone());
    let final_loss = (final_output.clone() - target_y).powf_scalar(2.0).mean();

    println!("Final Model Output: {:}", final_output.to_data());
    println!("Final Loss: {:}", final_loss.to_data());

    // You should observe that the loss decreases with each step,
    // indicating that our parameters are converging towards the target!

    return model; // Return the trained model
}

// --- Inference Function ---
// This function takes a trained model and an input tensor and makes a prediction.
pub fn run_inference<B: Backend>(model: LinearNeuronModule<B>) {
    let device = Default::default();
    println!("\n--- Inference Example ---");

    // Create a new input tensor for inference (e.g., a single data point)
    let new_input = Tensor::<B, 2>::from_data([[8.0f32]], &device);
    println!("New Input: {:?}", new_input.to_data());

    // Perform the forward pass to get the model's prediction
    let prediction = model.forward(new_input);
    println!("Model Prediction: {:?}", prediction.to_data());

    println!("--- Inference Example Finished ---");
}

// Example main function to run the training and inference:
/*
fn main() {
    let trained_model = linear_neuron_module_example(); // Train the model and get the trained instance
    run_inference(trained_model); // Call inference with the trained model
}
*/
```
See the difference?

We initialize the model using its Config.
We initialize an optimizer.
Inside the loop, loss.backward() automatically finds all the parameters in our model and calculates their gradients.
optimizer.step(...) takes care of updating all those parameters based on the gradients and the learning rate.
No more manual w.detach().require_grad()! This is so much cleaner and less error-prone. 🙌

Building Bigger LEGOs: Composing Modules
The real power comes when you start putting these LEGO bricks together. A Module can contain other Modules. Let's build something slightly more complex: a simple two-layer network. It will have:

A linear layer.
A ReLU activation function.
Another linear layer.
What's a ReLU? It's an activation function. Think of it as a "switch" that introduces non-linearity into our network. Real-world data is rarely linear, so we need these non-linearities to learn complex patterns. ReLU (Rectified Linear Unit) is super simple: if the input is positive, it passes it through; if it's negative, it outputs zero. Burn has a built-in nn::ReLU module for this.

Let's build our TwoLayerNet:

```rust
use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::{self, ReLU}; // Include ReLU
use burn::config::Config;
use burn::tensor::{Tensor, backend::Backend};

type MyAutodiffBackend = Autodiff<NdArray>;

// --- 1. Define our Two-Layer Network Module ---
#[derive(Module, Debug)]
pub struct TwoLayerNet<B: Backend> {
    linear1: nn::Linear<B>,
    relu: ReLU, // Our activation function module
    linear2: nn::Linear<B>,
}

// --- 2. Define the Configuration ---
#[derive(Config, Debug)]
pub struct TwoLayerNetConfig {
    input_features: usize,
    hidden_features: usize, // Size of the intermediate layer
    output_features: usize,
}

impl TwoLayerNetConfig {
    // The `init` method creates our network.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TwoLayerNet<B> {
        TwoLayerNet {
            linear1: nn::LinearConfig::new(self.input_features, self.hidden_features).init(device),
            relu: ReLU::new(), // ReLU doesn't need parameters, so init is simple
            linear2: nn::LinearConfig::new(self.hidden_features, self.output_features).init(device),
        }
    }
}

impl<B: Backend> TwoLayerNet<B> {
    // The `forward` method defines the data flow through our network.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input); // Pass through first linear layer
        let x = self.relu.forward(x);       // Apply activation
        let x = self.linear2.forward(x);    // Pass through second linear layer
        x // Return the final output
    }
}

// --- Example Usage (Initialization) ---
pub fn create_two_layer_net() {
    let device = Default::default();
    let config = TwoLayerNetConfig::new(10, 20, 5); // 10 inputs, 20 hidden, 5 outputs
    let model: TwoLayerNet<MyAutodiffBackend> = config.init(&device);

    println!("\n--- Created a Two-Layer Network ---");
    println!("{:?}", model);
}

// Example main function:
/*
fn main() {
    let trained_model = train_linear_neuron();
    run_inference(&trained_model);
    create_two_layer_net(); // Create and print the 2-layer net
}
*/
```
See how easy that was? We just declared our layers and activation function as fields in the TwoLayerNet struct, and in the forward function, we defined the order they should be applied. Burn's #[derive(Module)] takes care of finding all the parameters in linear1 and linear2 when we train or inspect the model. This is the power of composability. You can build incredibly complex networks by combining simpler modules.

nference: Using Our Trained Model
Training is great, but the end goal is usually to use the model to make predictions on new data. This is called inference.

Our run_inference function already showed the basics: you just call the forward method with your new input data.

Key Points about Inference:

No Gradients Needed: During inference, we don't need to calculate gradients. This can make it faster. While our example uses the Autodiff backend, for deployment, you often switch to a backend without Autodiff (like NdArray directly) or use methods to disable gradient tracking, which can save computation. Burn handles this gracefully.
Model State: The model object contains all the learned weights and biases. You'd typically save these trained parameters to a file and load them back when you need to run inference later. (We'll cover saving and loading in a future post!)


What's Next?
We've made huge progress! We've learned about Tensors, Autodiff, and now how to structure our models using Modules and Configs. We even built a complete, runnable example that trains a simple neuron and uses it for inference, all managed automatically by Burn's powerful traits.

This is a solid foundation for building more complex neural networks. But so far, our data is just hardcoded tensors. In real-world AI/ML, you'll work with large datasets!

In the next post, we'll shift our focus to Data Loading. We'll explore how to work with Burn's Dataset and Batcher traits to efficiently prepare and feed real data into our models.

Stay tuned for data adventures!

Found Modules intriguing? Got ideas for your own custom LEGO bricks? Share your thoughts below! Let's learn and build AI/ML in Rust together!
