# 4: Rust Meets AI: Data Pipeline — Approach the convoy belt of ML

![project image](./rust-burn-wide.jpg?raw=true)

As someone who loves learning new things, I often find myself exploring various fields. Lately, I’ve been diving into Rust, with AI and machine learning always on my mind. When I discovered a growing Rust AI ecosystem, it clicked — why not combine both? I’m no expert in either Rust or AI, so this is a learning journey. If you have any insights or corrections, please share them here or on my GitHub. Let’s learn together\!

**What’s for today:**

Welcome back to our Rust AI/ML adventure\! In this series, we’re moving beyond basic setups and delving into real-world applications. Today, we tackle a crucial component of any Machine Learning project: building an efficient data processing line. Think of it as setting up the infrastructure that feeds our model with the right data, in the right way.

**We’ll cover:**

* **The Dataset Trait:** Defining our data structure using simple Rust types.
* **The Batcher:** Building the moving belt that prepares our data batches.
* **The DataLoader**: The packaging station that organises and delivers data to the model.
* **The Model:** A basic NN setup for our binary classification task.

Get ready to get closer to the infrastructure for our AI factory\! 🏭.

(We’ll introduce the final piece, **the Learner**, which automates everything, in a future post.)

## The Objective

As in the previous example, we will start by defining the objective we want to reach:

Given a point in a 2D space, we want to determine if it’s inside or outside a circle with a radius of 1.

Our points will have coordinates in the range of \-1.5 to 1.5 to keep them within a comparable dimension.

This data that we can create on the fly is referred as synthetic data.

## The import section

Is still complex for me identifying all the import statement needed so I like to explicitly share it:

```rust
use burn::prelude::*;
use burn::{
   backend::{Autodiff, NdArray},
   config::Config,
   data::{
       dataloader::DataLoader, dataloader::DataLoaderBuilder, dataloader::batcher::Batcher,
       dataset::Dataset,
   },
   module::Module,
   nn,
   tensor::{Tensor, backend::Backend},
};
use rand::prelude::*;
use std::sync::Arc;
```

## The Dataset: Defining Our Data Structure

At its core, a dataset is a collection of data related to a specific task. The data can be of many types, such as images, text, audio, or video.

Essentially, a dataset is simply a collection of our data with input values and corresponding labels.

We’ll create a Dataset that auto-generates this synthetic data for us.

```rust
#[derive(Debug, Clone)]
pub struct CircleClassificationItem {
   pub input: [f32; 2], // Make it 2D [1, 2] for easier batching
   pub target: f32,     // Make it 2D [1, 1]
}

pub struct CircleDataset {
   items: Vec<CircleClassificationItem>,
}
impl CircleDataset {
   pub fn new(num_samples: usize) -> Self {
       let mut rng = rand::rng();
       let mut items = Vec::new();
       let radius_sq = 1.0f32;
       for _ in 0..num_samples {
           let x = rng.random_range(-1.5..1.5);
           let y = rng.random_range(-1.5..1.5);
           let distance_sq = x * x + y * y;
           let target = if distance_sq < radius_sq { 1.0 } else { 0.0 };
           items.push(CircleClassificationItem {
               input: [x, y],
               target,
           });
       }
       Self { items }
   }
}
// Update get to return 2D Tensors for default collation
impl Dataset<CircleClassificationItem> for CircleDataset {
   fn get(&self, index: usize) -> Option<CircleClassificationItem> {
       self.items.get(index).cloned()
   }

   fn len(&self) -> usize {
       self.items.len()
   }
}
```
here we are defining:

* The **CircleClassificationItem**: the single item in our dataset
* The **CircleDataset**: the structure containing all our items
* A **new** method: that is generating the data according to our definition
* An implementation of the **DataSet Trait** for our **CircleDataset**

```rust
// (not project code) Burn definition of the data set Trait
pub trait Dataset<I>: Send + Sync {
   fn get(&self, index: usize) -> Option<I>;
   fn len(&self) -> usize;
}
```

## The Batcher: The Moving Belt

The Batcher is the core of our data pipeline, like the moving belt of our conveyor. It takes individual data items (like our point coordinates) and groups them into ‘batches.’ These batches are then transformed into Tensors, which is the format our Neural Network model understands.

```rust
// This struct holds the final Tensors for a batch
#[derive(Debug, Clone)]
pub struct CircleBatch<B: Backend> {
   pub inputs: Tensor<B, 2>,
   pub targets: Tensor<B, 1, Int>,
}

#[derive(Clone, Default)] // Batchers need to be Clone
pub struct CircleBatcher {}

impl<B: Backend> Batcher<B, CircleClassificationItem, CircleBatch<B>> for CircleBatcher {
   fn batch(&self, items: Vec<CircleClassificationItem>, device: &B::Device) -> CircleBatch<B> {
       // Extract all the input tensors and target tensors from the Vec
       let inputs = items
           .iter()
           .map(|item| TensorData::from([item.input]))
           .map(|data| Tensor::<B, 2>::from_data(data, device))
           .map(|tensor| tensor.reshape([1, 2]))
           .collect();

       let targets = items
           .iter()
           .map(|item| Tensor::<B, 1, Int>::from_data([item.target.elem::<B::IntElem>()], device))
           .collect();

       // Concatenate along dimension 0. Since each tensor is [1, N],
       // concatenating N of them along dim 0 results in [N, M].
       let batched_input = Tensor::cat(inputs, 0);
       let batched_target = Tensor::cat(targets, 0);
       // Return a new 'Item', but this time it contains batched tensors.
       CircleBatch {
           inputs: batched_input,
           targets: batched_target,
       }
   }
}
```

This Batcher takes our simple data types and outputs data in a format with Tensors, along with integer labels indicating whether the point is inside or outside the circle.

## The Model: Lean and Focused

Here’s the model structure, just to visualise the use of multiple layers in the neural network.

```rust
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
   linear1: nn::Linear<B>,
   activation: nn::Relu,
   linear2: nn::Linear<B>,
}

#[derive(Config)]
pub struct ModelConfig {
   hidden_size: usize,
}

impl ModelConfig {
   pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
       Model {
           linear1: nn::LinearConfig::new(2, self.hidden_size).init(device),
           activation: nn::Relu::new(),
           linear2: nn::LinearConfig::new(self.hidden_size, 1).init(device),
       }
   }
}

impl<B: Backend> Model<B> {
   pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
       let x = self.linear1.forward(input);
       let x = self.activation.forward(x);
       self.linear2.forward(x)
   }
}
```

To connect the DataLoader and the Model the process is the same as in our previous example. We now have a structured dataset that can be easily passed to the fit function to train the model and validate the results.

### The DataLoader: The Packaging Station

The DataLoader is the final stage, similar to the packaging station. It uses the Batcher to create batches and then handles the process of shuffling, splitting into training and validation sets, and delivering these batches to the model in a structured way. It ensures that the data is organized and fed to the model efficiently during training.

The DataLoader guarantees that the data and labels are correct, that the data is properly shuffled between epochs, and that no duplicate data is used during different steps.

```rust
pub fn run_training() {
   let device = Default::default();
   let config = ModelConfig::new(16); // 16 hidden neurons
   let _model: Model<MyAutodiffBackend> = config.init(&device);
   let batch_size = 700;

   // --- Build the train and test dataset ---
   let train_dataset = CircleDataset::new(10000);
   let test_dataset = CircleDataset::new(1500);

   // --- Create Batchers ---
   let train_batcher = CircleBatcher {};
   let test_batcher = CircleBatcher {};

   // --- Prepare the DataLoader
   let train_dataloader: Arc<dyn DataLoader<MyAutodiffBackend, CircleBatch<MyAutodiffBackend>>> =
       DataLoaderBuilder::new(train_batcher.clone())
           .batch_size(batch_size)
           .shuffle(42)
           .num_workers(1)
           .build(train_dataset);

   let test_dataloader: Arc<dyn DataLoader<MyAutodiffBackend, CircleBatch<MyAutodiffBackend>>> =
       DataLoaderBuilder::new(test_batcher.clone())
           .batch_size(batch_size)
           .num_workers(1)
           .build(test_dataset);

   // Visualise an epoch
   println!("training data splitted in batches and shuffle");
   for i in train_dataloader.iter() {
       println!(
           "data: {:?}, label: {:?}",
           i.inputs.shape(),
           i.targets.shape()
       );
   }
   println!("validating data splitted in bactches");
   for i in test_dataloader.iter() {
       println!(
           "data: {:?}, label: {:?}",
           i.inputs.shape(),
           i.targets.shape()
       );
   }
}
```
Now, we can use this organised data (which changes each iteration) to train our model.

## My Journey & What’s Next

This was a deep dive into data preparation, which is a critical part of any Machine Learning project. In our prior example, our data setup was very basic. Now that we have a robust data foundation, we’ll see how it impacts a real ML pipeline.

We will introduce:

* **Learner:** How to automate all the steps for training a NN correctly.
* **Data Saving:** How to save our learning results for later use (or deployment) for the inference step, where the actual value lies.
