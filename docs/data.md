# Post 5: AI and ML in Rust: Datasets & DataLoaders - The Conveyor Belt for Our AI Factory!

Hey everyone, welcome back to our Rust AI/ML adventure! In our last post, we assembled our first neural networks using Burn's Module trait. But we've been working with "toy" data. To tackle more interesting problems, we need a way to manage and feed larger amounts of data.

This post was a real deep dive! We explored Dataset and DataLoader, and let me tell you, getting all the pieces to fit – especially the Batcher implementation, how TrainStep interacts with the Learner, and the specific tensor types expected by the loss function in Burn 0.17.0 (the version I'm using) – took a lot of experimentation. But we have a working solution, and I'm excited to share the code that finally clicked!

We'll cover:

* The Dataset Trait: Using simple Rust types for our items.
* The Batcher: Our custom implementation that prepares Tensors.
* The DataLoader: Putting it all together.
* The Model: A lean structure for our binary classification.
* TrainStep & ValidStep: The core logic for the Learner.
* Training with Learner: Showing the working setup.

Get ready to build the infrastructure for our AI factory! 🏭

## The Dataset: Keeping it Simple with Rust Types

My journey led me to a common pattern: define the Dataset to work with simple Rust structs, and let the Batcher handle the conversion to Tensors. This keeps the Dataset logic clean.

Our data item for the "inside or outside a circle" problem:

(Start of Code Block)
// From base_data.rs
#[derive(Debug, Clone)]
pub struct CircleClassificationItem {
pub input: [f32; 2],
pub target: f32, // 0.0 or 1.0
}
(End of Code Block)

And our CircleDataset implementation:

(Start of Code Block)
// From base_data.rs
use burn::data::dataset::Dataset;
use rand::prelude::*; // As per your working code

pub struct CircleDataset {
items: Vec&lt;CircleClassificationItem>,
}

impl CircleDataset {
pub fn new(num_samples: usize) -> Self {
let mut rng = rand::rng(); // As per your working code
let mut items = Vec::new();
let radius_sq = 1.0f32;

    for _ in 0..num_samples {
        // Assuming random_range is your working method via prelude
        let x: f32 = rng.gen_range(-1.5..1.5); // Or random_range
        let y: f32 = rng.gen_range(-1.5..1.5); // Or random_range
        let distance_sq = x * x + y * y;
        let target = if distance_sq < radius_sq { 1.0 } else { 0.0 };
        items.push(CircleClassificationItem { input: [x, y], target });
    }
    Self { items }
}
}

impl Dataset&lt;CircleClassificationItem> for CircleDataset {
fn get(&amp;self, index: usize) -> Option&lt;CircleClassificationItem> {
self.items.get(index).cloned()
}

fn len(&self) -> usize {
    self.items.len()
}
}
(End of Code Block)

## The Batcher: Converting to Tensors

The Batcher is where the magic happens: taking our Rust structs and turning them into batched Tensors. A key learning for Burn 0.17.0 was that the BinaryCrossEntropyLoss (when used in TrainStep as we did) expected Int targets (as 1D tensors). Our Batcher also needs to handle the device.

First, the CircleBatch struct, which holds the Tensors for a batch:

(Start of Code Block)
// From base_data.rs
use burn::tensor::{backend::Backend, Tensor, Int, Data};

#[derive(Debug, Clone)]
pub struct CircleBatch&lt;B: Backend> {
pub inputs: Tensor&lt;B, 2>,
pub targets: Tensor&lt;B, 1, Int>, // Batcher produces 1D Int targets
}
(End of Code Block)

And our custom CircleBatcher:

(Start of Code Block)
// From base_data.rs
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::ElementConversion;

#[derive(Clone, Default)]
pub struct CircleBatcher {}

// Note the Batcher&lt;B, I, O> signature from your working code
impl&lt;B: Backend> Batcher&lt;B, CircleClassificationItem, CircleBatch&lt;B>> for CircleBatcher {
fn batch(&amp;self, items: Vec&lt;CircleClassificationItem>, device: &amp;B::Device) -> CircleBatch&lt;B> {
let inputs_to_cat: Vec&lt;Tensor&lt;B, 2>> = items
.iter()
.map(|item| Data::from([item.input]))
.map(|data| Tensor::&lt;B, 2>::from_data(data, device))
// The reshape to [1,2] might be implicit if Data::from([item.input]) yields that shape
// for Tensor::&lt;B,2>::from_data. Your working code had a reshape here.
.map(|tensor| tensor.reshape([1, 2]))
.collect();

    let targets_to_cat: Vec<Tensor<B, 1, Int>> = items
        .iter()
        .map(|item| Tensor::<B, 1, Int>::from_data(
            Data::new(vec![item.target.elem::<B::IntElem>()], [1]),
            device
        ))
        .collect();

    let batched_inputs = Tensor::cat(inputs_to_cat, 0);
    let batched_targets = Tensor::cat(targets_to_cat, 0);

    CircleBatch {
        inputs: batched_inputs,
        targets: batched_targets,
    }
}
}
(End of Code Block)

This Batcher takes CircleClassificationItem (with Rust types) and outputs CircleBatch<B> (with Tensors), and it produces 1D Int targets, reflecting the structure of your base_data.rs. The item.target.elem::<B::IntElem>() is key for the Float to Int conversion.

## The Model: Lean and Focused

In my working version, the model structure is very lean, without an intermediate activation function or the loss function as a field.

(Start of Code Block)
// From base_data.rs
use burn::module::Module;
use burn::nn::{self, LinearConfig}; // Assuming Relu is not used
use burn::config::Config;
// Backend already imported if you use prelude or specific import

#[derive(Module, Debug)]
pub struct Model&lt;B: Backend> {
linear1: nn::Linear&lt;B>,
linear2: nn::Linear&lt;B>,
// relu field is not present as per your code
}

#[derive(Config)]
pub struct ModelConfig {
hidden_size: usize,
}

impl ModelConfig {
pub fn init&lt;B: Backend>(&amp;self, device: &amp;B::Device) -> Model&lt;B> {
Model {
linear1: nn::LinearConfig::new(2, self.hidden_size).init(device),
linear2: nn::LinearConfig::new(self.hidden_size, 1).init(device),
}
}
}

impl&lt;B: Backend> Model&lt;B> {
pub fn forward(&amp;self, input: Tensor&lt;B, 2>) -> Tensor&lt;B, 2> {
let x = self.linear1.forward(input);
self.linear2.forward(x) // Output logits shape [BatchSize, 1]
}
}
(End of Code Block)
(I've omitted Relu from the model struct and initialization, as you indicated it was useless because it wasn't used in your forward pass.)

## TrainStep and ValidStep: The Core Logic

This is where a lot of the specifics for Burn 0.17.0 became clear.

The BinaryCrossEntropyLoss (from BinaryCrossEntropyLossConfig) is initialized on-the-fly in each step.
It expects 1D Float logits (so we .squeeze(1) the model output) and 1D Int targets (which our Batcher now provides).
TrainOutput::new(...) has a very specific signature which your code reflects.
We use ClassificationOutput as the generic O in TrainOutput<O>.
(Start of Code Block)
// From base_data.rs
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};
use burn::backend::{AutodiffBackend, Backend}; // Already imported if prelude
use burn::nn::loss::BinaryCrossEntropyLossConfig;
// Tensor, Int already imported

impl&lt;B: AutodiffBackend> TrainStep&lt;CircleBatch&lt;B>, ClassificationOutput&lt;B>> for Model&lt;B> {
fn step(&amp;self, item: CircleBatch&lt;B>) -> TrainOutput&lt;ClassificationOutput&lt;B>> {
let logits_2d = self.forward(item.inputs.clone()); // Shape: [BatchSize, 1]

    // Initialize loss function inside the step
    let loss = BinaryCrossEntropyLossConfig::new()
        .init(&logits_2d.device()) // Get device from logits tensor
        .forward(logits_2d.clone().squeeze(1), item.targets.clone()); // Logits squeezed, targets are [BS, Int]

    let gradients = loss.backward();

    let classification_output = ClassificationOutput {
        loss: loss.clone(),
        output: logits_2d, // Store original [BatchSize, 1] Float logits
        targets: item.targets.clone(), // Store [BatchSize] Int targets
    };

    // This TrainOutput::new call is based on your working base_data.rs
    TrainOutput::new(
        self, // The model
        gradients,
        classification_output,
    )
}
}

impl&lt;B: Backend> ValidStep&lt;CircleBatch&lt;B>, ClassificationOutput&lt;B>> for Model&lt;B> {
fn step(&amp;self, item: CircleBatch&lt;B>) -> ClassificationOutput&lt;B> {
let logits_2d = self.forward(item.inputs.clone()); // Shape: [BatchSize, 1]

    let loss = BinaryCrossEntropyLossConfig::new()
        .init(&logits_2d.device())
        .forward(logits_2d.clone().squeeze(1), item.targets.clone());

    ClassificationOutput {
        loss,
        output: logits_2d,
        targets: item.targets,
    }
}
}
(End of Code Block)

This TrainStep directly reflects the structure in your base_data.rs, including passing self (the model) as the first argument to TrainOutput::new.

## The Learner Setup: Bringing it Home

Finally, the LearnerBuilder setup, using our custom components and settings from your file.

(Start of Code Block)
// From base_data.rs
use burn::backend::{Autodiff, NdArray}; // Already imported
use burn::train::{LearnerBuilder, metric::LossMetric};
use burn::record::CompactRecorder;
use burn::optim::AdamConfig;
use burn::data::dataloader::DataLoaderBuilder;
use burn::prelude::Config; // For ModelConfig::new
// Crate-local imports: ModelConfig, CircleDataset, CircleBatcher, Model
// Type aliases MyAutodiffBackend, MyInferenceBackend assumed defined

type MyAutodiffBackend = Autodiff&lt;NdArray>;
type MyInferenceBackend = NdArray;

pub fn run_training() {
let device = Default::default();
let config = ModelConfig::new(16); // From your code
let optim_config = AdamConfig::new();

// Initialize model once
let model: Model<MyAutodiffBackend> = config.init(&device);

let train_dataset = CircleDataset::new(10000);
let test_dataset = CircleDataset::new(400);
let batch_size = 150;

// Batchers are Default, so no device needed at init for CircleBatcher {}
// The device is passed during the .batch() call
let train_batcher = CircleBatcher {};
let test_batcher = CircleBatcher {};

let train_dataloader = DataLoaderBuilder::new(train_batcher) // Pass by value
    .batch_size(batch_size)
    .shuffle(42)
    .num_workers(1)
    .build(train_dataset);

let test_dataloader = DataLoaderBuilder::new(test_batcher) // Pass by value
    .batch_size(batch_size)
    .num_workers(1)
    .build(test_dataset);


let learner = LearnerBuilder::new("/tmp/burn_post_5_learner")
    .metric_train_numeric(LossMetric::new())
    .metric_valid_numeric(LossMetric::new())
    .with_file_checkpointer(CompactRecorder::new())
    .devices(vec![device.clone()])
    .num_epochs(100)
    .summary() // From your code
    .build(
        model, // Pass the initialized model
        optim_config.init(),
        0.02, // Learning rate as per your file
    );

println!("--- Starting Training (Based on Your Working Code) ---");
let _model_trained = learner.fit(train_dataloader, test_dataloader);
println!("--- Learner Training Finished ---");
}

fn main() {
run_training();
}
(End of Code Block)

## My Journey & What's Next

This was a deep dive! The key takeaways from finally getting this to build (based on your working code) were:
* Precisely defining the Dataset item (simple Rust types) and the Batch (holding Tensors).
* Crafting a Batcher that correctly prepares Tensor batches, including the specific Int type and 1D shape for targets as required by BinaryCrossEntropyLoss and ClassificationOutput in this setup.
* Initializing BinaryCrossEntropyLoss on-the-fly within TrainStep.
* Squeezing the logits to 1D before passing to the loss function.
* Understanding the exact arguments required by TrainOutput::new in Burn 0.17.0.
* Ensuring the DataLoader for the validation phase in Learner::fit uses a Batcher compatible with the base backend.

It's a testament to Rust's type system that we eventually get there, but it also shows how version-specific API details can be! Now that we have a solid data pipeline and training loop, the next logical step is Saving and Loading Models.
