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

// --- Backend and Data (as before) ---
type MyAutodiffBackend = Autodiff<NdArray>;
//type MyInferenceBackend = NdArray;

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

// --- Model ---
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

// --- Training Loop (Simplified - Manual Approach) ---
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

    // Visualize an epoch
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

fn main() {
    run_training();
}
