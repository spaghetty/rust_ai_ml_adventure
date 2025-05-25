use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::{
    backend::{Autodiff, NdArray},
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataloader::batcher::Batcher, dataset::Dataset},
    module::Module,
    nn,
    optim::AdamConfig,       // Using Adam this time!
    record::CompactRecorder, // For saving later (optional here)
    tensor::{
        Tensor,
        backend::{AutodiffBackend, Backend},
    },
    train::{
        ClassificationOutput,
        LearnerBuilder,
        TrainOutput,
        TrainStep,
        ValidStep,
        // New! Burn's training module helps a lot!
        metric::LossMetric,
    },
};
use rand::prelude::*;

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
            let x = rng.random_range(-1.4..1.4);
            let y = rng.random_range(-1.4..1.4);
            let distance_sq = x * x + y * y;
            let target = if distance_sq < radius_sq { 1.0 } else { 0.0 };
            println!("{:?} --> {:}\n", [x, y], target);
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
    pub targets: Tensor<B, 1, Int>, // We will produce Int targets here
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
        let x = self.linear2.forward(x);
        x
    }
}
// --- TrainStep/ValidStep Implementations ---
impl<B: AutodiffBackend> TrainStep<CircleBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: CircleBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let output = self.forward(item.inputs.clone());
        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&output.device())
            .forward(output.clone().squeeze(1), item.targets.clone());

        let gradients = loss.backward();
        TrainOutput::new(
            self,
            gradients,
            ClassificationOutput {
                loss: loss.clone(),
                output,
                targets: item.targets.clone(),
            },
        )
    }
}
impl<B: Backend> ValidStep<CircleBatch<B>, ClassificationOutput<B>> for Model<B> {
    /* ... as before ... */
    fn step(&self, item: CircleBatch<B>) -> ClassificationOutput<B> {
        let output_logits = self.forward(item.inputs.clone());

        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&output_logits.device())
            .forward(output_logits.clone().squeeze(1), item.targets.clone());

        ClassificationOutput {
            loss: loss.clone(),
            output: output_logits,
            targets: item.targets.clone(),
        }
    }
}

// --- Training Loop (Simplified - Manual Approach) ---
pub fn run_training() {
    let device = Default::default();
    let config = ModelConfig::new(16); // 16 hidden neurons
    let model: Model<MyAutodiffBackend> = config.init(&device);

    let optim_config = AdamConfig::new();

    let train_dataset = CircleDataset::new(10000);
    let test_dataset = CircleDataset::new(1000);

    // --- Create Batchers ---
    // --- Create Batchers (Using our Custom Batcher!) ---
    let train_batcher = CircleBatcher {};
    let test_batcher = CircleBatcher {};
    // We specify the Backend, the input item (CircleClassificationItem)
    // and the output item (also CircleClassificationItem, but batched).
    // The `_` lets Rust infer the Item types, which is often convenient.
    let batch_size = 700;

    let train_dataloader = DataLoaderBuilder::new(train_batcher.clone())
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(4)
        .build(train_dataset);

    let test_dataloader = DataLoaderBuilder::new(test_batcher.clone())
        .batch_size(batch_size)
        .num_workers(1)
        .build(test_dataset);

    // --- Build the Learner (Same as before) ---
    let learner = LearnerBuilder::new("../data/example6/burn_post_6_learner")
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(200)
        .summary()
        .build(model, optim_config.init(), 0.005);

    println!("--- Starting Training with Learner ---");
    let model_trained = learner.fit(train_dataloader, test_dataloader);
    println!("--- Learner Training Finished ---");

    model_trained
        .save_file(
            format!("../data/example6/model"),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
        )
        .expect("Trained model saved successfully");
}

fn main() {
    run_training();
}
