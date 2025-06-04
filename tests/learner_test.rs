//! Educational tests for Burn's Learner component: demonstrates end-to-end supervised training with model, optimizer, loss, and dataloaders.
//!
//! This file is designed to show how to use Burn's high-level Learner abstraction
//! for training and validation, and to verify that the training loop is working as expected.
use burn::backend::{Autodiff, NdArray};
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::module::Module;
use burn::nn::loss::MseLoss;
use burn::nn::loss::Reduction;
use burn::nn::{Linear, LinearConfig};
use burn::optim::SgdConfig;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Shape, Tensor, TensorData};
use burn::train::metric::LossMetric;
use burn::train::{
    LearnerBuilder,
    renderer::{MetricState, MetricsRenderer, TrainingProgress},
};
use burn::train::{RegressionOutput, TrainOutput, TrainStep, ValidStep};
use std::sync::{Arc, Mutex};

const BATCH_SIZE: usize = 16;

// --- Dummy dataset and batcher for regression (copied from integration test) ---
#[derive(Debug, Clone)]
struct ToyRegressionItem {
    x: [f32; 1],
    y: f32,
}

#[derive(Debug, Clone)]
struct ToyRegressionBatch<B: Backend> {
    x: Tensor<B, 2>,
    y: Tensor<B, 1>,
}

#[derive(Debug, Clone, Default)]
struct ToyRegressionBatcher;
impl<B: Backend> Batcher<B, ToyRegressionItem, ToyRegressionBatch<B>> for ToyRegressionBatcher {
    fn batch(&self, items: Vec<ToyRegressionItem>, device: &B::Device) -> ToyRegressionBatch<B> {
        let n = items.len();
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for item in items {
            x.push(item.x[0]);
            y.push(item.y);
        }
        ToyRegressionBatch {
            x: Tensor::from_data(TensorData::new(x, Shape::new([n, 1])), device),
            y: Tensor::from_data(TensorData::new(y, Shape::new([n])), device),
        }
    }
}

#[derive(Debug, Clone)]
struct ToyRegressionDataset {
    data: Vec<ToyRegressionItem>,
}

impl ToyRegressionDataset {
    fn new(size: usize) -> Self {
        // Normalize x to [0, 1] for stable training
        let data = (0..size)
            .map(|i| {
                let x = i as f32 / (size as f32 - 1.0);
                ToyRegressionItem {
                    x: [x],
                    y: 2.0 * x + 1.0,
                }
            })
            .collect();
        Self { data }
    }
}
impl Dataset<ToyRegressionItem> for ToyRegressionDataset {
    fn get(&self, index: usize) -> Option<ToyRegressionItem> {
        self.data.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.data.len()
    }
}

// --- Minimal Linear Regression Model ---
#[derive(Module, Debug)]
struct LinearModel<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> LinearModel<B> {
    fn new(device: &<B as Backend>::Device) -> Self {
        Self {
            linear: LinearConfig::new(1, 1).init(device),
        }
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }
}

// Implement TrainStep and ValidStep for LinearModel
impl<B: AutodiffBackend> TrainStep<ToyRegressionBatch<B>, RegressionOutput<B>> for LinearModel<B> {
    fn step(&self, batch: ToyRegressionBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let output = self.forward(batch.x.clone());
        let output = output.clone().reshape([output.shape().dims[0], 1]);
        let target = batch.y.clone().reshape([batch.y.shape().dims[0], 1]);
        let loss = MseLoss::new().forward(output.clone(), target.clone(), Reduction::Mean);
        let gradients = loss.backward();
        let regression = RegressionOutput::new(loss, output, target);
        TrainOutput::new(self, gradients, regression)
    }
}
impl<B: Backend> ValidStep<ToyRegressionBatch<B>, RegressionOutput<B>> for LinearModel<B> {
    fn step(&self, batch: ToyRegressionBatch<B>) -> RegressionOutput<B> {
        let output = self.forward(batch.x.clone());
        let output = output.clone().reshape([output.shape().dims[0], 1]);
        let target = batch.y.clone().reshape([batch.y.shape().dims[0], 1]);
        let loss = MseLoss::new().forward(output.clone(), target.clone(), Reduction::Mean);
        RegressionOutput::new(loss, output, target)
    }
}

// Custom Renderer
#[derive(Clone)]
struct CustomRenderer {
    storage: Arc<Mutex<Vec<f32>>>,
}

impl CustomRenderer {
    fn new() -> Self {
        CustomRenderer {
            storage: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn last(&self) -> f32 {
        let size = self.storage.lock().unwrap().len();
        let batch = if size / 2 > BATCH_SIZE {
            BATCH_SIZE
        } else {
            size / 2
        };
        let storage = self.storage.lock().unwrap();
        let mut avg = 0.0;
        for i in 1..batch + 1 {
            avg += storage[storage.len() - i];
        }
        avg / batch as f32
    }

    fn first(&self) -> f32 {
        let size = self.storage.lock().unwrap().len();
        let batch = if size / 2 > BATCH_SIZE {
            BATCH_SIZE
        } else {
            size / 2
        };
        let storage = self.storage.lock().unwrap();
        let mut avg = 0.0;
        for i in 0..batch {
            avg += storage[i];
        }
        avg / batch as f32
    }
}

impl MetricsRenderer for CustomRenderer {
    fn update_train(&mut self, state: MetricState) {
        match state {
            MetricState::Generic(x) => {
                let loss = x.serialize.replace(",1", "").parse::<f32>().unwrap();
                self.storage.lock().unwrap().push(loss);
                //println!("update_train -> {} - {}", x.name, loss);
            }
            MetricState::Numeric(_, _) => {}
        }
    }

    fn update_valid(&mut self, state: MetricState) {
        match state {
            MetricState::Generic(_) => {}
            MetricState::Numeric(_, _) => {}
        }
    }

    fn render_train(&mut self, _: TrainingProgress) {}

    fn render_valid(&mut self, _: TrainingProgress) {}
}

/// Test that Burn's Learner can successfully train a simple regression model
/// and that the loss decreases over epochs.
#[test]
fn test_learner_training_loop() {
    type MyBackend = Autodiff<NdArray>;
    let device = <MyBackend as Backend>::Device::default();
    let batcher = ToyRegressionBatcher::default();
    let train_dataset = ToyRegressionDataset::new(128);
    let valid_dataset = ToyRegressionDataset::new(32);

    let train_loader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(8)
        .num_workers(1)
        .build(train_dataset);

    let valid_loader = DataLoaderBuilder::new(batcher)
        .batch_size(8)
        .num_workers(1)
        .build(valid_dataset);

    let model = LinearModel::<MyBackend>::new(&device);
    let optimizer = SgdConfig::new().init();

    let learner = LearnerBuilder::new("./target/tmp/learner_test_1")
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .num_epochs(3)
        .summary()
        .build(model.clone(), optimizer, 0.01);

    let pre_prediction = model
        .forward(Tensor::from_data([[5]], &device))
        .into_scalar(); // 11 = 2 * 5 + 1
    let pre_loss = (pre_prediction - 11.0).powi(2) / 2.0;

    let output = learner.fit(train_loader, valid_loader);

    let post_prediction = output
        .forward(Tensor::from_data([[5]], &device))
        .into_scalar(); // 11 = 2 * 5 + 1
    let post_loss = (post_prediction - 11.0).powi(2) / 2.0;

    assert!(
        post_loss < pre_loss,
        "Training loss should decrease over epochs"
    );
}

/// Test that Burn's Learner logs and tracks Loss metric
#[test]
fn test_learner_with_loss_metric() {
    type MyBackend = Autodiff<NdArray>;
    let device = <MyBackend as Backend>::Device::default();
    let batcher = ToyRegressionBatcher::default();
    let train_dataset = ToyRegressionDataset::new(128);
    let valid_dataset = ToyRegressionDataset::new(32);

    let train_loader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(BATCH_SIZE)
        .num_workers(1)
        .build(train_dataset);

    let valid_loader = DataLoaderBuilder::new(batcher)
        .batch_size(BATCH_SIZE)
        .num_workers(1)
        .build(valid_dataset);

    let model = LinearModel::<MyBackend>::new(&device);
    let optimizer = SgdConfig::new().init();

    let renderer = CustomRenderer::new();
    let learner = LearnerBuilder::new("./target/tmp/learner_test_2")
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .num_epochs(10)
        .renderer(renderer.clone())
        .build(model, optimizer, 0.015);

    let _output = learner.fit(train_loader, valid_loader);
    //println!("{} - {}", renderer.first(), renderer.last())
    assert!(
        renderer.last() < renderer.first(),
        "MSE should decrease over epochs: first {}, last {}",
        renderer.first(),
        renderer.last()
    );
}
