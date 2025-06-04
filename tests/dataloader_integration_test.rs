//! Integration test: DataLoader + Model + Loss + Optimizer (Burn)
//!
//! This test demonstrates a minimal training loop using Burn's DataLoader, a simple model,
//! loss function, and optimizer. The goal is to show end-to-end integration and verify
//! that the optimizer updates model parameters.

use burn::backend::{Autodiff, NdArray};
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData, Shape};
use burn::nn::loss::MseLoss;
use std::sync::Arc;

// --- Dummy dataset and batcher for regression ---
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
        // y = 2x + 1, with no noise
        let data = (0..size)
            .map(|i| {
                let x = i as f32 / (size as f32 - 1.0);
                ToyRegressionItem { x: [x], y: 2.0 * x + 1.0 }
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

#[test]
fn test_dataloader_model_optimizer_integration() {
    type MyBackend = Autodiff<NdArray>;
    let device = <MyBackend as Backend>::Device::default();
    let batcher = ToyRegressionBatcher::default();
    let dataset = ToyRegressionDataset::new(128);
    let dataloader: Arc<dyn DataLoader<MyBackend, ToyRegressionBatch<MyBackend>>> =
        DataLoaderBuilder::new(batcher)
            .batch_size(8)
            .num_workers(1)
            .build(dataset);

    let mut model = LinearModel::<MyBackend>::new(&device);
    let mut optimizer = SgdConfig::new().init();
    let loss_fn = MseLoss::new();

    // Record initial loss
    let mut total_loss_before = 0.0;
    let mut n_batches = 0;
    for batch in dataloader.iter() {
        let y_pred = model.forward(batch.x.clone());
        let loss = loss_fn.forward(y_pred.clone(), batch.y.clone().unsqueeze_dim(1), burn::nn::loss::Reduction::Mean);
        total_loss_before += loss.to_data().to_vec::<f32>().unwrap()[0];
        n_batches += 1;
    }
    let avg_loss_before = total_loss_before / n_batches as f32;

    // One epoch of training
    for batch in dataloader.iter() {
        let y_pred = model.forward(batch.x.clone());
        let loss = loss_fn.forward(y_pred.clone(), batch.y.clone().unsqueeze_dim(1), burn::nn::loss::Reduction::Mean);
        let grads = loss.backward();
        model = optimizer.step(0.001, model.clone(), GradientsParams::from_grads(grads, &model));
    }

    // Record final loss
    let mut total_loss_after = 0.0;
    let mut n_batches = 0;
    for batch in dataloader.iter() {
        let y_pred = model.forward(batch.x.clone());
        let loss = loss_fn.forward(y_pred.clone(), batch.y.clone().unsqueeze_dim(1), burn::nn::loss::Reduction::Mean);
        total_loss_after += loss.to_data().to_vec::<f32>().unwrap()[0];
        n_batches += 1;
    }
    let avg_loss_after = total_loss_after / n_batches as f32;

    println!("Loss before: {avg_loss_before}, after: {avg_loss_after}");
    assert!(avg_loss_after < avg_loss_before, "Loss should decrease after optimizer step");
}
