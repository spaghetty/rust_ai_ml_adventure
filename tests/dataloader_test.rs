//! Tests demonstrating usage and constraints of Burn's DataLoader.
//!
//! This is meant as an educational guide for learning Burn's dataloader pattern.

use burn::data::dataloader::{ DataLoader, DataLoaderBuilder};
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{Tensor, backend::Backend, TensorData, Shape};
use burn::backend::NdArray;
use burn::data::dataset::Dataset;
use std::sync::Arc;
type MyBackend = NdArray;
// --- Custom Item & Batch Types (adapted from batcher_test.rs) ---
#[derive(Debug, Clone, PartialEq)]
struct FeatureItem {
    features: [f32; 2],
    label: f32,
}

#[derive(Debug, Clone)]
struct FeatureBatch<B: Backend> {
    features: Tensor<B, 2>,
    labels: Tensor<B, 1>,
}

#[derive(Debug, Clone, Default)]
struct FeatureBatcher;

impl<B: Backend> Batcher<B, FeatureItem, FeatureBatch<B>> for FeatureBatcher {
    fn batch(&self, items: Vec<FeatureItem>, device: &B::Device) -> FeatureBatch<B> {
        let n = items.len();
        let mut features = Vec::with_capacity(n * 2);
        let mut labels = Vec::with_capacity(n);
        for item in items {
            features.extend_from_slice(&item.features);
            labels.push(item.label);
        }
        FeatureBatch {
            features: Tensor::from_data(TensorData::new(features, Shape::new([n, 2])), device),
            labels: Tensor::from_data(TensorData::new(labels, Shape::new([n])), device),
        }
    }
}

// --- Minimal Dataset ---
#[derive(Debug, Clone)]
struct SimpleDataset {
    data: Vec<FeatureItem>,
}

impl SimpleDataset {
    fn new(size: usize) -> Self {
        let data = (0..size)
            .map(|i| FeatureItem { features: [i as f32, (i * 2) as f32], label: (i % 2) as f32 })
            .collect();
        Self { data }
    }
}

impl Dataset<FeatureItem> for SimpleDataset {
    fn get(&self, index: usize) -> Option<FeatureItem> {
        self.data.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.data.len()
    }
}

/// Test: Standard dataloader usage with single worker and shuffling.
#[test]
fn test_dataloader_usage() {

    let batcher = FeatureBatcher::default();
    let dataset = SimpleDataset::new(23);
    let batch_size = 5;
    let dataloader: Arc<dyn DataLoader<MyBackend, FeatureBatch<MyBackend>>> =
        DataLoaderBuilder::new(batcher)
            .batch_size(batch_size)
            .shuffle(123)
            .num_workers(1)
            .build(dataset);

    let mut n_batches = 0;
    for batch in dataloader.iter() {
        assert_eq!(batch.features.shape().dims[1], 2, "Feature dim should be 2");
        assert_eq!(batch.features.shape().dims[0], batch.labels.shape().dims[0], "Batch size match");
        assert!(batch.features.shape().dims[0] <= batch_size, "No batch larger than batch_size");
        n_batches += 1;
    }
    assert_eq!(n_batches, (23 + batch_size - 1) / batch_size, "Batch count");
}

/// Test: Dataloader with multiple workers (threaded loading).
#[test]
fn test_dataloader_multiworker() {
    
    let batcher = FeatureBatcher::default();
    let dataset = SimpleDataset::new(20);
    let batch_size = 4;
    let dataloader: Arc<dyn DataLoader<MyBackend, FeatureBatch<MyBackend>>> =
        DataLoaderBuilder::new(batcher)
            .batch_size(batch_size)
            .shuffle(42)
            .num_workers(2)
            .build(dataset);
    let mut total = 0;
    for batch in dataloader.iter() {
        total += batch.features.shape().dims[0];
    }
    assert_eq!(total, 20, "Should see all items with multiworker");
}

/// Test: Dataloader with empty dataset yields no batches.
#[test]
fn test_dataloader_empty_dataset() {
    type MyBackend = NdArray;
    let batcher = FeatureBatcher::default();
    let dataset = SimpleDataset::new(0);
    let batch_size = 8;
    let dataloader: Arc<dyn DataLoader<MyBackend, FeatureBatch<MyBackend>>> =
        DataLoaderBuilder::new(batcher)
            .batch_size(batch_size)
            .num_workers(1)
            .build(dataset);
    let mut seen = false;
    for _batch in dataloader.iter() {
        seen = true;
    }
    assert!(!seen, "No batches should be yielded for empty dataset");
}

/// Test: Batch size larger than dataset size yields one batch.
#[test]
fn test_dataloader_batch_larger_than_dataset() {

    let batcher = FeatureBatcher::default();
    let dataset = SimpleDataset::new(3);
    let batch_size = 10;
    let dataloader: Arc<dyn DataLoader<MyBackend, FeatureBatch<MyBackend>>> =
        DataLoaderBuilder::new(batcher)
            .batch_size(batch_size)
            .num_workers(1)
            .build(dataset);
    let mut count = 0;
    for batch in dataloader.iter() {
        assert_eq!(batch.features.shape().dims[0], 3, "Batch should contain all items");
        count += 1;
    }
    assert_eq!(count, 1, "Should yield a single batch");
}

/// This test demonstrates constraints: batcher and dataset must be Send + Sync, and batcher must be Clone.
/// If you try to use types that are not Send/Sync, compilation will fail.
/// This is enforced at compile time by Burn's DataLoaderBuilder.
#[test]
fn test_dataloader_constraints_compile() {
    // Uncommenting the below lines will cause a compile error if FeatureBatcher or SimpleDataset are not Send + Sync + Clone.
    // let dataloader = DataLoaderBuilder::new(FeatureBatcher)
    //     .batch_size(2)
    //     .num_workers(1)
    //     .build(SimpleDataset::new(3));
    assert!(true, "See comments for compile-time constraint demonstration.");
}

// --- Error Handling and Advanced Dataloader Feature Tests ---

/// Note on panic behavior in multithreaded DataLoader tests:
/// ---------------------------------------------------------
/// When using Burn's DataLoader with multiple workers (threads), if a panic occurs inside the
/// batcher or dataset (e.g., during `batch` or `get`), the panic actually happens in a worker thread,
/// not in the main test thread. The main thread receives a `RecvError` when it tries to collect
/// batches from the worker, which is caused by the worker thread panicking.
///
/// As a result, the panic message observed by the test harness is not the original message from
/// inside the worker (e.g., "Intentional panic in batcher!"), but rather a generic message like:
///     "called `Result::unwrap()` on an `Err` value: RecvError"
///
/// Therefore, we use `#[should_panic]` (without an expected message) to assert that *any* panic
/// occurs, regardless of its message. This is the idiomatic approach for testing panics in
/// multithreaded code with Rust and Burn.
///
/// For more details, see Burn's DataLoader implementation or Rust's documentation on panic
/// propagation in multithreaded environments.
///
/// Test: Dataloader panics if the batcher panics during batching.
#[test]
#[should_panic]
fn test_dataloader_batcher_panics() {
    #[derive(Debug, Clone, Default)]
    struct PanicBatcher;
    impl<B: Backend> Batcher<B, FeatureItem, FeatureBatch<B>> for PanicBatcher {
        fn batch(&self, _items: Vec<FeatureItem>, _device: &B::Device) -> FeatureBatch<B> {
            panic!("Intentional panic in batcher!");
        }
    }
    let dataset = SimpleDataset::new(5);
    let dataloader: Arc<dyn DataLoader<MyBackend, FeatureBatch<MyBackend>>> =
        DataLoaderBuilder::new(PanicBatcher)
            .batch_size(2)
            .num_workers(1)
            .build(dataset);
    for _batch in dataloader.iter() {
        // Should panic during batch creation
    }
}

/// Test: Dataloader panics if the dataset panics during get.
#[test]
#[should_panic]
fn test_dataloader_dataset_panics() {
    #[derive(Debug, Clone)]
    struct PanicDataset;
    impl Dataset<FeatureItem> for PanicDataset {
        fn get(&self, _index: usize) -> Option<FeatureItem> {
            panic!("Intentional panic in dataset!");
        }
        fn len(&self) -> usize { 5 }
    }
    let batcher = FeatureBatcher::default();
    let dataloader: Arc<dyn DataLoader<MyBackend, FeatureBatch<MyBackend>>> =
        DataLoaderBuilder::new(batcher)
            .batch_size(2)
            .num_workers(1)
            .build(PanicDataset);
    for _batch in dataloader.iter() {
        // Should panic during dataset get
    }
}

/// Test: Dataloader handles partial batches (last batch smaller than batch size).
#[test]
fn test_dataloader_partial_batch_handling() {
    let batcher = FeatureBatcher::default();
    let dataset = SimpleDataset::new(7);
    let batch_size = 3;
    let dataloader: Arc<dyn DataLoader<MyBackend, FeatureBatch<MyBackend>>> =
        DataLoaderBuilder::new(batcher)
            .batch_size(batch_size)
            .num_workers(1)
            .build(dataset);
    let mut batch_sizes = vec![];
    for batch in dataloader.iter() {
        batch_sizes.push(batch.features.shape().dims[0]);
    }
    // Should yield [3, 3, 1] (last batch is partial)
    assert_eq!(batch_sizes, vec![3, 3, 1], "Partial batch should be present");
}

// --- Advanced: Custom Sampler (if supported by Burn) ---
// If Burn supports custom samplers, add a test here. Otherwise, leave as a note.
// #[test]
// fn test_dataloader_custom_sampler() {
//     // Example: Use a custom sampler to yield indices in reverse order, or weighted sampling.
//     // Not implemented here, as Burn 0.17.0 may not yet support custom samplers directly.
//     // See Burn documentation for updates.
// }

// --- Notes ---
// - The DataLoader will automatically use threads if num_workers > 1, so your batcher and dataset must be thread-safe.
// - The batcher must be Clone, as it may be cloned for each worker.
// - The dataset must implement burn::data::dataset::Dataset.
// - This test is for learning and demonstration purposes.
