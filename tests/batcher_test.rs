use burn::data::{
    dataloader::batcher::Batcher,
    // dataset::Dataset,
};
use burn::tensor::{Tensor, backend::Backend, TensorData, DType};
use burn::backend::{NdArray};
use burn::prelude::*;

// Custom Item Types
#[derive(Debug, Clone, PartialEq)]
struct FeatureItem {
    features: [f32; 2],
    label: f32,
}

impl FeatureItem {
    fn new(features: [f32; 2], label: f32) -> Self {
        Self { features, label }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct CustomItem {
    feature: f32,
    label: f32,
}

impl CustomItem {
    fn new(feature: f32, label: f32) -> Self {
        Self { feature, label }
    }
}

#[derive(Debug)]
struct CustomProcessed {
    feature: f32,
    label: f32,
}

// Batch Types
#[derive(Debug)]
struct FeatureBatch<B: Backend> {
    features: Tensor<B, 2>,
    labels: Tensor<B, 1>,
}

#[derive(Debug)]
struct CustomBatch<B: Backend> {
    features: Tensor<B, 1>,
    labels: Tensor<B, 1>,
}

// Example 1: Basic Batcher Implementation
#[derive(Debug, Clone, Default)]
struct SimpleBatcher;

impl<B: Backend> Batcher<B, f32, Tensor<B, 1>> for SimpleBatcher {
    fn batch(&self, items: Vec<f32>, device: &B::Device) -> Tensor<B, 1> {
        let len = items.clone().len();
        let data = TensorData::new(items, Shape::new([len]));
        // Create tensor on the specified device
        Tensor::from_data(data, device)
    }
}

#[test]
fn test_simple_batcher() {
    type MyBackend = NdArray;
    let batcher = SimpleBatcher::default(); // Use default instead of empty struct
    let device = Default::default();
    
    // Test with empty batch
    let empty_batch: Tensor<MyBackend, 1> = batcher.batch(vec![], &device);
    assert_eq!(empty_batch.shape().dims, &[0]);
    
    // Test with single item
    let single_batch: Tensor<MyBackend, 1> = batcher.batch(vec![1.0], &device);
    assert_eq!(single_batch.shape().dims, &[1]);
    assert_eq!(single_batch.to_data(), TensorData::new(vec![1.0], Shape::new([1])).convert_dtype(DType::F32), "Single item batch");
    
    // Test with multiple items
    let batch: Tensor<MyBackend, 1> = batcher.batch(vec![1.0, 2.0, 3.0], &device);
    assert_eq!(batch.shape().dims, &[3]);
    assert_eq!(batch.to_data(), TensorData::new(vec![1.0, 2.0, 3.0], Shape::new([3])).convert_dtype(DType::F32), "Multiple items batch");
    
    // Test with negative numbers
    let batch: Tensor<MyBackend, 1> = batcher.batch(vec![-1.0, -2.0, -3.0], &device);
    assert_eq!(batch.shape().dims, &[3]);
    assert_eq!(batch.to_data(), TensorData::new(vec![-1.0, -2.0, -3.0], Shape::new([3])).convert_dtype(DType::F32), "Negative numbers batch");
}

// Example 2: Batcher with Multiple Features
#[derive(Debug, Clone, Default)]
struct FeatureBatcher;

impl<B: Backend> Batcher<B, FeatureItem, FeatureBatch<B>> for FeatureBatcher {
    fn batch(&self, items: Vec<FeatureItem>, device: &B::Device) -> FeatureBatch<B> {
        // Extract features and labels
        let (features, labels): (Vec<_>, Vec<_>) = items.clone().into_iter()
            .map(|item| (item.features, item.label))
            .unzip();
            
        // Convert to tensors with proper shapes
        let flat_features: Vec<f32> = features.into_iter().flat_map(|f| f.into_iter()).collect();
        let features_tensor = Tensor::from_data(
            TensorData::new(flat_features, Shape::new([items.clone().len(), 2])),
            device
        );
        let labels_tensor = Tensor::from_data(
            TensorData::new(labels, Shape::new([items.clone().len()])),
            device
        );
        
        FeatureBatch {
            features: features_tensor,
            labels: labels_tensor,
        }
    }
}

#[test]
fn test_feature_batcher() {
    type MyBackend = NdArray;
    let batcher = FeatureBatcher::default();
    let device = Default::default();    
    // Create some test data
    let items = vec![
        FeatureItem::new([1.0, 2.0], 1.0),
        FeatureItem::new([3.0, 4.0], 0.0),
    ];
    
    // Create batch
    let batch: FeatureBatch<MyBackend> = batcher.batch(items.clone(), &device);
    
    // Verify features
    assert_eq!(batch.features.shape().dims, &[2, 2]);
    assert_eq!(batch.features.to_data(), TensorData::new::<f32, _>(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2])));
    
    // Verify labels
    assert_eq!(batch.labels.shape().dims, &[2]);
    assert_eq!(batch.labels.to_data(), TensorData::new::<f32, _>(vec![1.0, 0.0], Shape::new([2])));
    
    // Test empty batch
    let empty_batch: FeatureBatch<MyBackend> = batcher.batch(vec![], &device);
    assert_eq!(empty_batch.features.shape().dims, &[0, 2]);
    assert_eq!(empty_batch.labels.shape().dims, &[0]);
    
    // Test batch with different features
    let items = vec![
        FeatureItem::new([0.5, -0.5], 0.0),
        FeatureItem::new([-1.0, 1.0], 1.0),
    ];
    let batch: FeatureBatch<MyBackend> = batcher.batch(items.clone(), &device);
    assert_eq!(batch.features.to_data(), TensorData::new::<f32, _>(vec![0.5, -0.5, -1.0, 1.0], Shape::new([2, 2])));
    assert_eq!(batch.labels.to_data(), TensorData::new::<f32, _>(vec![0.0, 1.0], Shape::new([2])));
}


// Example 3: Batcher with Custom Processing
#[derive(Debug, Clone, Default)]
struct NormalizingBatcher;

impl<B: Backend> Batcher<B, f32, Tensor<B, 1>> for NormalizingBatcher {
    fn batch(&self, items: Vec<f32>, device: &B::Device) -> Tensor<B, 1> {
        if items.is_empty() {
            return Tensor::from_data(TensorData::new(vec![] as Vec<f32>, Shape::new([0])), device);
        }

        // Calculate mean and standard deviation
        let mean = items.iter().sum::<f32>() / items.len() as f32;
        let variance = items.iter()
            .map(|x: &f32| (x - mean).powi(2))
            .sum::<f32>()
            / items.len() as f32;
        let std_dev = variance.sqrt();
        
        // Handle zero standard deviation case
        if std_dev == 0.0 {
            return Tensor::from_data(TensorData::new(vec![0.0; items.len()], Shape::new([items.len()])), device);
        }
        
        // Normalize items
        let normalized: Vec<f32> = items.clone().into_iter()
            .map(|x: f32| (x - mean) / std_dev)
            .collect();
            
        // Create tensor with proper shape
        Tensor::from_data(TensorData::new(normalized, Shape::new([items.len()])), device)
    }
}

#[test]
fn test_normalizing_batcher() {
    type MyBackend = NdArray;
    let batcher = NormalizingBatcher::default();
    let device = Default::default(); 
    
    // Test empty batch
    let empty_batch: Tensor<MyBackend, 1> = batcher.batch(vec![], &device);
    assert_eq!(empty_batch.shape().dims, &[0]);
    
    // Test with identical values (should handle zero std deviation)
    let identical_batch: Tensor<MyBackend, 1> = batcher.batch(vec![1.0, 1.0, 1.0], &device);
    assert_eq!(identical_batch.shape().dims, &[3]);
    assert_eq!(identical_batch.to_data(), TensorData::new::<f32, _>(vec![0.0, 0.0, 0.0], Shape::new([3])));
    
    // Test with positive numbers
    let batch: Tensor<MyBackend, 1> = batcher.batch(vec![1.0, 2.0, 3.0, 4.0], &device);
    let data = batch.to_data();
    
    // Mean should be 0 (within small tolerance)
    assert!((data.iter::<f32>().sum::<f32>() / data.shape[0] as f32).abs() < 1e-6, "Mean should be 0");
    
    // Standard deviation should be 1 (within small tolerance)
    let std_dev = (data.iter()
        .map(|x: f32| x.powi(2))
        .sum::<f32>()
        / data.shape[0] as f32)
        .sqrt();
    assert!((std_dev - 1.0).abs() < 1e-6, "Standard deviation should be 1");
    
    // Test with negative numbers
    let batch: Tensor<MyBackend, 1> = batcher.batch(vec![-1.0, -2.0, -3.0, -4.0], &device);
    let data = batch.to_data();
    assert!((data.iter::<f32>().sum::<f32>() / data.shape[0] as f32).abs() < 1e-6, "Mean should be 0");
    assert!((std_dev - 1.0).abs() < 1e-6);
}

// Example 4: Batcher with Padding
#[derive(Debug, Clone, Default)]
struct PaddingBatcher;

impl<B: Backend> Batcher<B, Vec<f32>, Tensor<B, 2>> for PaddingBatcher {
    fn batch(&self, items: Vec<Vec<f32>>, device: &B::Device) -> Tensor<B, 2> {
        if items.is_empty() {
            return Tensor::from_data(TensorData::new::<f32, _>(vec![], Shape::new([0, 0])), device);
        }

        // Find maximum length
        //let num_items = items.clone().len();
        let max_len = items.clone().iter().map(|v| v.len()).max().unwrap_or(0);
        
        // Pad sequences
        let padded: Vec<Vec<f32>> = items.clone().into_iter()
            .map(|seq| {
                let mut padded_seq = seq;
                while padded_seq.len() < max_len {
                    padded_seq.push(0.0); // Pad with zeros
                }
                padded_seq
            })
            .collect();
            
        // Create tensor with proper shape
        Tensor::from_data(
            TensorData::new::<f32, _>(
                padded.into_iter()
                    .flat_map(|v| v.into_iter())
                    .collect::<Vec<f32>>(),
                Shape::new([items.len(), max_len])
            ),
            device
        )
    }
}

#[test]
fn test_padding_batcher() {
    type MyBackend = NdArray;
    let device = Default::default(); 
    let batcher = PaddingBatcher::default();
    
    // Create batch with sequences of different lengths
    let batch: Tensor<MyBackend, 2> = batcher.batch(
        vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0, 5.0],
            vec![6.0],
        ],
        &device,
    );
    
    // Verify padding
    assert_eq!(batch.shape().dims, &[3, 3]); // 3 sequences, max length 3
    assert_eq!(batch.to_data(), TensorData::new::<f32, _>(vec![
        1.0, 2.0, 0.0, // First sequence (padded with 0)
        3.0, 4.0, 5.0, // Second sequence
        6.0, 0.0, 0.0, // Third sequence (padded with 0)
    ], Shape::new([3, 3])));
    
    // Test empty batch
    let empty_batch: Tensor<MyBackend, 2> = batcher.batch(vec![], &device);
    assert_eq!(empty_batch.shape().dims, &[0, 0]);
}

// Example 5: Batcher with Custom Item Type
#[derive(Debug, Clone, Default)]
struct CustomBatcher;

impl<B: Backend> Batcher<B, CustomItem, CustomBatch<B>> for CustomBatcher {
    fn batch(&self, items: Vec<CustomItem>, device: &B::Device) -> CustomBatch<B> {
        // Process items
        let processed: Vec<CustomProcessed> = items.iter()
            .map(|item| CustomProcessed {
                feature: item.feature * 2.0,
                label: item.label + 1.0,
            })
            .collect();
            
        // Create tensors with proper shapes
        CustomBatch {
            features: Tensor::from_data(
                TensorData::new(processed.iter().map(|p| p.feature).collect::<Vec<_>>(), Shape::new([items.len()])),
                device
            ),
            labels: Tensor::from_data(
                TensorData::new(processed.iter().map(|p| p.label).collect::<Vec<_>>(), Shape::new([items.len()])),
                device
            ),
        }
    }
}

#[test]
fn test_custom_batcher() {
    type MyBackend = NdArray;
    let device = Default::default(); 
    let batcher = CustomBatcher::default();

    
    // Create batch
    let batch: CustomBatch<MyBackend> = batcher.batch(
        vec![
            CustomItem::new(1.0, 0.0),
            CustomItem::new(2.0, 1.0),
        ],
        &device,
    );
    
    // Verify features (should be doubled)
    assert_eq!(batch.features.shape().dims, &[2]);
    assert_eq!(batch.features.to_data(), TensorData::new::<f32, _>(vec![2.0, 4.0], Shape::new([2])));
    
    // Verify labels (should be incremented by 1)
    assert_eq!(batch.labels.shape().dims, &[2]);
    assert_eq!(batch.labels.to_data(), TensorData::new::<f32, _>(vec![1.0, 2.0], Shape::new([2])));
    
    // Test empty batch
    let empty_batch: CustomBatch<MyBackend> = batcher.batch(vec![], &device);
    assert_eq!(empty_batch.features.shape().dims, &[0]);
    assert_eq!(empty_batch.labels.shape().dims, &[0]);
}