use burn::data::dataset::Dataset;

use burn::backend::NdArray;
use burn::tensor::Tensor;

// Example 1: Simple Dataset Implementation
#[derive(Debug, Clone)]
struct SimpleDataset {
    data: Vec<f32>,
}

impl SimpleDataset {
    fn new(data: Vec<f32>) -> Self {
        Self { data }
    }
}

impl Dataset<f32> for SimpleDataset {
    fn get(&self, index: usize) -> Option<f32> {
        self.data.get(index).cloned()
    }
    
    fn len(&self) -> usize {
        self.data.len()
    }
}
    
#[test]
fn test_simple_dataset() {
    // Create a dataset with 5 elements
    let dataset = SimpleDataset::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // Test dataset length
    assert_eq!(dataset.len(), 5);
    
    // Test item retrieval
    assert_eq!(dataset.get(0), Some(1.0));
    assert_eq!(dataset.get(2), Some(3.0));
    assert_eq!(dataset.get(4), Some(5.0));
    assert_eq!(dataset.get(5), None); // Out of bounds
}

// Example 2: Dataset with Custom Item Type
#[derive(Debug, Clone, PartialEq)]
struct FeatureItem {
    features: [f32; 2],
    label: f32,
}

impl FeatureItem {
    /// Creates a new FeatureItem with the given features and label
    fn new(features: [f32; 2], label: f32) -> Self {
        Self { features, label }
    }

    /// Creates a new FeatureItem with random features and a binary label
    fn random() -> Self {
        use rand::prelude::*;
        let mut rng = rand::rng();
        let features = [rng.random_range(0.0..1.0), rng.random_range(0.0..1.0)];
        let label = if rng.random_bool(0.5) { 1.0 } else { 0.0 };
        Self::new(features, label)
    }
}

impl std::fmt::Display for FeatureItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Features: [{}, {}], Label: {:.1}", self.features[0], self.features[1], self.label)
    }
}

#[test]
fn test_feature_item() {
    // Test equality
    let item1 = FeatureItem::new([1.0, 2.0], 1.0);
    let item2 = FeatureItem::new([1.0, 2.0], 1.0);
    let item3 = FeatureItem::new([1.0, 2.0], 0.0);
    
    assert_eq!(item1, item2);
    assert_ne!(item1, item3);
    
    // Test random item creation
    let random_item = FeatureItem::random();
    assert!(random_item.label == 0.0 || random_item.label == 1.0);
    assert!(random_item.features[0] >= 0.0 && random_item.features[0] <= 1.0);
    assert!(random_item.features[1] >= 0.0 && random_item.features[1] <= 1.0);
    
    // Test display format
    let item = FeatureItem::new([0.5, 0.75], 1.0);
    assert_eq!(format!("{}", item), "Features: [0.5, 0.75], Label: 1.0");
}

#[derive(Debug, Clone)]
struct FeatureDataset {
    items: Vec<FeatureItem>,
}

impl FeatureDataset {
    fn new(features: Vec<[f32; 2]>, labels: Vec<f32>) -> Self {
        assert_eq!(features.len(), labels.len());
        let items = features.into_iter()
            .zip(labels.into_iter())
            .map(|(feat, label)| FeatureItem { features: feat, label })
            .collect();
        Self { items }
    }
}

impl Dataset<FeatureItem> for FeatureDataset {
    fn get(&self, index: usize) -> Option<FeatureItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[test]
fn test_feature_dataset() {
    // Test with empty dataset
    let empty_dataset = FeatureDataset::new(vec![], vec![]);
    assert_eq!(empty_dataset.len(), 0);
    assert_eq!(empty_dataset.get(0), None);

    // Test with single element
    let single_dataset = FeatureDataset::new(
        vec![[1.0, 2.0]],
        vec![1.0],
    );
    assert_eq!(single_dataset.len(), 1);
    let item = single_dataset.get(0).unwrap();
    assert_eq!(item.features, [1.0, 2.0]);
    assert_eq!(item.label, 1.0);

    // Test with multiple elements
    let dataset = FeatureDataset::new(
        vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        vec![1.0, 0.0, 1.0],
    );
    assert_eq!(dataset.len(), 3);

    // Test item retrieval
    let item1 = dataset.get(0).unwrap();
    assert_eq!(item1.features, [1.0, 2.0]);
    assert_eq!(item1.label, 1.0);

    let item2 = dataset.get(1).unwrap();
    assert_eq!(item2.features, [3.0, 4.0]);
    assert_eq!(item2.label, 0.0);

    let item3 = dataset.get(2).unwrap();
    assert_eq!(item3.features, [5.0, 6.0]);
    assert_eq!(item3.label, 1.0);

    assert_eq!(dataset.get(3), None); // Out of bounds

    // Test feature access
    let item = dataset.get(0).unwrap();
    assert_eq!(item.features.len(), 2);
    assert_eq!(item.features[0], 1.0);
    assert_eq!(item.features[1], 2.0);
}

// Example 3: Dataset with Tensor Output
#[derive(Debug, Clone)]
struct TensorDataset {
    data: Vec<[f32; 2]>,
}

impl TensorDataset {
    fn new(data: Vec<[f32; 2]>) -> Self {
        Self { data }
    }
}

impl Dataset<Tensor<NdArray, 2>> for TensorDataset {
    fn get(&self, index: usize) -> Option<Tensor<NdArray, 2>> {
        let Some(item) = self.data.get(index) else { return None };
        let device = Default::default();
        Some(Tensor::from_data([*item], &device))
    }
    
    fn len(&self) -> usize {
        self.data.len()
    }
}
    
#[test]
fn test_tensor_dataset() {
    // Create a dataset with tensor output
    let dataset = TensorDataset::new(vec![[1.0, 2.0], [3.0, 4.0]]);
    
    // Test dataset length
    assert_eq!(dataset.len(), 2);
    
    // Test tensor output
    let tensor = dataset.get(0).unwrap();
    assert_eq!(tensor.shape().dims, &[1, 2]);
    assert_eq!(tensor.to_data().to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
}
    
// Example 4: Custom Dataset with Validation
#[derive(Debug, Clone)]
struct ValidatedDataset {
    data: Vec<f32>,
}
    
impl ValidatedDataset {
    fn new(data: Vec<f32>) -> Self {
        // Validate data constraints
        assert!(data.iter().all(|&x| x >= 0.0 && x <= 1.0));
        Self { data }
    }
}
    
impl Dataset<f32> for ValidatedDataset {
    fn get(&self, index: usize) -> Option<f32> {
        self.data.get(index).cloned()
    }
    
    fn len(&self) -> usize {
        self.data.len()
    }
}
    
#[test]
fn test_validated_dataset() {
    // This should work
    let valid_dataset = ValidatedDataset::new(vec![0.5, 0.2, 0.8]);
    assert_eq!(valid_dataset.len(), 3);
    
    // This should panic (uncomment to test)
    // let invalid_dataset = ValidatedDataset::new(vec![1.5, 0.2, 0.8]);
}
