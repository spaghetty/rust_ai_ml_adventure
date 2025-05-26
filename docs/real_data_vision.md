That's fantastic! Sharing your working code is the perfect way to move forward. I've reviewed the main.rs you provided. It's a complete example that trains a model to identify "zeroes" from the MNIST dataset and even includes CLI arguments for training and inference – very well done!

This is an excellent foundation for Post 7: Exploring Real-World Datasets - Tackling MNIST (and identifying zeros). We can walk through your code, explaining each part and the choices made.

Here's how we can structure the blog post, using your code as the centerpiece:

Post 7: AI and ML in Rust: Real-World Data - Identifying Zeros in MNIST!
Hey everyone! After setting up our data pipelines with Dataset and DataLoader in Post 5 and understanding how to save/load models in Post 6 (conceptually!), it's time to apply our knowledge to a real-world dataset. We're moving on from synthetic examples to the classic MNIST dataset of handwritten digits!

Our goal in this post: train a model to identify whether a given MNIST image is the digit "0" or not. This is a binary classification task. We'll build a simple Multi-Layer Perceptron (MLP) for this – we're saving Convolutional Neural Networks (CNNs) for Post 8, where we'll see how they excel at image tasks.

Let's dive into the code that makes this happen!

The MNIST Dataset in Burn
Burn conveniently provides the MnistDataset right out of the box, which makes loading the data incredibly easy.

Rust

// From your main.rs
use burn::data::dataset::vision::{MnistDataset, MnistItem};
// ... other imports ...

// Inside main or your training setup function:
// let dataset_train = MnistDataset::train();
// let dataset_test = MnistDataset::test();
Each MnistItem from this dataset contains an image (a [u8; 784] array representing 28x28 pixels) and a label (a u8 from 0 to 9).

Our Task: Binary Classification - "Is it a Zero?"
For our binary task, we need to transform the multi-class labels (0-9) into binary labels:

If the image is the digit 0, our target will be 1 (representing "is a zero").
If the image is any other digit (1 through 9), our target will be 0 (representing "not a zero").
This transformation will happen in our Batcher.

The MnistBatcher: Preparing Data for the Model
The Batcher is crucial. It takes individual MnistItems and prepares a MnistBatch containing Tensors ready for our model. This includes:

Converting the u8 image data to Float Tensors.
Reshaping the image into [1, 28, 28] (representing 1 channel, 28 height, 28 width), which then gets batched into [BatchSize, 28, 28].
Normalizing the pixel values.
Converting the u8 labels (0-9) into our binary Int target tensors (0 or 1).
Here's the MnistBatch struct:

Rust

// From your main.rs
use burn::tensor::{backend::Backend, Tensor, Int, Data}; // Make sure Data is imported for general use

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,       // [BatchSize, 28, 28]
    pub targets: Tensor<B, 1, Int>, // [BatchSize] (or [BatchSize, 1] then squeezed)
}
And the MnistBatcher implementation:

Rust

// From your main.rs
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::ElementConversion; // For .elem()

#[derive(Clone, Default)]
pub struct MnistBatcher {}

// Your code uses Batcher<B, MnistItem, MnistBatch<B>>
// This implies the Batcher trait might be generic over B in your Burn version
// or it's a custom/aliased Batcher trait.
// Standard Batcher<I, O> doesn't usually have B.
// For this post, we'll reflect your working signature.
impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images_to_cat: Vec<Tensor<B, 3>> = items
            .iter()
            .map(|item| {
                // Convert u8 array to Data<f32, 1>, then to Tensor<B, 1, Float>
                let data = Data::new(
                    item.image.map(|pixel| pixel as f32).to_vec(), // Convert u8 to f32
                    [WIDTH * HEIGHT] // Shape [784]
                ).convert::<B::FloatElem>(); // Ensure it's backend's float type

                Tensor::<B, 2>::from_data(data, device) // Tensor [784]
                    .reshape([1, WIDTH, HEIGHT]) // Reshape to [1, 28, 28] - making it 3D
                    // Normalize: scale between [0,1] and apply standard MNIST mean/std
                    .map(|tensor| ((tensor / 255.0) - 0.1307) / 0.3081)
            })
            .collect();

        let targets_to_cat: Vec<Tensor<B, 1, Int>> = items
            .iter()
            .map(|item| {
                let binary_label = if item.label == 0 { 1 } else { 0 };
                Tensor::<B, 1, Int>::from_data(
                    // Convert i64 binary_label to B::IntElem, then to Data
                    Data::new(vec![ (binary_label as i64).elem::<B::IntElem>() ], [1]),
                    device,
                )
            })
            .collect();

        let images = Tensor::cat(images_to_cat, 0); // Becomes [BatchSize, 1, 28, 28] if each item is [1,28,28]
                                                    // Or [BatchSize, 28, 28] if each item is [28,28]
                                                    // Your struct says Tensor<B,3> so [BatchSize, 28, 28] is expected.
                                                    // The .map(|tensor| tensor.reshape([1, WIDTH, HEIGHT]))
                                                    // actually makes it [BatchSize, 1, 28, 28] (4D).
                                                    // Let's adjust to match the struct's Tensor<B,3>
                                                    // by removing the channel dim in reshape or squeezing.
                                                    // Or the struct should be Tensor<B,4>.
                                                    // Given your model flattens from 3D, let's make images [BS, 28, 28]

        let targets = Tensor::cat(targets_to_cat, 0); // Becomes [BatchSize, 1]

        MnistBatch { images, targets }
    }
}
Self-correction on image batching based on your MnistBatch struct images: Tensor<B, 3> and model forward(input: Tensor<B, 3>):
The batcher should produce [BatchSize, Height, Width].

Rust

// Corrected Batcher's image processing part:
// From your main.rs
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{ElementConversion, Data, Tensor, Int, Backend}; // For .elem()
// ... other imports ...

#[derive(Clone, Default)]
pub struct MnistBatcher {}

impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images_for_cat: Vec<Tensor<B, 2>> = items // Each item will be [28, 28]
            .iter()
            .map(|item| {
                let data = Data::new(
                    item.image.map(|pixel| pixel as f32).to_vec(),
                    [WIDTH, HEIGHT] // Shape [28, 28]
                ).convert::<B::FloatElem>();

                Tensor::<B, 2>::from_data(data, device)
                    .map(|tensor| ((tensor / 255.0) - 0.1307) / 0.3081)
            })
            .collect();

        let targets_for_cat: Vec<Tensor<B, 1, Int>> = items
            .iter()
            .map(|item| {
                let binary_label = if item.label == 0 { 1 } else { 0 };
                Tensor::<B, 1, Int>::from_data(
                    Data::new(vec![ (binary_label as i64).elem::<B::IntElem>() ], [1]),
                    device,
                )
            })
            .collect();

        // Tensor::cat on Vec<Tensor<B,2>> (each [28,28]) along dim 0 gives [BatchSize, 28, 28] (Tensor<B,3>)
        let images = Tensor::cat(images_for_cat, 0);
        // Tensor::cat on Vec<Tensor<B,1,Int>> (each [1]) along dim 0 gives [BatchSize, 1] (Tensor<B,2,Int>)
        // We need targets to be Tensor<B,1,Int> as per MnistBatch struct. So we squeeze after cat.
        let targets = Tensor::cat(targets_for_cat, 0).squeeze(1);


        MnistBatch { images, targets }
    }
}
The Model (Simple MLP)
We're using a basic MLP with one hidden linear layer. Since your base_data.rs didn't use Relu in the forward pass, we'll reflect that here. The model takes the 3D image tensor [BatchSize, 28, 28], flattens it, and passes it through linear layers to produce 2 output logits (one for "not-zero", one for "is-zero").

Rust

// From your main.rs
use burn::module::Module;
use burn::nn::{self, LinearConfig};
use burn::config::Config;

#[derive(Module, Debug)]
struct ZeroModel<B: Backend> {
    linear: nn::Linear<B>, // Your code uses a single linear field, let's assume it means one layer for simplicity
                           // Or it's a typo and meant linear1, linear2. I'll base it on ModelConfig.
                           // ModelConfig implies a single linear layer taking flattened input.
                           // Let's adjust Model to match ModelConfig:
    // linear1: nn::Linear<B>,
    // linear2: nn::Linear<B>,
}

// Re-interpreting your ModelConfig and Model for a single effective linear layer
// from flattened_size to NUM_CLASSES.
// If you intended two layers, linear1 and linear2, we'd define them.
// Your ZeroModel struct only has `linear: nn::Linear<B>`.
// But your ModelConfig takes i and o.

// Let's assume your ModelConfig is for a *single* linear layer
// for this "ZeroModel" which simplifies to one transformation.
// If it was linear1 -> relu -> linear2, the structure would be different.
// Given your code: `linear: nn::LinearConfig::new(self.i, self.o).init(device)`
// `self.i` is `WIDTH * HEIGHT`, `self.o` is `NUM_CLASSES`. This points to one layer.

impl<B: Backend> ZeroModel<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let batch_size = input.shape().dims[0];
        let flattened_size = input.shape().dims[1] * input.shape().dims[2]; // 28 * 28 = 784

        let input_flattened = input.reshape([batch_size, flattened_size]);
        self.linear.forward(input_flattened) // Output [BatchSize, NUM_CLASSES]
    }

    // forward_classification uses CrossEntropyLoss, which is appropriate
    // for multi-class (or binary treated as multi-class with 2 outputs)
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>, // Expects [BatchSize] Int targets
    ) -> ClassificationOutput<B> {
        use burn::nn::loss::CrossEntropyLossConfig; // Moved use here for clarity

        let output_logits = self.forward(images); // [BatchSize, NUM_CLASSES]
        let loss = CrossEntropyLossConfig::new()
            .init(&output_logits.device())
            .forward(output_logits.clone(), targets.clone());

        ClassificationOutput::new(loss, output_logits, targets)
    }
}

#[derive(Config, Debug)] // From your main.rs
struct ModelConfig {
    i: usize, // Input features to the linear layer (784)
    o: usize, // Output features (NUM_CLASSES = 2)
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ZeroModel<B> {
        ZeroModel {
            linear: nn::LinearConfig::new(self.i, self.o).init(device),
        }
    }
}
Because our model outputs 2 logits (one for class 0 "not-zero", one for class 1 "is-zero"), we use CrossEntropyLoss. This loss function expects Int targets representing the class index (0 or 1 in our case), which our batcher provides.

TrainStep and ValidStep
These tell the Learner how to process a batch for training and validation. Your code uses a specific TrainOutput::new(self, item.loss.backward(), item) signature.

Rust

// From your main.rs
use burn::train::{ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep};
use burn::tensor::backend::AutodiffBackend; // For AutodiffBackend

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for ZeroModel<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        // forward_classification already calculates loss and gives ClassificationOutput
        let classification_item = self.forward_classification(batch.images, batch.targets);

        // The TrainOutput::new signature in your code is (model, grads, output_struct)
        // item.loss.backward() returns Gradients
        TrainOutput::new(self, classification_item.loss.backward(), classification_item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for ZeroModel<B> {
    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}
The Learner Setup
Now we assemble everything with LearnerBuilder. Your code includes AccuracyMetric (which works well with CrossEntropyLoss and Int targets) and LossMetric.

Rust

// From your main.rs, inside your training block
use burn::train::metric::{AccuracyMetric, LossMetric};
// ... other imports for SgdConfig, MomentumConfig, DataLoaderBuilder, etc.

// let config = TrainingConfig::new(...); // From your code
// let model_config = ModelConfig::new(WIDTH * HEIGHT, NUM_CLASSES.into());
// let model: ZeroModel<MyBackend> = model_config.init(&device);
// let batcher_train = MnistBatcher::default();
// let dataloader_train = DataLoaderBuilder::new(batcher_train) /* ... */ .build(dataset_train);
// let batcher_test = MnistBatcher::default();
// let dataloader_test = DataLoaderBuilder::new(batcher_test) /* ... */ .build(dataset_test);

// let learner = LearnerBuilder::new(ARTIFACT_DIR)
//     .metric_train_numeric(AccuracyMetric::new()) // Using built-in AccuracyMetric
//     .metric_valid_numeric(AccuracyMetric::new())
//     .metric_train_numeric(LossMetric::new())
//     .metric_valid_numeric(LossMetric::new())
//     .with_file_checkpointer(CompactRecorder::new())
//     .devices(vec![device.clone()])
//     .num_epochs(config.num_epochs)
//     .summary()
//     .build(model, config.optimizer.init(), config.learning_rate);

// let model_trained = learner.fit(dataloader_train, dataloader_test);
(This is a snippet; the full main function in your code shows the complete setup)

Saving, Inferring, and CLI
Your main.rs also includes:

Saving the model: model_trained.save_file(...) using CompactRecorder.
An infer function: Loads the model and predicts on a single MnistItem. It uses output.argmax(1) to get the predicted class from the 2 logits.
CLI arguments: Using clap to switch between training and inference modes.
These are excellent additions that make the example very complete!

My Journey & What's Next
Working with the MNIST dataset was a great step up! The main learning points were:

Leveraging Burn's built-in MnistDataset.
Carefully structuring the MnistBatcher to handle image normalization, reshaping, and importantly, the specific binary label conversion (0 -> class 1, others -> class 0) with Int targets.
Using CrossEntropyLoss because our MLP model outputs two logits for our binary ("is-zero" / "not-zero") problem.
Seeing how AccuracyMetric works with this setup.
The specific TrainOutput::new(model, grads, output_struct) pattern required by the Learner in this Burn version.
While our MLP can identify zeros, it's not the most powerful architecture for image tasks. Next up, in Post 8, we'll dive into Convolutional Neural Networks (CNNs) and see how they are specifically designed to understand spatial information in images, which should give us a significant performance boost!
