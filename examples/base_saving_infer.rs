// In your inference script (e.g., infer_main.rs)
use burn::backend::NdArray; // Our base backend for inference
use burn::nn::Sigmoid;
use burn::prelude::*; // Often brings in Module, Config, Tensor, Backend, etc.
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};

// You MUST have the Model and ModelConfig definitions identical to your training script
// (Often, these are defined in a shared library/module)

// --- Placeholder for your Model and ModelConfig struct definitions ---
// (Copy them here from your training script or shared lib.rs)
// #[derive(Module, Debug)] pub struct Model<B: Backend> { /* ... */ }
// #[derive(Config)] pub struct ModelConfig { /* ... */ }
// impl ModelConfig { pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> { /* ... */ } }
// impl<B: Backend> Model<B> { pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> { /* ... */ } }
// --- End Placeholder ---

type MyInferenceBackend = NdArray; // Define the backend for inference

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

fn load_and_infer(model_path_str: &str, input_data: [f32; 2]) {
    let device = Default::default();

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load(model_path_str.into(), &device)
        .expect("Trained model should exist");

    // 1. Initialize a new model instance with the same architecture
    let config = ModelConfig::new(16); // Must match the config used for the saved model
    let mut model: Model<MyInferenceBackend> = config.init(&device).load_record(recorder);

    println!("Model loaded from {} ...", model_path_str);

    let input_tensor = Tensor::<MyInferenceBackend, 2>::from_data([input_data], &device);

    println!("Input data: {:?}", input_data);
    // Model forward pass (remember our model outputs logits)
    let output_logits = model.forward(input_tensor);
    let output_prob = Sigmoid::new().forward(output_logits); // Convert logits to probability
    let prediction: f32 = output_prob.squeeze::<1>(1).into_scalar(); // Get the actual f32 value

    println!("Output probability: {:.4}", prediction);
    if prediction > 0.5 {
        println!("Prediction: Point is INSIDE the circle.");
    } else {
        println!("Prediction: Point is OUTSIDE the circle.");
    }
}

fn main() {
    load_and_infer("../data/example6/model.mpk", [0.5, 0.5]); // Should be inside
    load_and_infer("../data/example6/model.mpk", [1.0, 1.0]); // Should be outside (on the edge or outside)
}
