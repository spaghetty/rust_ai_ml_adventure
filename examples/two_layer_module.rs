use burn::{
    backend::{Autodiff, NdArray},
    config::Config,
    module::Module,
    nn::{self, LinearConfig, Sigmoid},
    optim::{GradientsParams, Optimizer, SgdConfig},
    tensor::{Tensor, backend::Backend},
};

// Define our Autodiff backend type
type MyAutodiffBackend = Autodiff<NdArray>;

// --- 1. Two-Layer Network Module Definition ---

#[derive(Module, Debug)]
pub struct TwoLayerNet<B: Backend> {
    linear1: nn::Linear<B>,
    activation: Sigmoid,
    linear2: nn::Linear<B>,
}

#[derive(Config, Debug)]
pub struct TwoLayerNetConfig {
    input_features: usize,
    hidden_features: usize,
    output_features: usize,
}

impl TwoLayerNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TwoLayerNet<B> {
        TwoLayerNet {
            linear1: LinearConfig::new(self.input_features, self.hidden_features).init(device),
            activation: Sigmoid::new(),
            linear2: LinearConfig::new(self.hidden_features, self.output_features).init(device),
        }
    }
}

impl<B: Backend> TwoLayerNet<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);
        x
    }
}

// --- 5. Training Function for Two-Layer Network ---

pub fn train_two_layer_net() {
    let device = Default::default();
    let learning_rate = 0.03;
    let hidden_size = 10;

    let input_x = Tensor::<MyAutodiffBackend, 2>::from_data(
        [[1.0f32], [2.0], [5.0], [6.0], [7.0], [22.0]],
        &device,
    );
    let target_y = Tensor::<MyAutodiffBackend, 2>::from_data(
        [[5.0f32], [5.0], [5.0], [15.0], [15.0], [15.0]],
        &device,
    );

    let config = TwoLayerNetConfig::new(1, hidden_size, 1);
    let mut model = config.init(&device);

    let optimizer_config = SgdConfig::new();
    let mut optimizer = optimizer_config.init();

    println!("\n--- Training the Two-Layer Network (2000 steps) ---");

    for i in 0..50000 {
        let output_y = model.forward(input_x.clone());
        let loss = (output_y.clone() - target_y.clone())
            .powf_scalar(2.0)
            .mean();
        let gradients = loss.backward();

        model = optimizer.step(
            learning_rate.into(),
            model.clone(),
            GradientsParams::from_grads(gradients, &model),
        );

        println!("Step {}: Loss: {:.4}", i + 1, loss.to_data());
    }

    println!("--- Training Finished ---");
    let final_output = model.forward(input_x.clone());
    println!(
        "Final Model Output:\nInput: {:}\nTarget: {:}\nOutput: {:}",
        input_x.into_data(),
        target_y.into_data(),
        final_output.into_data()
    );

    println!("\n--- Two-Layer Net Quick Inference Test ---");
    let test_input_1 = Tensor::<MyAutodiffBackend, 2>::from_data([[3.0f32]], &device);
    let test_input_2 = Tensor::<MyAutodiffBackend, 2>::from_data([[8.0f32]], &device);
    let test_input_3 = Tensor::<MyAutodiffBackend, 2>::from_data([[5.0f32]], &device);
    let test_input_4 = Tensor::<MyAutodiffBackend, 2>::from_data([[22.0f32]], &device);
    println!(
        "Test [3.0] -> Pred: {:}",
        model.forward(test_input_1).into_data()
    );
    println!(
        "Test [8.0] -> Pred: {:}",
        model.forward(test_input_2).into_data()
    );
    println!(
        "Test [5.0] -> Pred: {:}",
        model.forward(test_input_3).into_data()
    );
    println!(
        "Test [22.0] -> Pred: {:}",
        model.forward(test_input_4).into_data()
    );
    println!("--- Inference Test Finished ---");
}

// --- 6. Main Function to Run Everything ---

fn main() {
    println!("===== Running Two-Layer Net Example =====");
    train_two_layer_net();
    println!("=======================================");
}
