# 🔥 Rust Deep Learning with Burn: From Zero to MNIST 🔥
----
[![Tests](https://github.com/spaghetty/rust_ai_ml_adventure/actions/workflows/rust.yml/badge.svg)](https://github.com/spaghetty/rust_ai_ml_adventure/actions/workflows/rust.yml)

----

![project image](./docs/rust-burn.jpg?raw=true)

This repository contains the code and journey of learning and applying the **Burn deep learning framework** in **Rust**. It's a live document of building understanding from the ground up, culminating in practical examples like MNIST digit recognition.

## Our Goal:

To clearly demonstrate and document the process of building neural networks with Burn, covering:

✓ Tensors & Automatic Differentiation

✓ Modular Network Design

✓ Data Loading & Batching Pipelines

✓ Training Loops with the Burn `Learner`

✓ Saving, Loading, and Inferring with Models

## Why This Matters:

* **Showcasing Burn:** A powerful, modern deep learning toolkit for Rust.
* **Rust for Speed & Safety:** Leveraging Rust's strengths in the AI/ML domain.
* **Practical Learning Resource:** Provides working examples and a relatable learning path for newcomers.

Follow along, contribute, or use the examples to kickstart your own AI projects in Rust!

### Best way to learn

If you are curious, or want to try something please contribute to this project.
You can contribute with:
 * pull requests for improvement
 * issue for unclear part or improvement ideas
 * requests for missing area to explore.

Working together is the best aproach to learn and grow

### Getting Started (Example)

To run the examples (once you have Rust and Cargo installed):

1.  Clone the repository:
    ```bash
    git clone git@github.com:spaghetty/rust_ai_ml_adventure.git
    cd rust_ai_ml_adventure
    ```
2.  To run the training for an example (e.g., from the MNIST project):
    ```bash
    cargo run --example <name_of_the_example> ##(eg. base_tensor)
    ```
3.  To run inference with a trained model (specific command might vary per example):
    ```bash
    cargo test
    cargo test test_tensor ##(eg. single file)
    ```


### Blog Post Series

*[my medium blog](https://medium.com/@spaghetty)*

* [**Post 1:**](https://medium.com/@spaghetty/rust-meets-ai-e6e754ba273d) - [Intro & Tensors](./docs/01-tensor.md)
* [**Post 2:**](https://medium.com/@spaghetty/2-ai-ml-in-rust-unleashing-autodiff-gradients-explained-41e7a2cec94d) - [Autodiff](./docs/02-autodiff.md)
* ... and more to come!
