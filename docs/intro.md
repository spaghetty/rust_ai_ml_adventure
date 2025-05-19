# Post 1: AI and ML in Rust: A Wild Idea (That Might Just Work!)
## My Journey

Alright team, buckle up! I've been having a blast playing with __Rust__ lately, and while exploring the world of __Artificial Intelligence__ and __Machine Learning__, I stumbled upon something wild: there's a whole, surprisingly active AI/ML ecosystem in _Rust_! Crates like __RIG__, __Polars__, __linfa__, __smartcore__, and __Burn__ are out there!

This discovery, fueled by my curiosity for Rust and AI/ML, sparked a slightly impulsive, definitely ambitious project. I'm diving into AI/ML, I'm sticking with Rust because I like it, and I'm going to figure it out step by step. And as I do, I want to share my findings with anyone else who's curious enough to join this adventure!

In this first post, I'll just set the scene for the journey.

## Let's jump in!Rust for AI/ML: Because Why Not? (And Also, It's Pretty Cool)

Okay, okay, it's not just because I'm having a Rust phase! At least I suppose, considering all the cool project around rust and AI something should be there and I'll try to discover it.

The ecosystem is growing! Beyond Burn, there are very active crates like:
* __Polars__ for data manipulation that is currently challenging __Pandas__ that is a de facto standard in this field
* __NdArray__ & __Nalgebra__ foundational numerical libraries
* __RIG__ for LLMs integration and agents implementation

and I'm sure I'll discover many more

## Burn: My Chosen Tool for Neural Network Shenanigans

AI/ML is a big field, but a huge part of it today is neural networks. And for that, seems that we have the right framework.

__Burn__ is a Deep Learning framework written in Rust, for Rust. It's designed to be flexible and performant.

* __Backend Swap Party__: Burn lets you write your model code once and run it on different hardware (CPU, various GPUs).
* __Feels Like Home__ (If Home is Made of Code): Burn uses Rust's features and patterns, so it integrates nicely.
* __Eager and Ready__: Burn runs operations as you write them, which can make building and debugging neural networks feel more direct.

# Burn vs. Python Frameworks: Picking My Kind of Fun

Let's be real: Python frameworks are the standard for a reason. They're easy to start with and have tons of resources.

* Python Frameworks: The comfy, well-worn path.
* Burn (and Rust for AI/ML): The exciting, slightly overgrown trail. Requires more effort, but the feeling of accomplishment (and the performance!) is amazing.

I'm taking the trail. It fits my mood.

# Getting Started: Gearing Up for the Ascent

Ready to join this slightly impulsive expedition? Here some basic for set up a Rust environment

Install Rust (If You Haven't Already):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
