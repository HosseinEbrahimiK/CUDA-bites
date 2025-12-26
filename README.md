# CUDA-Kar

Welcome to **CUDA Kar** (means someone who knows CUDA! :D). This repository contains my personal journey digging into the LLM stack. In this repository we gonna takle simple to complex parallel programming concepts and break them down into small, digestible "bites" for myself and anyone else who wants to follow along.


We start from the basics Hello World of CUDA (vector addition) and aim to perform highly optimized implementations of the building blocks of modern LLMs:
- **Flash Attention**: Understanding memory hierarchy and tiling.
- **Matrix Multiplication (GEMM)**: The workhorse of deep learning.
- **Quantization**: Compressing models without losing intelligence.

## Prerequisites

- **C/C++ specifics**: We are writing low-level code here.
- **NVIDIA GPU**: The hardware to run the code (or a Colab instance).
- **CUDA Toolkit**: The compiler (nvcc) and drivers.

## Roadmap

- [x] **0. Preliminaries**: Understanding the CPU vs. GPU paradigm.
- [x] **1. Vector Addition**: Hello World of CUDA.
- [ ] **2. Matrix Multiplication**: Naive to Tiled implementation.
- [ ] **3. Reduction**: Parallelizing sum/max operations.
- [ ] **4. Softmax**: Numerical stability in parallel.

Let's get our hands dirty.
