# Preliminaries

## Role of CUDA and Kernels

To understand CUDA, we first need to understand the problem. An Large Language Model (LLM) is essentially a massive mathematical structure composed of billions of parameters (numbers). To generate a single word, the computer must perform billions of calculations (mostly multiplications and additions) instantly.

- **CPU (Central Processing Unit)**: Your laptop's main processor is like a PhD mathematician. It is brilliant at complex, sequential logic (algebra, calculus, running an OS), but it works on one or a few problems at a time.

- **GPU (Graphics Processing Unit)**: A GPU is like an army of thousands of elementary school students. Individually, they aren't as "smart" as the PhD, but they can solve 10,000 simple multiplication problems simultaneously.

![cpu_gpu](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/gpu-devotes-more-transistors-to-data-processing.png)

LLMs, or Deep Learning Models in general, don't need complex logic for every calculation; they need massive volume. This is where CUDA comes in.

### What is CUDA? (The "Language")
CUDA (Compute Unified Device Architecture) is a software layer and programming model created by NVIDIA. It allows developers to "talk" to the GPU.

Without CUDA, GPUs are just graphics cards designed to render video game pixels independently in parallel. CUDA unlocks the GPU's ability to do General Purpose Computing (GPC).

It acts as a bridge. It lets you write code in languages like C++ or Python (via libraries like PyTorch) that sends instructions specifically to the GPU's "army" of cores (elementary school students).

CUDA organizes the thousands of GPU cores into a hierarchy (Grids, Blocks, and Threads) so they don't crash into each other while working in parallel. If the GPU is a massive factory full of workers, CUDA is the management system and the intercoms that allow the boss (the CPU) to give orders to the factory floor.

### What is a Kernel? (The "Task")
In the world of CUDA, a Kernel is a small, specialized function that you send to the GPU to be executed.

When you launch a kernel, you aren't running it once. You are telling the GPU: "Take this one set of instructions and have 10,000 threads execute it at the exact same time, but on different pieces of data."

#### How a Kernel Works:

1. Write the Kernel: A programmer writes a function, for example, multiply_numbers(x, y).

2. Launch: The CPU tells the GPU, "Run multiply_numbers on this list of 1 million numbers."

3. Parallel Execution: The GPU spins up 1 million threads. Every single thread runs the exact same Kernel code, but Thread #1 works on data point #1, and Thread #99 works on data point #99.

## Host (CPU) vs Device (GPU)
In CUDA terminology, we come across these terms a lot:

- **Host**: The CPU and its system memory. Controls the execution flow of the application.
- **Device**: The GPU and its on-board memory. Executes the parallel compute intensive tasks (kernels).
- **on-chip**: The GPU's on-board memory.
- **off-chip**: The CPU's system memory.

Data must be moved from Host to Device for processing, and results moved back from Device to Host.

## Memory Management
Managing memory on the GPU is similar to C `malloc`/`free` but with CUDA-specific APIs. It's useful to memorize the following functions:

- `cudaMalloc`: Allocates a single, physically contiguous block of memory on the device.
- `cudaFree`: Free memory on the device.
- `cudaMemcpy`: Copy memory between host and device.

And following flags for `cudaMemcpy`:

- `cudaMemcpyHostToDevice`: Copy from host to device.
- `cudaMemcpyDeviceToHost`: Copy from device to host.

```cpp
float *d_data;
int size = N * sizeof(float);

// Allocate memory on GPU
cudaMalloc((void**)&d_data, size);

// Copy data from Host to Device
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// Use d_data in kernels...

// Copy results back to Host
cudaMemcpy(h_result, d_data, size, cudaMemcpyDeviceToHost);

// Free GPU memory
cudaFree(d_data);
```

## Function Specifiers
To distinguish where a function runs and is called from in CUDA code:

- `__global__`: Runs on Device (GPU), called from Host (CPU). This defines a **Kernel**.
- `__device__`: Runs on Device, called from Device. Helper functions for kernels.
- `__host__`: Runs on Host, called from Host. (Default C++ behavior).

```cpp
__global__ void myKernel() {
    // Code running on GPU
}

__device__ int deviceFunc() {
    // Helper called by kernel
    return 0;
}
```

## Kernel Launch (fun stuff!)
To launch/run a kernel (function with `__global__`), we use the triple angle bracket syntax: `<<< ... >>>`.

```cpp
// Kernel launch
myKernel<<<gridDim, blockDim>>>(args...);
```
With gridDim and blockDim, we define the number of threads to be executed and how they are structured in the grid. **This is the most intersting part about parallel programming, in my opinion!** Think about your data, your calculation, and how you can adapt it to the GPU's architecture of threads and blocks. Beautiful!

![thread_hierarchy](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/grid-of-thread-blocks.png)

## Thread Hierarchy
Threads are formed in a hierarchy to map to the hardware structure the CUDA is built on. Dont't worry about the GPU hardware structure as we will come back to them in later sections.

1. **Thread: CUDA Core (Execution Unit)**. The smallest unit of execution. Each thread runs the same code but on different data.
2. **Warp: Warp Scheduler**. A group of 32 threads that execute the same instruction in a Single-Instruction, Multiple-Thread (SIMT) fashion.
3. **Block: Streaming Multiprocessor (SM)**. A collection of warps (up to 1024 threads total) that can cooperate via shared memory and synchronize with each other.
4. **Grid: GPU Device**. A collection of independent thread blocks that execute concurrently across all available SMs.

Built-in variables to locate a thread within the grid and block. 

- `gridDim`: Dimensions of the grid.
- `blockDim`: Dimensions of the block.
- `blockIdx`: Index of the current block within the grid.
- `threadIdx`: Index of the current thread within the block.

Each of these intrinsics is a 3-component vector with a `.x`, `.y`, and `.z` member. Dimensions not specified by a launch configuration will default to 1. `threadIdx` and `blockIdx` are zero indexed. That is, `threadIdx.x` will take on values from 0 up to and including `blockDim.x-1`. `.y` and `.z` operate the same in their respective dimensions.

Similarly, `blockIdx.x` will take on values from 0 up to and including `gridDim.x-1`, and the same for `.y` and `.z` dimensions, respectively.

These allow an individual thread to identify what work it should carry out.

**Global Index Calculation (1D):**
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

**Global Index Calculation (2D):**
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int idy = blockIdx.y * blockDim.y + threadIdx.y;
```

## Overall flow of a CUDA program

1.  **Initialize data on Host**: Allocate memory on the CPU and fill it with input data.
2.  **Allocate memory on Device**: Use `cudaMalloc` to reserve space on the GPU for inputs and outputs.
3.  **Transfer data (Host $\to$ Device)**: Use `cudaMemcpy` to move input data from CPU memory to GPU memory.
4.  **Launch Kernel**: Invoke the kernel function with specific grid and block dimensions to process data in parallel on the GPU.
5.  **Transfer results (Device $\to$ Host)**: Use `cudaMemcpy` to move the calculated results from GPU memory back to CPU memory.
6.  **Free memory**: Release the allocated memory on the GPU using `cudaFree`.

```cpp
// Kernel: The function that runs on the GPU
__global__ void myKernel(float* d_in, float* d_out) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Perform work
    d_out[idx] = d_in[idx] * 2.0f; 
}

int main() {
    // 1. Initialize data on Host (CPU)
    float* h_in = ...; 
    float* h_out = ...;

    // 2. Allocate memory on Device (GPU)
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // 3. Transfer data (Host -> Device)
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // 4. Launch Kernel
    // <<<Grid Dimensions, Block Dimensions>>>
    myKernel<<<blocks, threads>>>(d_in, d_out);

    // 5. Transfer results (Device -> Host)
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // 6. Free memory
    cudaFree(d_in);
    cudaFree(d_out);
}
```
Following this receipe, we will go through a series of examples to understand how to use CUDA to parallelize calculations and learn more stuff along the way.