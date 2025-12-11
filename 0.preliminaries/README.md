# Preliminaries

## Role of CUDA and Kernels

To understand CUDA, we first need to understand the problem. An Large Language Model (LLM) is essentially a massive mathematical structure composed of billions of parameters (numbers). To generate a single word, the computer must perform billions of calculations (mostly multiplications and additions) instantly.

- **CPU (Central Processing Unit)**: Your laptop's main processor is like a PhD mathematician. It is brilliant at complex, sequential logic (algebra, calculus, running an OS), but it works on one or a few problems at a time.

- **GPU (Graphics Processing Unit)**: A GPU is like an army of thousands of elementary school students. Individually, they aren't as "smart" as the PhD, but they can solve 10,000 simple multiplication problems simultaneously.

LLMs don't need complex logic for every calculation; they need sheer volume. This is where CUDA comes in.

### What is CUDA? (The "Language")
CUDA (Compute Unified Device Architecture) is a software layer and programming model created by NVIDIA. It allows developers to "talk" to the GPU.

Without CUDA, GPUs are just graphics cards designed to render video game pixels independently in parallel. CUDA unlocks the GPU's ability to do General Purpose computing.

It acts as a bridge. It lets you write code in languages like C++ or Python (via libraries like PyTorch) that sends instructions specifically to the GPU's "army" of cores.

CUDA organizes the thousands of GPU cores into a hierarchy (Grids, Blocks, and Threads) so they don't crash into each other while working.

If the GPU is a massive factory full of workers, CUDA is the management system and the intercoms that allow the boss (the CPU) to give orders to the factory floor.

### What is a Kernel? (The "Task")
In the world of CUDA, a Kernel is a small, specialized function that you send to the GPU to be executed.

When you launch a kernel, you aren't running it once. You are telling the GPU: "Take this one set of instructions and have 10,000 threads execute it at the exact same time, but on different pieces of data."

#### How a Kernel Works:

1. Write the Kernel: A programmer writes a function, for example, multiply_numbers(x, y).

2. Launch: The CPU tells the GPU, "Run multiply_numbers on this list of 1 million numbers."

3. Parallel Execution: The GPU spins up 1 million threads. Every single thread runs the exact same Kernel code, but Thread #1 works on data point #1, and Thread #99 works on data point #99.

## Host (CPU) vs Device (GPU)
In CUDA terminology, we came across two terms a lot:

- **Host**: The CPU and its system memory. Controls the execution flow of the application.
- **Device**: The GPU and its on-board memory. Executes the parallel compute intensive tasks (kernels).

Data must be moved from Host to Device for processing, and results moved back from Device to Host.

## Memory Management
Managing memory on the GPU is similar to C `malloc`/`free` but with CUDA-specific APIs.

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

## Memory Hierarchy
CUDA exposes several types of memory with different scopes and speeds:

1.  **Registers**: Fastest memory. Private to each thread.
2.  **Local Memory**: Slower, part of global memory but private to a thread (used for spills).
3.  **Shared Memory**: Fast, on-chip memory. Shared by threads within the same **Block**. Used for efficient inter-thread communication.
4.  **Global Memory**: Largest but slowest (off-chip). Accessible by all threads and the Host.
5.  **Constant Memory**: Read-only cache, fast if all threads read the same address.

## Function Specifiers
To distinguish where a function runs and is called from:

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

## Kernel Launch
To launch a kernel (function with `__global__`), we use the triple angle bracket syntax: `<<< ... >>>`.

```cpp
// Kernel launch
myKernel<<<gridDim, blockDim>>>(args...);
```

- **gridDim** (or blocks): Number of thread blocks in the grid.
- **blockDim** (or threads): Number of threads per block.

Total threads = `gridDim * blockDim`.

## Thread Hierarchy
Threads are organized in a hierarchy to map to the hardware structure (SMs, SPs).

1.  **Grid**: Complete collection of threads executing a kernel. Made of Blocks.
2.  **Block**: Group of threads that can share memory (Shared Memory) and synchronize.

Built-in variables to locate a thread:
- `gridDim`: Dimensions of the grid.
- `blockDim`: Dimensions of the block.
- `blockIdx`: Index of the current block within the grid.
- `threadIdx`: Index of the current thread within the block.

**Global Index Calculation (1D):**
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

## Synchronization
Since threads run in parallel, we often need to coordinate them.

- **`__syncthreads()`**: Barrier synchronization for threads within the **same Block**. All threads in the block must reach this point before any can proceed.
- **`cudaDeviceSynchronize()`**: Called from the Host. Blocks the CPU until all preceding GPU tasks (kernels, copies) are complete.

## Error Handling
CUDA calls return an error code (`cudaError_t`). Always check it!

```cpp
cudaError_t err = cudaMalloc(&ptr, size);
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
}
```

For kernels (which return void), check `cudaGetLastError()`:
```cpp
myKernel<<<...>>>();
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel Error: %s\n", cudaGetErrorString(err));
}
```
