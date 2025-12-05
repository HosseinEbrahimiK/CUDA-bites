# CUDA Preliminaries

## __global__

The `__global__` keyword is used to declare a kernel function in CUDA. A kernel function is a function that is executed on the GPU. It is called from the host (CPU) and is executed in parallel on the GPU.

```cuda
__global__ void kernel() {
    // kernel code
}
```

## <<< >>>

The <<< >>> syntax is used to launch a kernel function on the GPU. It takes two arguments: the number of blocks and the number of threads per block.

```cuda
    kernel<<<blocks, threads>>>();  
```

## blocks and threads

Think of blocks as groups of threads. Each block is a group of threads that are executed in parallel. Each thread is a unit of work that is executed on the GPU.


