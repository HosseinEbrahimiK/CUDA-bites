#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel for naive matrix multiplication
__global__ void matMul(const float *A, const float *B, float *C, int M, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < M && j < N) {
    float sum = 0;
    for (int k = 0; k < N; k++) {
      sum += A[i * N + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
  }
}

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "Error %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

int main() {
  int M = 1024; // Rows of A and C
  int N = 1024; // Cols of A, Rows of B, Cols of B, Cols of C

  size_t size_A = M * N * sizeof(float);
  size_t size_B = N * N * sizeof(float);
  size_t size_C = M * N * sizeof(float);

  // Host memory allocation
  float *h_A = (float *)malloc(size_A);
  float *h_B = (float *)malloc(size_B);
  float *h_C = (float *)malloc(size_C);

  // Initialize matrices
  for (int i = 0; i < M * N; ++i)
    h_A[i] = (float)(rand() % 100) / 10.0f;
  for (int i = 0; i < N * N; ++i)
    h_B[i] = (float)(rand() % 100) / 10.0f;

  // Device memory allocation
  float *d_A, *d_B, *d_C;
  checkCudaError(cudaMalloc((void **)&d_A, size_A), "Alloc A");
  checkCudaError(cudaMalloc((void **)&d_B, size_B), "Alloc B");
  checkCudaError(cudaMalloc((void **)&d_C, size_C), "Alloc C");

  // Copy data to device
  checkCudaError(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice),
                 "Copy A");
  checkCudaError(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice),
                 "Copy B");

  // Launch kernel
  int threadsPerBlock = 32;
  dim3 blockDim(threadsPerBlock, threadsPerBlock);
  dim3 gridDim((N + threadsPerBlock - 1) / threadsPerBlock,
               (M + threadsPerBlock - 1) / threadsPerBlock);

  printf("Launching kernel with Grid(%d, %d), Block(%d, %d)\n", gridDim.x,
         gridDim.y, blockDim.x, blockDim.y);

  matMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N);

  checkCudaError(cudaGetLastError(), "Kernel Launch");
  checkCudaError(cudaDeviceSynchronize(), "Kernel Sync");

  // Copy result back to host
  checkCudaError(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost),
                 "Copy C");

  // Verify
  printf("Verifying result...\n");
  // Just verify 100 random elements
  int pass = 1;
  for (int i = 0; i < 100; ++i) {
    int r = rand() % M;
    int c = rand() % N;
    float expected = 0.0f;
    for (int k = 0; k < N; ++k) {
      expected += h_A[r * N + k] * h_B[k * N + c];
    }
    if (fabs(h_C[r * N + c] - expected) > 1e-3) {
      printf("Mismatch at (%d, %d): GPU=%f, CPU=%f\n", r, c, h_C[r * N + c],
             expected);
      pass = 0;
      break;
    }
  }

  if (pass) {
    printf("PASSED\n");
  } else {
    printf("FAILED\n");
  }

  // Free memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}