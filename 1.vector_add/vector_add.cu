#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

int main() {

  int N = 1 << 20; // 1M elements
  size_t size = N * sizeof(float);

  // Allocate Host memory
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);

  // Initialize inputs
  for (int i = 0; i < N; ++i) {
    h_A[i] = i;
    h_B[i] = i;
  }

  // Allocate Device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  // Copy Host -> Device
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Launch Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  // Copy Device -> Host
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // Verify (check first 5 elements)
  for (int i = 0; i < 5; i++) {
    printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
  }

  // Free memory on device
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free memory on host
  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
