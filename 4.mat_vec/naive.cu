#include <cuda_runtime.h>
#include <math.h>

__global__ void mat_vec_naive(float *A, float *V, float *B, const int M,
                              const int N) {

  // A: M x N
  // V: N x 1
  // B: M x 1

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < M) {
    B[index] = 0;
    for (int i = 0; i < N; i++) {
      B[index] += A[index * N + i] * V[i];
    }
  }
}

void cudaErrorCheck(cudaError_t err) {
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    exit(1);
  }
}

int main() {

  float *A, *V, *B;
  const int M = 1024;
  const int N = 1024;

  // Allocate memory on the host
  malloc(&A, M * N * sizeof(float));
  malloc(&V, N * sizeof(float));
  malloc(&B, M * sizeof(float));

  // Initialize the input data
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = (float)rand() / (float)RAND_MAX;
    }
  }

  for (int i = 0; i < N; i++) {
    V[i] = (float)rand() / (float)RAND_MAX;
  }

  // Allocate memory on the device
  float *d_A, *d_V, *d_B;
  cudaErrorCheck(cudaMalloc(&d_A, M * N * sizeof(float)));
  cudaErrorCheck(cudaMalloc(&d_V, N * sizeof(float)));
  cudaErrorCheck(cudaMalloc(&d_B, M * sizeof(float)));

  // Copy the input data to the device
  cudaErrorCheck(
      cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
  cudaErrorCheck(cudaMemcpy(d_V, V, N * sizeof(float), cudaMemcpyHostToDevice));

  // Launch the kernel
  dim3 threadsPerBlock(256);
  dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x);
  mat_vec_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_V, d_B, M, N);

  // Copy the output data back to the host
  cudaErrorCheck(cudaMemcpy(B, d_B, M * sizeof(float), cudaMemcpyDeviceToHost));

  // Free the memory
  cudaFree(d_A);
  cudaFree(d_V);
  cudaFree(d_B);

  free(A);
  free(V);
  free(B);

  return 0;
}
