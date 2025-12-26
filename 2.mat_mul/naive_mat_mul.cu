#include <stdio.h>

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

int main() {
  int M = 1 << 20;
  int N = 1 << 20;

  float *h_A = (float *)malloc(M * N * sizeof(float));
  float *h_B = (float *)malloc(M * N * sizeof(float));
  float *h_C = (float *)malloc(M * N * sizeof(float));

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      h_A[i * N + j] = i;
      h_B[i * N + j] = j;
    }
  }

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * N * sizeof(float));
  cudaMalloc(&d_B, M * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));

  cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, M * N * sizeof(float), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGridX = (M + threadsPerBlock - 1) / threadsPerBlock;
  int blocksPerGridY = (N + threadsPerBlock - 1) / threadsPerBlock;
  matMul<<<blocksPerGridX, blocksPerGridY>>>(d_A, d_B, d_C, M, N);

  cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 5; i++) {
    printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}