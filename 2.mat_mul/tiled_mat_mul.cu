#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 32

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "Error %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// CUDA kernel for tiled matrix multiplication
// C = A * B
// A is M x N
// B is N x N
// C is M x N
__global__ void tiledMatMul(const float *A, const float *B, float *C, int M,
                            int N) {
  // Block row and column
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread row and column within a tile
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Row and Column of the element in C
  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;

  float acc = 0.0f;

  // Loop over all tiles required to compute the C element
  // Inner dimension is N
  for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {

    // Collaborative loading of A and B tiles into shared memory
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Load A (M x N)
    // Row `row`, Column `t * TILE_SIZE + tx`
    int aIdxRow = row;
    int aIdxCol = t * TILE_SIZE + tx;
    if (aIdxRow < M && aIdxCol < N) {
      As[ty][tx] = A[aIdxRow * N + aIdxCol];
    } else {
      As[ty][tx] = 0.0f;
    }

    // Load B (N x N)
    // Row `t * TILE_SIZE + ty`, Column `col`
    int bIdxRow = t * TILE_SIZE + ty;
    int bIdxCol = col;
    if (bIdxRow < N && bIdxCol < N) {
      Bs[ty][tx] = B[bIdxRow * N + bIdxCol];
    } else {
      Bs[ty][tx] = 0.0f;
    }

    // Synchronize to make sure the tiles are loaded
    __syncthreads();

    // Perform computation on the tiles
    for (int k = 0; k < TILE_SIZE; ++k) {
      acc += As[ty][k] * Bs[k][tx];
    }

    // Synchronize before loading the next tile
    __syncthreads();
  }

  // Write result
  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}

// Host verification function
void matMulCPU(const float *A, const float *B, float *C, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < N; ++k) {
        sum += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
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
  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE,
                     (M + TILE_SIZE - 1) / TILE_SIZE);

  printf("Launching kernel with Grid(%d, %d), Block(%d, %d)\n", blocksPerGrid.x,
         blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);

  tiledMatMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N);

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