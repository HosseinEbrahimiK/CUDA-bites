#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vec_add(float *a, float *b, float *c) {
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}

int main() {

  float *a, *b, *c;
  int n = 2 << 10;

  a = (float *)malloc(n * sizeof(float));
  b = (float *)malloc(n * sizeof(float));
  c = (float *)malloc(n * sizeof(float));

  for (int i = 0; i < n; i++) {
    a[i] = i;
    b[i] = i;
  }

  vec_add<<<1, n>>>(a, b, c);

  cudaGetLastError();
  cudaDeviceSynchronize();

  for (int i = 0; i < n; i++) {
    printf("%f + %f = %f\n", a[i], b[i], c[i]);
  }

  free(a);
  free(b);
  free(c);

  return 0;
}
