#include <stdio.h>
#include <cuda.h>

#define N 4

__global__ void transpose(float *A, float *B) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N)
        B[j * N + i] = A[i * N + j];
}

int main() {
    float h_A[N*N], h_B[N*N];
    for (int i = 0; i < N*N; ++i) h_A[i] = i;

    float *d_A, *d_B;
    cudaMalloc((void**)&d_A, N*N*sizeof(float));
    cudaMalloc((void**)&d_B, N*N*sizeof(float));
    cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    transpose<<<blocks, threads>>>(d_A, d_B);

    cudaMemcpy(h_B, d_B, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N*N; ++i) printf("%f ", h_B[i]);
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}
