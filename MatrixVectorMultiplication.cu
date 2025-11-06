#include <cuda_runtime.h>
#include <iostream>

__global__ void matVecMulKernel(float* mat, float* vec, float* res, int numRows, int numCols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0.0f;
        for (int i = 0; i < numCols; ++i) {
            sum += mat[row * numCols + i] * vec[i];
        }
        res[row] = sum;
    }
}

int main() {
    int numRows = 4;
    int numCols = 4;

    float h_mat[] = {1, 2, 3, 4,
                     5, 6, 7, 8,
                     9, 10, 11, 12,
                     13, 14, 15, 16};
    float h_vec[] = {1, 1, 1, 1};
    float h_res[numRows];

    float *d_mat, *d_vec, *d_res;
    cudaMalloc((void**)&d_mat, numRows * numCols * sizeof(float));
    cudaMalloc((void**)&d_vec, numCols * sizeof(float));
    cudaMalloc((void**)&d_res, numRows * sizeof(float));

    cudaMemcpy(d_mat, h_mat, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec, numCols * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numRows + threadsPerBlock - 1) / threadsPerBlock;

    matVecMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_mat, d_vec, d_res, numRows, numCols);

    cudaMemcpy(h_res, d_res, numRows * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numRows; ++i) {
        std::cout << h_res[i] << std::endl;
    }

    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(d_res);

    return 0;
}
