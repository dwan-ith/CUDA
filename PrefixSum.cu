#include <cuda_runtime.h>
#include <iostream>

__global__ void prescan(float *g_odata, float *g_idata, int n) {
    extern __shared__ float temp[];
    int thid = threadIdx.x;
    int offset = 1;
    temp[2 * thid] = g_idata[2 * thid];
    temp[2 * thid + 1] = g_idata[2 * thid + 1];
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if (thid == 0) {
        temp[n - 1] = 0;
    }
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    g_odata[2 * thid] = temp[2 * thid];
    g_odata[2 * thid + 1] = temp[2 * thid + 1];
}

int main() {
    int N = 1024;
    size_t size = N * sizeof(float);
    float *h_in = (float *)malloc(size);
    float *h_out = (float *)malloc(size);

    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f; 
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    prescan<<<1, N / 2, N * sizeof(float)>>>(d_out, d_in, N);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += h_out[i];
    }
    std::cout << "Sum of prefix scan results: " << sum << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
