#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>


__device__ void warp_reduce(volatile int *sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void sum_reduction(int *input, int *output) {
    __shared__ int sdata[1024];

    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[threadIdx.x] = input[idx] + input[idx + blockDim.x];
    __syncthreads();
    for (int s = blockDim.x / 2; s >= 32; s /= 2) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        // int idx = threadIdx.x * 2 * s;
        // if (idx < blockDim.x) {
        //     sdata[idx] += sdata[idx + s];
        __syncthreads(); 
    }
    if (threadIdx.x < 32) {
        warp_reduce(sdata, threadIdx.x);
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

//host code
int main(int argc, char *argv[]) {
    int size = 1024 * 1024;
    int *h_input, *h_output;
    int *d_input, *d_output;

    h_input = (int*)malloc(size * sizeof(int));
    h_output = (int*)malloc((size/1024) * sizeof(int));

    for (int i = 0; i < size; i++) {
        h_input[i] = 1;
    }

    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, (size/1024) * sizeof(int));

    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blocksize(1024);
    int GridSize =(int) ceil(((float)size / blocksize.x)   /2);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sum_reduction<<<GridSize, blocksize>>>(d_input, d_output);
    sum_reduction<<<1, blocksize>>>(d_output, d_output);
    

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); 
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    
    cudaMemcpy(h_output, d_output, (size/1024) * sizeof(int), cudaMemcpyDeviceToHost);
    int final_sum = h_output[0];

    int cpu_sum = 0;
    clock_t cpu_start = clock();
    for (int i = 0; i < size; i++) {
        cpu_sum += h_input[i];
    }
    clock_t cpu_end = clock();
    int cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;
    std::cout << "CPU execution time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU sum: " << final_sum << std::endl;
    std::cout << "CPU sum: " << cpu_sum << std::endl;



     free(h_input);
     free(h_output);
     cudaFree(d_input);
     cudaFree(d_output);

     return 0;
}

