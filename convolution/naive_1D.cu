

#include <iostream>
#include <math.h>
#include <stdlib.h>


void createArray(int *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = rand() % 10;
    }
}
__global__ void convolution_1D(int *input, int *mask, int *output, int size, int mask_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int rad = mask_size/2;
    int start = tid - rad;
    int end = start + mask_size;
    int tmp = 0;
    for ( int j = start; j < end ; ++ j) {
        if (j >= 0 && j < size)
        tmp += input[j] * mask[j-start];
    }
    output[tid] = tmp;
}

int main(int argc, char**argv) {
    int size = 1 << 30;
    int mask_size = 3;
    int * input;
    int * mask;
    int * out;
    

    cudaMallocManaged(&input, size * sizeof(int));
    cudaMallocManaged(&out, size  * sizeof(int));
    cudaMallocManaged(&mask, mask_size * sizeof(int));

    for (int i = 0; i < mask_size; i++) {
        mask[i] = rand() % 10;
    }
    createArray(input, size);

    dim3 TB(1024);
    dim3 Grid(size / TB.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    convolution_1D<<<Grid, TB>>>(input, mask, out, size, mask_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;


    //cpu convolution
    clock_t cpu_start = clock();
    for (int i = 0; i < size; i++) {
        int rad = mask_size/2;
        int start = i - rad;
        int end = start + mask_size;
        int tmp = 0;
        for ( int j = start; j < end ; ++ j) {
            if (j >= 0 && j < size)
            tmp += input[j] * mask[j-start];
        }
        if (out[i] != tmp)
            std::cout << "Mismatch at index " << i << ": GPU result = " << out[i] << ", CPU result = " << tmp << std::endl;
    }
    clock_t cpu_end = clock();
    double cpu_time = double(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000;
    std::cout << "CPU execution time: " << cpu_time << " ms" << std::endl;

    cudaFree(input);
    cudaFree(mask);
    cudaFree(out);

    return 0;

    
}


