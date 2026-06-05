#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cooperative_groups.h>




__device__ void warp_reduce(volatile int *sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__device__  int thread_sum(int *input, int size) {
    int sum = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < size/4; i += blockDim.x * gridDim.x) {
        int4 val = ((int4*)input)[i];
        sum += val.x + val.y + val.z + val.w;
    }   
    return sum;
}

__device__ int block_sum(cooperative_groups::thread_group block, int *temp, int val) {
    int lane = block.thread_rank();
    for (int i = block.size()/2; i > 0; i >>= 1) {
        temp[lane] = val;
        block.sync();
        if (lane < i) {
            val += temp[lane + i];
        }
        block.sync();
    }
    return val;
}

__global__ void sum_reduction(int *input, int *sum, int size) {
    extern __shared__ int temp[];
    cooperative_groups::thread_group block = cooperative_groups::this_thread_block();
    int val = thread_sum(input, size);
    val = block_sum(block, temp, val);
    if (block.thread_rank() == 0) {
        atomicAdd(sum, val);
    }
}


int main () {


    int size = 1<<30;

    int *input, *sum;
    cudaMallocManaged(&input, size * sizeof(int));
    cudaMallocManaged(&sum, sizeof(int));

    for (int i = 0; i < size; i++) {
        input[i] = 1;
    }
    dim3 blocksize(256);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int gridSizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
    for (int g = 0; g < 7; g++) {
        int GridSize = gridSizes[g];
        *sum = 0;
        cudaEventRecord(start);
        sum_reduction<<<GridSize, blocksize, blocksize.x * sizeof(int)>>>(input, sum, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "GridSize: " << GridSize << " | Time: " << milliseconds << " ms | Sum: " << sum[0] << std::endl;
    }

     cudaFree(input);
     cudaFree(sum);

     return 0;
}