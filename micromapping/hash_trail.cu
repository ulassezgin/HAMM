#include <cuda_runtime.h>
#include <stdio.h>

// Device kernel that allocates memory
__global__ void allocateMemoryKernel(int **d_ptr) {
    // Allocate memory inside the kernel
    cudaMalloc(d_ptr, sizeof(int) * 10);
    if (*d_ptr == nullptr) {
        printf("cudaMalloc failed inside kernel\n");
    } else {
        printf("Memory allocated inside kernel\n");
    }
}

int main() {
    int *d_ptr = nullptr;

    // Launch kernel to allocate memory
    allocateMemoryKernel<<<1, 1>>>(&d_ptr);
    cudaDeviceSynchronize();

    // Check if memory allocation was successful
    if (d_ptr == nullptr) {
        printf("Memory allocation failed in kernel\n");
    } else {
        printf("Memory successfully allocated in kernel\n");

        // Free the allocated memory
        cudaFree(d_ptr);
    }

    return 0;
}
