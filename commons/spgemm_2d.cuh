#ifndef __SPGEMM_2D_CUH__
#define __SPGEMM_2D_CUH__

#include "../CPU_includes/dtypes.hpp"
#include "handle.cuh"
#include "../microtile/microtile_2d.cuh"
#include "../macrotile/macrotile_2d.cuh"

void spgemm(const csr_matrix *mat_a, const csr_matrix *mat_b, csr_matrix *mat_c, duration_metadata *duration) {
    clock_t start, end;
    csr_matrix *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
    csr_matrix *dev_a_wrapper = nullptr, *dev_b_wrapper = nullptr, *dev_c_wrapper = nullptr;
    int *dev_row_ptr = nullptr, *dev_col_ind = nullptr;
    float *dev_val = nullptr;

    // Allocate memory on the device for mat_b
    HANDLE_ERROR(cudaMalloc(&dev_b, sizeof(csr_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_row_ptr, (mat_b->num_rows + 1) * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_col_ind, mat_b->num_nonzeros * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_val, mat_b->num_nonzeros * sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(dev_row_ptr, mat_b->row_ptr, (mat_b->num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_col_ind, mat_b->col_idx, mat_b->num_nonzeros * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_val, mat_b->values, mat_b->num_nonzeros * sizeof(float), cudaMemcpyHostToDevice));

    dev_b_wrapper = new csr_matrix;
    dev_b_wrapper->row_ptr = dev_row_ptr; 
    dev_b_wrapper->col_idx = dev_col_ind; 
    dev_b_wrapper->values = dev_val;
    dev_b_wrapper->num_rows = mat_b->num_rows; 
    dev_b_wrapper->num_cols = mat_b->num_cols; 
    dev_b_wrapper->num_nonzeros = mat_b->num_nonzeros;
    HANDLE_ERROR(cudaMemcpy(dev_b, dev_b_wrapper, sizeof(csr_matrix), cudaMemcpyHostToDevice));

    printf("Starting to Microtile...\n");
    start = clock();
    microtile_2d dev_microtiles_2d;
    int n_microtiles = microscheduler(dev_b, &dev_microtiles_2d);
    end = clock();
    duration->microtile_duration = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Microtile duration: %f\n", duration->microtile_duration);
    printf("Number of microtiles: %d\n", n_microtiles);

    // Free memory for dev_b_wrapper
    HANDLE_ERROR(cudaFree(dev_row_ptr));
    HANDLE_ERROR(cudaFree(dev_col_ind));
    HANDLE_ERROR(cudaFree(dev_val));
    delete dev_b_wrapper;
    dev_b_wrapper = nullptr;

    /**************************************************/
    /***************MACROTILE HANDLING*****************/

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, sizeof(csr_matrix)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_row_ptr, (mat_a->num_rows + 1) * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_col_ind, mat_a->num_nonzeros * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_val, mat_a->num_nonzeros * sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(dev_row_ptr, mat_a->row_ptr, (mat_a->num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_col_ind, mat_a->col_idx, mat_a->num_nonzeros * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_val, mat_a->values, mat_a->num_nonzeros * sizeof(float), cudaMemcpyHostToDevice));

    dev_a_wrapper = new csr_matrix;
    dev_a_wrapper->row_ptr = dev_row_ptr; 
    dev_a_wrapper->col_idx = dev_col_ind; 
    dev_a_wrapper->values = dev_val;
    dev_a_wrapper->num_rows = mat_a->num_rows; 
    dev_a_wrapper->num_cols = mat_a->num_cols; 
    dev_a_wrapper->num_nonzeros = mat_a->num_nonzeros;
    HANDLE_ERROR(cudaMemcpy(dev_a, dev_a_wrapper, sizeof(csr_matrix), cudaMemcpyHostToDevice));

    printf("***********\n");

    printf("Starting to Macrotile...\n");
    start = clock();
    macrotile_2d dev_macrotiles_2d;
    int n_macrotiles = macroscheduler(dev_a, &dev_macrotiles_2d);
    end = clock();
    duration->macrotile_duration = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Macrotile duration: %f\n", duration->macrotile_duration);
    printf("Number of macrotiles: %d\n", n_macrotiles);

    /***************************************************/
    // free the memory
    HANDLE_ERROR(cudaFree(dev_row_ptr));
    HANDLE_ERROR(cudaFree(dev_col_ind));
    HANDLE_ERROR(cudaFree(dev_val));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_a));

    // Free device 2D array if needed
    for (int i = 0; i < dev_microtiles_2d.num_rows; i++) {
        HANDLE_ERROR(cudaFree(dev_microtiles_2d.tiles[i]));
    }
    HANDLE_ERROR(cudaFree(dev_microtiles_2d.tiles));

    for (int i = 0; i < dev_macrotiles_2d.num_rows; i++) {
        HANDLE_ERROR(cudaFree(dev_macrotiles_2d.tiles[i]));
    }
    HANDLE_ERROR(cudaFree(dev_macrotiles_2d.tiles));
}

#endif // __SPGEMM_2D_CUH__
