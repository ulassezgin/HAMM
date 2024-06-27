#ifndef __SPGEMM_CUH__
#define __SPGEMM_CUH__

#include <iostream>

#include "../CPU_includes/dtypes.hpp"

#include "handle.cuh"
#include "../microtile/microtile.cuh"
#include "../macrotile/macrotile.cuh"
#include "../commons/print_dev.cuh"
#include "../micromapping/micromapping.cuh"
using namespace std;

void spgemm(const csr_matrix *mat_a, const csr_matrix *mat_b, csr_matrix *mat_c, duration_metadata *duration)
{
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
    dev_b_wrapper->row_ptr = dev_row_ptr; dev_b_wrapper->col_idx = dev_col_ind; dev_b_wrapper->values = dev_val;
    dev_b_wrapper->num_rows = mat_b->num_rows; dev_b_wrapper->num_cols = mat_b->num_cols; dev_b_wrapper->num_nonzeros = mat_b->num_nonzeros;
    HANDLE_ERROR(cudaMemcpy(dev_b, dev_b_wrapper, sizeof(csr_matrix), cudaMemcpyHostToDevice));
    
    printf("Starting to Microtile...\n");
    start = clock();
    microtile_metadata *dev_microtiles = nullptr;
    int n_microtiles = microscheduler(dev_b, &dev_microtiles);
    end = clock();
    duration->microtile_duration = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Microtile duration: %f\n", duration->microtile_duration);
    printf("Number of microtiles: %d\n", n_microtiles);

    HANDLE_ERROR(cudaFree(dev_row_ptr));
    HANDLE_ERROR(cudaFree(dev_col_ind));
    HANDLE_ERROR(cudaFree(dev_val));
    delete dev_b_wrapper;
    dev_b_wrapper = nullptr;

    //hash table for microtiles
    int n_microtiles_row = ceil((float)mat_b->num_rows / T_TILE_HEIGHT);
    start = clock();
    microtile_hash_table table_micro = array_to_bucket_l_ht(dev_microtiles, n_microtiles_row, n_microtiles);
    end = clock();
    duration->microtile_hash_duration = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Microtile hash duration: %f\n", duration->microtile_hash_duration);
    microtile_hash_table* ptr_table_micro;
    HANDLE_ERROR(cudaMalloc((void**)&ptr_table_micro, sizeof(microtile_hash_table)));
    HANDLE_ERROR(cudaMemcpy(ptr_table_micro, &table_micro, sizeof(microtile_hash_table), cudaMemcpyHostToDevice));

    print_in_dev_table <<<1,1>>>(ptr_table_micro, n_microtiles_row);
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
    dev_a_wrapper->row_ptr = dev_row_ptr; dev_a_wrapper->col_idx = dev_col_ind; dev_a_wrapper->values = dev_val;
    dev_a_wrapper->num_rows = mat_a->num_rows; dev_a_wrapper->num_cols = mat_a->num_cols; dev_a_wrapper->num_nonzeros = mat_a->num_nonzeros;
    HANDLE_ERROR(cudaMemcpy(dev_a, dev_a_wrapper, sizeof(csr_matrix), cudaMemcpyHostToDevice));

    printf("***********\n");

    printf("Starting to Macrotile...\n");
    start = clock();
    macrotile_metadata *dev_macrotiles = nullptr;
    int n_macrotiles = macroscheduler(dev_a, &dev_macrotiles);
    end = clock();
    duration->macrotile_duration = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Macrotile duration: %f\n", duration->macrotile_duration);
    printf("Number of macrotiles: %d\n", n_macrotiles);    
    
    int n_macrotiles_row = ceil((float)mat_a->num_rows / TILE_HEIGHT);
    start = clock();
    macrotile_hash_table table_macro = array_to_bucket_l_ht(dev_macrotiles, n_macrotiles_row, n_macrotiles);
    end = clock();
    duration->macrotile_hash_duration = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Macrotile hash duration: %f\n", duration->macrotile_hash_duration);
    macrotile_hash_table* ptr_table_macro;
    
    HANDLE_ERROR(cudaMalloc((void**)&ptr_table_macro, sizeof(macrotile_hash_table)));
    HANDLE_ERROR(cudaMemcpy(ptr_table_macro, &table_macro, sizeof(macrotile_hash_table), cudaMemcpyHostToDevice));
    // print_in_dev_table <<<1,1>>>(ptr_table_macro, n_macrotiles_row);

    int n_macrotiles_col = ceil((float)mat_a->num_cols / TILE_WIDTH);
    start = clock();
    macrotile_hash_table table_macro_c = c_array_to_bucket_l_ht(dev_macrotiles, n_macrotiles_col, n_macrotiles);
    end = clock();    
    duration->macrotile_hash_duration_c = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Macrotile hash duration c: %f\n", duration->macrotile_hash_duration_c);
    macrotile_hash_table* ptr_table_macro_c;
    HANDLE_ERROR(cudaMalloc((void**)&ptr_table_macro_c, sizeof(macrotile_hash_table)));
    HANDLE_ERROR(cudaMemcpy(ptr_table_macro_c, &table_macro_c, sizeof(macrotile_hash_table), cudaMemcpyHostToDevice));
    // print_in_dev_table <<<1,1>>>(ptr_table_macro_c, n_macrotiles_col);
    HANDLE_ERROR(cudaDeviceSynchronize());
    // free_microtile_metadata(dev_microtiles);
    // free_macrotile_metadata(dev_macrotiles);

    // select max row or col
    int max_n = (n_microtiles_row > n_macrotiles_col) ? n_microtiles_row : n_macrotiles_col;
    
    int n_threads = 1024;
    int n_blocks = ceil((float)max_n / 1024);
    start = clock();
    micromap(ptr_table_micro, ptr_table_macro_c);
    end = clock();
    duration->micromap_duration = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Micromap duration: %f\n", duration->micromap_duration);
    
    print_in_dev_table <<<1,1>>>(ptr_table_macro_c, n_macrotiles_col);
    /***************************************************/
    // free the memory
    HANDLE_ERROR(cudaFree(dev_row_ptr));
    HANDLE_ERROR(cudaFree(dev_col_ind));
    HANDLE_ERROR(cudaFree(dev_val));
    HANDLE_ERROR(cudaFree(dev_b)); 
}


#endif // __SPGEMM_CUH__