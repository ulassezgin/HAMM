#ifndef __SIZE_PREDICTION_CUH__
#define __SIZE_PREDICTION_CUH__

#include "../CPU_includes/dtypes.hpp"
#include "../commons/handle.cuh"
#include "../commons/print_dev.cuh"

__global__ void size_pred_kernel(macrotile_hash_node* pool, size_t *bucket_count)
{
    int macro_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (macro_idx < *bucket_count)
    {
        macrotile_hash_node *current_macro = &pool[macro_idx];
        for(int i = 0; i < current_macro->data->num_microtiles; i++)
        {
            microtile_metadata *current_micro = current_macro->data->microtiles[i];
            int n_rows = current_micro->matrix.num_rows;
            for(int j = 0; j < n_rows; j++)
            {
                int row_ptr_start = current_micro->matrix.row_ptr[j];
                int row_ptr_end = current_micro->matrix.row_ptr[j + 1];
                int n_cols = row_ptr_end - row_ptr_start;
                current_micro->res_matrix.num_nonzeros += n_cols;
            }

            printf("Macro: %d, Micro: %d, Nonzeros: %d\n", macro_idx, i, current_micro->res_matrix.num_nonzeros);
        }

        macro_idx += stride;
    }
}

void memory_allocation_micro_res(microtile_hash_table *table_micro)
{
    microtile_hash_table *table_micro_h = new microtile_hash_table;
    HANDLE_ERROR(cudaMemcpy(table_micro_h, table_micro, sizeof(microtile_hash_table), cudaMemcpyDeviceToHost));
    microtile_hash_node *pool_h = new microtile_hash_node[table_micro_h->count];
    HANDLE_ERROR(cudaMemcpy(pool_h, table_micro_h->pool, table_micro_h->count * sizeof(microtile_hash_node), cudaMemcpyDeviceToHost));
    for(int i = 0; i < table_micro_h->count; i++)
    {
        microtile_hash_node *current_micro = &pool_h[i];
        microtile_metadata *h_tile = new microtile_metadata;
        HANDLE_ERROR(cudaMemcpy(h_tile, current_micro->data, sizeof(microtile_metadata), cudaMemcpyDeviceToHost));
        printf("Microtile: %d, Nonzeros: %d\n", i, h_tile->res_matrix.num_nonzeros);
    }
}

void size_prediction(macrotile_hash_table *table_macro)
{
    size_t bucket_count_h;
    HANDLE_ERROR(cudaMemcpy(&bucket_count_h, &table_macro->count, sizeof(size_t), cudaMemcpyDeviceToHost));
    printf("Bucket count: %d\n", bucket_count_h);

    int threads_per_block = 1024;
    int blocks = (bucket_count_h + threads_per_block - 1) / threads_per_block;

    macrotile_hash_table *table_macro_h = new macrotile_hash_table;
    HANDLE_ERROR(cudaMemcpy(table_macro_h, table_macro, sizeof(macrotile_hash_table), cudaMemcpyDeviceToHost));

    size_pred_kernel<<<blocks, threads_per_block>>>(table_macro_h->pool, &table_macro->count);
    HANDLE_ERROR(cudaDeviceSynchronize());

}

#endif