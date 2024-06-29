#ifndef __ESC_CUH__
#define __ESC_CUH__

#include "../CPU_includes/dtypes.hpp"
#include "../commons/handle.cuh"

__global__ void count_operations_per_bucket(macrotile_hash_table *table_macro, int *operations_per_bucket)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= table_macro->count) return;

    int n_operations = 0;
    macrotile_hash_node *current = table_macro->entries[i];
    while (current != NULL)
    {
        n_operations += current->data->num_microtiles;
        current = current->next;
    }

    operations_per_bucket[i] = n_operations;
}

__global__ void calculate_num_concurrent_ops(int *operations_per_bucket, int n_max_operation_in_a_round, int *n_buckets_to_operate_concurrently, int total_buckets, int start_bucket)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx != 0) return;

    int n_operations_total = 0;
    int n_buckets_to_operate = 0;

    for (int i = start_bucket; i < total_buckets; i++)
    {
        int n_operations = operations_per_bucket[i];
        if (n_operations + n_operations_total <= n_max_operation_in_a_round)
        {
            n_operations_total += n_operations;
            n_buckets_to_operate++;
        }
        else
        {
            break;
        }
    }

    *n_buckets_to_operate_concurrently = n_buckets_to_operate;
}

void esc_scheduler(macrotile_hash_table *table_macro)
{
    int n_max_operation_in_a_round = pow(2, 16);
    int *operations_per_bucket;
    int *d_n_buckets_to_operate_concurrently;
    int total_buckets;

    HANDLE_ERROR(cudaMemcpy(&total_buckets, &table_macro->count, sizeof(int), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaMalloc((void**)&operations_per_bucket, total_buckets * sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_n_buckets_to_operate_concurrently, sizeof(int)));
    HANDLE_ERROR(cudaMemset(d_n_buckets_to_operate_concurrently, 0, sizeof(int)));

    // Kernel to count operations in each bucket
    count_operations_per_bucket<<<(total_buckets + 255) / 256, 256>>>(table_macro, operations_per_bucket);
    HANDLE_ERROR(cudaDeviceSynchronize());

    int h_n_buckets_to_operate_concurrently;
    int start_bucket = 0;
    int n_rounds = 0;

    while (start_bucket < total_buckets)
    {
        calculate_num_concurrent_ops<<<1, 1>>>(operations_per_bucket, n_max_operation_in_a_round, d_n_buckets_to_operate_concurrently, total_buckets, start_bucket);
        HANDLE_ERROR(cudaDeviceSynchronize());

        HANDLE_ERROR(cudaMemcpy(&h_n_buckets_to_operate_concurrently, d_n_buckets_to_operate_concurrently, sizeof(int), cudaMemcpyDeviceToHost));
        printf("Round %d: Number of buckets to operate concurrently: %d\n", n_rounds, h_n_buckets_to_operate_concurrently);

        start_bucket += h_n_buckets_to_operate_concurrently;
        n_rounds++;
    }

    printf("Total number of rounds: %d\n", n_rounds);

    HANDLE_ERROR(cudaFree(operations_per_bucket));
    HANDLE_ERROR(cudaFree(d_n_buckets_to_operate_concurrently));
}

#endif // __ESC_CUH__
