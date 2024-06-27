#ifndef __HASH_CUH__
#define __HASH_CUH__

#include "../CPU_includes/dtypes.hpp"
#include "lock.cuh"


#define SIZE (100 * 1024)

__device__ __host__ size_t hash_function(unsigned int key, size_t count)
{
    return (float)key / count;
}

void initialize_table(microtile_hash_table &table, int entries, int elements)
{
    table.count = entries;
    printf("Table count: %d\n", table.count);
    HANDLE_ERROR(cudaMalloc((void **)&table.entries, entries * sizeof(microtile_hash_node *)));
    HANDLE_ERROR(cudaMemset(table.entries, 0, entries * sizeof(microtile_hash_node *)));
    HANDLE_ERROR(cudaMalloc((void **)&table.pool, elements * sizeof(microtile_hash_node)));
    HANDLE_ERROR(cudaMalloc((void **)&table.elm_per_bucket, entries * sizeof(int)));
    HANDLE_ERROR(cudaMemset(table.elm_per_bucket, 0, entries * sizeof(int)));
}

void initialize_table(macrotile_hash_table &table, int entries, int elements)
{
    table.count = entries;
    printf("Table count: %d\n", table.count);
    HANDLE_ERROR(cudaMalloc((void **)&table.entries, entries * sizeof(macrotile_hash_node *)));
    HANDLE_ERROR(cudaMemset(table.entries, 0, entries * sizeof(macrotile_hash_node *)));
    HANDLE_ERROR(cudaMalloc((void **)&table.pool, elements * sizeof(macrotile_hash_node)));
    HANDLE_ERROR(cudaMalloc((void **)&table.elm_per_bucket, entries * sizeof(int)));
    HANDLE_ERROR(cudaMemset(table.elm_per_bucket, 0, entries * sizeof(int)));

}



void free_table(microtile_hash_table &table)
{
    free(table.entries);
    free(table.pool);
}

void free_table(macrotile_hash_table &table)
{
    free(table.entries);
    free(table.pool);
}

__global__ void add_to_table(microtile_hash_table table, int n_microtiles, microtile_metadata *microtiles, Lock *lock)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (idx < n_microtiles)
    {
        microtile_metadata *microtile = &microtiles[idx];
        size_t hashValue = hash_function(microtile->row_base, T_TILE_HEIGHT);
        for(int i = 0; i < 32; i++) {
            if ((idx % 32) == i) {
                microtile_hash_node *location = &table.pool[idx];
                location->data = microtile;
                lock[hashValue].lock();
                location->next = table.entries[hashValue];
                table.entries[hashValue] = location;
                table.elm_per_bucket[hashValue]++;
                lock[hashValue].unlock();
            }
        }
        idx += stride;
    }
}

__global__ void add_to_table(macrotile_hash_table table, int n_macrotiles, macrotile_metadata *macrotiles, Lock *lock)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (idx < n_macrotiles)
    {
        macrotile_metadata *macrotile = &macrotiles[idx];
        size_t hashValue = hash_function(macrotile->row_base, TILE_HEIGHT);
        for(int i = 0; i < 32; i++) {
            if ((idx % 32) == i) {
                macrotile_hash_node *location = &table.pool[idx];
                location->data = macrotile;
                lock[hashValue].lock();
                location->next = table.entries[hashValue];
                table.entries[hashValue] = location;
                table.elm_per_bucket[hashValue]++;
                lock[hashValue].unlock();
            }
        }
        idx += stride;
    }
}

__global__ void add_to_table_c(microtile_hash_table table, int n_microtiles, microtile_metadata *microtiles, Lock *lock)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (idx < n_microtiles)
    {
        microtile_metadata *microtile = &microtiles[idx];
        size_t hashValue = hash_function(microtile->col_base, T_TILE_WIDTH);
        for(int i = 0; i < 32; i++) {
            if ((idx % 32) == i) {
                microtile_hash_node *location = &table.pool[idx];
                location->data = microtile;
                lock[hashValue].lock();
                location->next = table.entries[hashValue];
                table.entries[hashValue] = location;
                table.elm_per_bucket[hashValue]++;
                lock[hashValue].unlock();
            }
        }
        idx += stride;
    }
}

__global__ void add_to_table_c(macrotile_hash_table table, int n_macrotiles, macrotile_metadata *macrotiles, Lock *lock)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (idx < n_macrotiles)
    {
        macrotile_metadata *macrotile = &macrotiles[idx];
        size_t hashValue = hash_function(macrotile->col_base, TILE_WIDTH);
        for(int i = 0; i < 32; i++) {
            if ((idx % 32) == i) {
                macrotile_hash_node *location = &table.pool[idx];
                location->data = macrotile;
                lock[hashValue].lock();
                location->next = table.entries[hashValue];
                table.entries[hashValue] = location;
                table.elm_per_bucket[hashValue]++;
                lock[hashValue].unlock();
            }
        }
        idx += stride;
    }
}

#endif