#ifndef __PRINT_DEV_CUH__
#define __PRINT_DEV_CUH__

#include "../CPU_includes/dtypes.hpp"
#include <iostream>

template <typename T>
__global__ void print_address(T *ptr)
{
    printf("Address: %p\n", ptr);
}

template <typename T>
__device__ void print_addressd(T *ptr)
{
    printf("Address: %p\n", ptr);
}

void print_microtiles(const microtile_metadata *dev_microtiles, int n_microtiles) {
    microtile_metadata *host_microtiles = new microtile_metadata[n_microtiles];
    HANDLE_ERROR(cudaMemcpy(host_microtiles, dev_microtiles, n_microtiles * sizeof(microtile_metadata), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n_microtiles; i++) {
        printf("Microtile %d\n", i);
        printf("\tRow base: %d\n", host_microtiles[i].row_base);
        printf("\tCol base: %d\n", host_microtiles[i].col_base);
        printf("\tNum elements: %d\n", host_microtiles[i].num_elements);
        printf("\tMatrix\n");
        printf("\t\tNum rows: %d\n", host_microtiles[i].matrix.num_rows);
        printf("\t\tNum cols: %d\n", host_microtiles[i].matrix.num_cols);
        printf("\t\tNum nonzeros: %d\n", host_microtiles[i].matrix.num_nonzeros);

        int num_rows = host_microtiles[i].matrix.num_rows;
        int num_nonzeros = host_microtiles[i].matrix.num_nonzeros;

        // Copy row_ptr, col_idx, and values from device to host
        int *row_ptr = new int[num_rows + 1];
        int *col_idx = new int[num_nonzeros];
        float *values = new float[num_nonzeros];

        HANDLE_ERROR(cudaMemcpy(row_ptr, host_microtiles[i].matrix.row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(col_idx, host_microtiles[i].matrix.col_idx, num_nonzeros * sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(values, host_microtiles[i].matrix.values, num_nonzeros * sizeof(float), cudaMemcpyDeviceToHost));

        printf("\t\tRow ptr: ");
        for (int j = 0; j < num_rows + 1; j++) {
            printf("%d ", row_ptr[j]);
        }
        printf("\n");
        printf("\t\tCol idx: ");
        for (int j = 0; j < num_nonzeros; j++) {
            printf("%d ", col_idx[j]);
        }
        printf("\n");
        printf("\t\tValues: ");
        for (int j = 0; j < num_nonzeros; j++) {
            printf("%.2f ", values[j]);
        }
        printf("\n");

        delete[] row_ptr;
        delete[] col_idx;
        delete[] values;
    }

    delete[] host_microtiles;
}

void print_macrotiles(const macrotile_metadata *dev_macrotiles, int n_macrotiles) {
    macrotile_metadata *host_macrotiles = new macrotile_metadata[n_macrotiles];
    HANDLE_ERROR(cudaMemcpy(host_macrotiles, dev_macrotiles, n_macrotiles * sizeof(macrotile_metadata), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n_macrotiles; i++) {
        printf("Macrotile %d\n", i);
        printf("\tRow base: %d\n", host_macrotiles[i].row_base);
        printf("\tCol base: %d\n", host_macrotiles[i].col_base);
        printf("\tNum elements: %d\n", host_macrotiles[i].num_elements);
        printf("\tMatrix\n");
        printf("\t\tNum rows: %d\n", host_macrotiles[i].matrix.num_rows);
        printf("\t\tNum cols: %d\n", host_macrotiles[i].matrix.num_cols);
        printf("\t\tNum nonzeros: %d\n", host_macrotiles[i].matrix.num_nonzeros);

        int num_rows = host_macrotiles[i].matrix.num_rows;
        int num_nonzeros = host_macrotiles[i].matrix.num_nonzeros;

        // Copy row_ptr, col_idx, and values from device to host
        int *row_ptr = new int[num_rows + 1];
        int *col_idx = new int[num_nonzeros];
        float *values = new float[num_nonzeros];

        HANDLE_ERROR(cudaMemcpy(row_ptr, host_macrotiles[i].matrix.row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(col_idx, host_macrotiles[i].matrix.col_idx, num_nonzeros * sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(values, host_macrotiles[i].matrix.values, num_nonzeros * sizeof(float), cudaMemcpyDeviceToHost));

        printf("\t\tRow ptr: ");
        for (int j = 0; j < num_rows + 1; j++) {
            printf("%d ", row_ptr[j]);
        }
        printf("\n");
        printf("\t\tCol idx: ");
        for (int j = 0; j < num_nonzeros; j++) {
            printf("%d ", col_idx[j]);
        }
        printf("\n");
        printf("\t\tValues: ");
        for (int j = 0; j < num_nonzeros; j++) {
            printf("%.2f ", values[j]);
        }
        printf("\n");

        delete[] row_ptr;
        delete[] col_idx;
        delete[] values;
    }

    delete[] host_macrotiles;
}

__global__ void print_in_dev_table(microtile_hash_table* table, int n_microtiles)
{
    printf("Hash Table Contents:\n");
    printf("====================\n");

    // Traverse each bucket in the hash table
    for (size_t i = 0; i < table->count; i++) {
        microtile_hash_node *current = table->entries[i];
        if (current != NULL) {
            printf("Bucket %ld:\n", i);
            // Traverse the linked list in each bucket
            while (current != NULL) {
                microtile_metadata *dev_data = current->data;
                printf("  rowbase: %u, colbase: %u\n", dev_data->row_base, dev_data->col_base);
                current = current->next;
            }
        }
    }
}

__global__ void print_in_dev_table(macrotile_hash_table* table, int n_macrotiles)
{
    printf("Hash Table Contents:\n");
    printf("====================\n");

    // Traverse each bucket in the hash table
    for (size_t i = 0; i < table->count; i++) {
        macrotile_hash_node *current = table->entries[i];
        if (current != NULL) {
            printf("Bucket %ld:\n", i);
            // Traverse the linked list in each bucket
            while (current != NULL) {
                macrotile_metadata *dev_data = current->data;
                microtile_metadata **current_micro = dev_data->microtiles;
                // for (int j = 0; j < dev_data->num_microtiles; j++) {
                //     printf("  rowbase: %u, colbase: %u, Microtile %d: rowbase: %u, colbase: %u\n",dev_data->row_base, dev_data->col_base, j, current_micro[j]->row_base, current_micro[j]->col_base);
                // }
                printf("  rowbase: %u, colbase: %u\n", dev_data->row_base, dev_data->col_base);

                current = current->next;
            }
        }
    }
}


#endif