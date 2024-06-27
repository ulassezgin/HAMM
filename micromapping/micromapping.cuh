#ifndef __MICROMAPPING_CUH__
#define __MICROMAPPING_CUH__

#include "hash.cuh"
#include "../commons/handle.cuh"

microtile_hash_table array_to_bucket_l_ht(microtile_metadata *microtiles, int rowbase_entry_size, int n_microtiles)
{
    microtile_hash_table table;
    initialize_table(table, rowbase_entry_size, n_microtiles);
    HANDLE_ERROR(cudaDeviceSynchronize());

    Lock *lock = new Lock[rowbase_entry_size]; 
    Lock *dev_lock;
    HANDLE_ERROR(cudaMalloc((void**)&dev_lock, rowbase_entry_size * sizeof(Lock)));
    HANDLE_ERROR(cudaMemcpy(dev_lock, lock, rowbase_entry_size * sizeof(Lock), cudaMemcpyHostToDevice));

    // printf("Adding to table...\n");

    add_to_table<<<100, 100>>>(table, n_microtiles, microtiles, dev_lock);
    HANDLE_ERROR(cudaDeviceSynchronize());

    // printf("Freeing table...\n");


    delete[] lock;  

    return table;
}

macrotile_hash_table array_to_bucket_l_ht(macrotile_metadata *macrotiles, int rowbase_entry_size, int n_macrotiles)
{
    macrotile_hash_table table;
    initialize_table(table, rowbase_entry_size, n_macrotiles);
    HANDLE_ERROR(cudaDeviceSynchronize());

    Lock *lock = new Lock[rowbase_entry_size]; 
    Lock *dev_lock;
    HANDLE_ERROR(cudaMalloc((void**)&dev_lock, rowbase_entry_size * sizeof(Lock)));
    HANDLE_ERROR(cudaMemcpy(dev_lock, lock, rowbase_entry_size * sizeof(Lock), cudaMemcpyHostToDevice));

    // printf("Adding to table...\n");

    add_to_table<<<100, 100>>>(table, n_macrotiles, macrotiles, dev_lock);
    HANDLE_ERROR(cudaDeviceSynchronize());

    // printf("Freeing table...\n");

    delete[] lock;  

    return table;
}

microtile_hash_table c_array_to_bucket_l_ht(microtile_metadata *microtiles, int colbase_entry_size, int n_microtiles)
{
    microtile_hash_table table;
    initialize_table(table, colbase_entry_size, n_microtiles);
    HANDLE_ERROR(cudaDeviceSynchronize());

    Lock *lock = new Lock[colbase_entry_size]; 
    Lock *dev_lock;
    HANDLE_ERROR(cudaMalloc((void**)&dev_lock, colbase_entry_size * sizeof(Lock)));
    HANDLE_ERROR(cudaMemcpy(dev_lock, lock, colbase_entry_size * sizeof(Lock), cudaMemcpyHostToDevice));

    // printf("Adding to table...\n");

    add_to_table_c<<<100, 100>>>(table, n_microtiles, microtiles, dev_lock);
    HANDLE_ERROR(cudaDeviceSynchronize());

    // printf("Freeing table...\n");

    delete[] lock;  

    return table;
}

macrotile_hash_table c_array_to_bucket_l_ht(macrotile_metadata *macrotiles, int colbase_entry_size, int n_macrotiles)
{
    macrotile_hash_table table;
    initialize_table(table, colbase_entry_size, n_macrotiles);
    HANDLE_ERROR(cudaDeviceSynchronize());

    Lock *lock = new Lock[colbase_entry_size]; 
    Lock *dev_lock;
    HANDLE_ERROR(cudaMalloc((void**)&dev_lock, colbase_entry_size * sizeof(Lock)));
    HANDLE_ERROR(cudaMemcpy(dev_lock, lock, colbase_entry_size * sizeof(Lock), cudaMemcpyHostToDevice));

    // printf("Adding to table...\n");

    add_to_table_c<<<100, 100>>>(table, n_macrotiles, macrotiles, dev_lock);
    HANDLE_ERROR(cudaDeviceSynchronize());

    // printf("Freeing table...\n");

    delete[] lock;  

    return table;
}

void allocate_microtiles_host(macrotile_hash_table &table, int n_macrotiles, int *count_buffer)
{
    for(int i = 0; i < n_macrotiles; i++)
    {
        macrotile_hash_node *current_macro = table.entries[i];
        if(current_macro == NULL) continue;
        while(current_macro != NULL)
        {
            current_macro->data->microtiles = new microtile_metadata*[count_buffer[i]];
            current_macro->data->num_microtiles = 0;
            current_macro = current_macro->next;
        }
    }
}

void allocate_microtiles_in_macro_table(macrotile_hash_table &table, int n_macrotiles, int *count_buffer)
{
    // table is dev allocated.
    macrotile_hash_table *h_table = new macrotile_hash_table;
    HANDLE_ERROR(cudaMemcpy(h_table, &table, sizeof(macrotile_hash_table), cudaMemcpyDeviceToHost));
    // print_address <<<1,1>>> (h_table);
    // print_address <<<1,1>>> (h_table->entries);
    macrotile_hash_node **h_entries = new macrotile_hash_node*[n_macrotiles];
    HANDLE_ERROR(cudaMemcpy(h_entries, h_table->entries, n_macrotiles * sizeof(macrotile_hash_node*), cudaMemcpyDeviceToHost));
    macrotile_hash_node *tmp_entry = new macrotile_hash_node;

    for(int i = 0; i < h_table->count; i++)
    {
        // print_address <<<1,1>>> (h_entries[i]);
        HANDLE_ERROR(cudaMemcpy(tmp_entry, h_entries[i], sizeof(macrotile_hash_node), cudaMemcpyDeviceToHost));
        if(tmp_entry == NULL) continue;
        macrotile_metadata *tmp_data = new macrotile_metadata;
        macrotile_hash_node *tmp_iter_ptr, tmp_iter;
        tmp_iter_ptr = tmp_entry;
        tmp_iter = *tmp_iter_ptr;
        while(tmp_entry != NULL)
        {
            HANDLE_ERROR(cudaMemcpy(tmp_data, tmp_iter.data, sizeof(macrotile_metadata), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMalloc((void**)&tmp_data->microtiles, count_buffer[i] * sizeof(microtile_metadata*)));
            HANDLE_ERROR(cudaMemcpy(tmp_iter.data, tmp_data, sizeof(macrotile_metadata), cudaMemcpyHostToDevice));
            // next is allocated in dev cpy to host
            // print_address <<<1,1>>> (tmp_entry);    
            // copy dev allcated tmp_entry->next to host allocated tmp_entry
            
            if(tmp_iter.next == NULL) break;
            HANDLE_ERROR(cudaMemcpy(&tmp_iter, tmp_iter.next, sizeof(macrotile_hash_node), cudaMemcpyDeviceToHost));
   
        }
        delete tmp_data;
           

        HANDLE_ERROR(cudaMemcpy(h_entries[i], tmp_entry, sizeof(macrotile_hash_node), cudaMemcpyHostToDevice));    
    }
    HANDLE_ERROR(cudaMemcpy(h_table->entries, h_entries, n_macrotiles * sizeof(macrotile_hash_node*), cudaMemcpyHostToDevice));
    delete h_table;
}

__global__ void count_matches(microtile_hash_table *table_micro, macrotile_hash_table *table_macro, int *count_buffer)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = max(table_micro->count, table_macro->count);
    if(idx < n)
    {
        // printf("Bucket %d\n", idx);
        macrotile_hash_node *current_macro = table_macro->entries[idx];
        if(current_macro == NULL) return;
        microtile_hash_node *current_micro = table_micro->entries[idx];
        if(current_micro == NULL) return;
        count_buffer[idx] = table_micro->elm_per_bucket[idx];
        // while(current_macro != NULL)
        // {
        //     while(current_micro != NULL)
        //     {
        //         // printf("s%d\n",current_macro->data->num_microtiles);
        //         // print_addressd(current_macro->data->microtiles[current_macro->data->num_microtiles]);
        //         // current_macro->data->microtiles[current_macro->data->num_microtiles] = current_micro->data;
        //         // current_macro->data->num_microtiles++;
        //         printf("%d Macro rowbase: %d, colbase: %d, next: %p\n",idx, current_macro->data->row_base, current_macro->data->col_base, current_micro->next);
        //         current_micro = current_micro->next;
        //     }
        //     current_macro = current_macro->next;

        // }
    }
}

__global__ void assign_matches(microtile_hash_table *table_micro, macrotile_hash_table *table_macro, int *count_buffer)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = max(table_micro->count, table_macro->count);
    if(idx < n)
    {
        macrotile_hash_node *current_macro = table_macro->entries[idx];
        if(current_macro == NULL) return;
        microtile_hash_node *current_micro = table_micro->entries[idx];
        if(current_micro == NULL) return;
        while(current_macro != NULL)
        {
            while(current_micro != NULL)
            {
                current_macro->data->microtiles[current_macro->data->num_microtiles] = current_micro->data;
                current_macro->data->num_microtiles++;
                current_micro = current_micro->next;
            }
            current_macro = current_macro->next;
            current_micro = table_micro->entries[idx];
        }
    }
}


void micromap(microtile_hash_table *table_micro, macrotile_hash_table *table_macro)
{
    int h_micro_count;
    HANDLE_ERROR(cudaMemcpy(&h_micro_count, &table_micro->count, sizeof(int), cudaMemcpyDeviceToHost));
    int h_macro_count;
    HANDLE_ERROR(cudaMemcpy(&h_macro_count, &table_macro->count, sizeof(int), cudaMemcpyDeviceToHost));
    int n = max(h_micro_count, h_macro_count);
    int *count_buffer;

    HANDLE_ERROR(cudaMalloc((void**)&count_buffer, n * sizeof(int)));
    
    int n_threads = 1024;
    int n_blocks = ceil((float)n / 1024);

    count_matches<<<n_blocks, n_threads>>>(table_micro, table_macro, count_buffer);
    HANDLE_ERROR(cudaDeviceSynchronize());

    int *h_count_buffer = new int[n];
    HANDLE_ERROR(cudaMemcpy(h_count_buffer, count_buffer, n * sizeof(int), cudaMemcpyDeviceToHost));

    for(int i = 0; i < n; i++)
    {
        // printf("Bucket %d: %d\n", i, h_count_buffer[i]);
    }
    int start = clock();
    allocate_microtiles_in_macro_table(*table_macro, n, h_count_buffer);
    int end = clock();
    printf("Allocation duration: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
    int h_count_micro;
    HANDLE_ERROR(cudaMemcpy(&h_count_micro, &table_micro->count, sizeof(int), cudaMemcpyDeviceToHost));
    int h_count_macro;
    HANDLE_ERROR(cudaMemcpy(&h_count_macro, &table_macro->count, sizeof(int), cudaMemcpyDeviceToHost));


    assign_matches<<<n_blocks, n_threads>>>(table_micro, table_macro, count_buffer);
    
    // print_in_dev_table<<<1,1>>>(table_macro, 2);

    delete[] h_count_buffer;
    HANDLE_ERROR(cudaFree(count_buffer));
}
#endif // __MICROMAPPING_CUH__