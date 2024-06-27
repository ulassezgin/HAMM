#ifndef __MICROTILE_CUH__
#define __MICROTILE_CUH__

// #include "../micromapping/hash.cuh"

__global__ void microscheduler_init(const csr_matrix *mat, microtile_metadata *dev_microtiles, int *d_valid_tile_idx, int batch_start, int batch_size, int d_valid_tile_count, int n_microtiles_col)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int valid_idx = idx + batch_start;

    if(valid_idx < batch_start + batch_size && valid_idx < d_valid_tile_count)
    {
        int index = d_valid_tile_idx[valid_idx];

        int row = index / n_microtiles_col;
        int col = index % n_microtiles_col;

        int start_row = row * T_TILE_HEIGHT;
        int start_col = col * T_TILE_WIDTH;

        int row_ptr_idx = 0;
        int row_ptr_val = 0;

        dev_microtiles[valid_idx].matrix.row_ptr[0] = 0;

        for(int i = start_row; i < min(start_row + T_TILE_HEIGHT, mat->num_rows); i++)
        {
            for(int j = mat->row_ptr[i]; j < mat->row_ptr[i + 1]; j++)
            {
                int col_idx = mat->col_idx[j];
                if(col_idx >= start_col && col_idx < min(start_col + T_TILE_WIDTH, mat->num_cols))
                {
                    dev_microtiles[valid_idx].matrix.col_idx[row_ptr_val] = col_idx - start_col;
                    dev_microtiles[valid_idx].matrix.values[row_ptr_val] = mat->values[j];
                    row_ptr_val++;
                }
            }
            row_ptr_idx++;
            dev_microtiles[valid_idx].matrix.row_ptr[row_ptr_idx] = row_ptr_val;
        }

        dev_microtiles[valid_idx].matrix.num_nonzeros = row_ptr_val;
        dev_microtiles[valid_idx].matrix.num_rows = min(T_TILE_HEIGHT, mat->num_rows - start_row);
        dev_microtiles[valid_idx].matrix.num_cols = min(T_TILE_WIDTH, mat->num_cols - start_col);

        dev_microtiles[valid_idx].row_base = start_row;
        dev_microtiles[valid_idx].col_base = start_col;
        dev_microtiles[valid_idx].num_elements = row_ptr_val;

        dev_microtiles[valid_idx].res_matrix.num_rows = min(T_TILE_HEIGHT, mat->num_rows - start_row);
        dev_microtiles[valid_idx].res_matrix.num_cols = min (T_TILE_WIDTH, mat->num_cols - start_col);
        dev_microtiles[valid_idx].res_matrix.num_nonzeros = 0;
    }
}

__global__ void count_valid_microtiles(const csr_matrix *mat, int *d_valid_tile_count, int *d_valid_tile_idx, int n_microtiles_col, int n_microtiles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n_microtiles)
    {
        int row = idx / n_microtiles_col;
        int col = idx % n_microtiles_col;

        int start_row = row * T_TILE_HEIGHT;
        int start_col = col * T_TILE_WIDTH;

        int valid_tile = 0;
        for(int i = start_row; i < min(start_row + T_TILE_HEIGHT, mat->num_rows); i++)
        {
            for(int j = mat->row_ptr[i]; j < mat->row_ptr[i + 1]; j++)
            {
                int col_idx = mat->col_idx[j];
                if(col_idx >= start_col && col_idx < min(start_col + T_TILE_WIDTH, mat->num_cols))
                {
                    valid_tile++;
                }
            }
        }

        if(valid_tile > 0)
        {
            int old = atomicAdd(d_valid_tile_count, 1);
            d_valid_tile_idx[old] = idx;
        }
    }
}

int microscheduler(const csr_matrix *mat, microtile_metadata **dev_microtiles)
{
    int h_rows, h_cols;
    HANDLE_ERROR(cudaMemcpy(&h_rows, &mat->num_rows, sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&h_cols, &mat->num_cols, sizeof(int), cudaMemcpyDeviceToHost));

    int n_microtiles_row = ceil((float) h_rows / T_TILE_HEIGHT);
    int n_microtiles_col = ceil((float)h_cols / T_TILE_WIDTH);
    int n_microtiles = n_microtiles_row * n_microtiles_col;

    int* d_valid_tile_count, h_valid_tile_count = 0;
    int * d_valid_tile_idx;

    HANDLE_ERROR(cudaMalloc((void**)&d_valid_tile_count, sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_valid_tile_idx, n_microtiles * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_valid_tile_count, &h_valid_tile_count, sizeof(int), cudaMemcpyHostToDevice));

    int block_size_cvmt = 1024;
    int grid_size_cvmt = ceil((float)n_microtiles / block_size_cvmt);
    count_valid_microtiles <<<grid_size_cvmt, block_size_cvmt>>> (mat, d_valid_tile_count, d_valid_tile_idx, n_microtiles_col, n_microtiles);
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaMemcpy(&h_valid_tile_count, d_valid_tile_count, sizeof(int), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaMalloc((void**)dev_microtiles, h_valid_tile_count * sizeof(microtile_metadata)));
    microtile_metadata *h_microtiles = new microtile_metadata[h_valid_tile_count];

    for (int i = 0; i < h_valid_tile_count; i++) {
        csr_matrix temp;

        int *d_rowptr, *d_colidx;
        float *d_values;
        HANDLE_ERROR(cudaMalloc((void**)&d_rowptr, (T_TILE_HEIGHT + 1) * sizeof(int)));
        HANDLE_ERROR(cudaMemset(d_rowptr, 0, (T_TILE_HEIGHT + 1) * sizeof(int)));
        HANDLE_ERROR(cudaMalloc((void**)&d_colidx, T_TILE_WIDTH * T_TILE_HEIGHT * sizeof(int)));
        HANDLE_ERROR(cudaMalloc((void**)&d_values, T_TILE_WIDTH * T_TILE_HEIGHT * sizeof(float)));

        temp.num_rows = T_TILE_HEIGHT;
        temp.num_cols = T_TILE_WIDTH;
        temp.num_nonzeros = 0;
        temp.row_ptr = d_rowptr;
        temp.col_idx = d_colidx;
        temp.values = d_values;

        h_microtiles[i].matrix = temp;
    }

    HANDLE_ERROR(cudaMemcpy(*dev_microtiles, h_microtiles, h_valid_tile_count * sizeof(microtile_metadata), cudaMemcpyHostToDevice));
    delete[] h_microtiles;

    const int table_size = ceil(h_rows / T_TILE_HEIGHT);


    // Insert the microtiles into the hash table PARALLELY

    

    int pitch = n_microtiles_col;
    int batch = pow(2,16);
    for(int batch_start = 0; batch_start < h_valid_tile_count; batch_start += batch)
    {
        int batch_end = min(batch_start + batch, h_valid_tile_count);
        int batch_size = batch_end - batch_start;

        int block_dim = 1024;
        int grid_dim = ceil((float)batch_size / block_dim);

        microscheduler_init <<<grid_dim, block_dim>>> (mat, *dev_microtiles, d_valid_tile_idx, batch_start, batch_size, h_valid_tile_count, n_microtiles_col);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    HANDLE_ERROR(cudaFree(d_valid_tile_count));
    HANDLE_ERROR(cudaFree(d_valid_tile_idx));

    return h_valid_tile_count;
}


#endif