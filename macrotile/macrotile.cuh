#ifndef __MACROTILE_CUH__
#define __MACROTILE_CUH__

#include "../CPU_includes/dtypes.hpp"
#include "../commons/handle.cuh"
#include "../CPU_includes/tile_size.hpp"

__global__ void macroscheduler_init(const csr_matrix *mat, macrotile_metadata *dev_macrotiles, int *d_valid_tile_idx, int batch_start, int batch_size, int d_valid_tile_count, int n_macrotiles_col)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int valid_idx = idx + batch_start;

    if(valid_idx < batch_start + batch_size && valid_idx < d_valid_tile_count)
    {
        int index = d_valid_tile_idx[valid_idx];

        int row = index / n_macrotiles_col;
        int col = index % n_macrotiles_col;

        int start_row = row * TILE_HEIGHT;
        int start_col = col * TILE_WIDTH;

        int row_ptr_idx = 0;
        int row_ptr_val = 0;

        dev_macrotiles[valid_idx].matrix.row_ptr[0] = 0;

        for(int i = start_row; i < min(start_row + TILE_HEIGHT, mat->num_rows); i++)
        {
            for(int j = mat->row_ptr[i]; j < mat->row_ptr[i + 1]; j++)
            {
                int col_idx = mat->col_idx[j];
                if(col_idx >= start_col && col_idx < min(start_col + TILE_WIDTH, mat->num_cols))
                {
                    dev_macrotiles[valid_idx].matrix.col_idx[row_ptr_val] = col_idx - start_col;
                    dev_macrotiles[valid_idx].matrix.values[row_ptr_val] = mat->values[j];
                    row_ptr_val++;
                }
            }
            row_ptr_idx++;
            dev_macrotiles[valid_idx].matrix.row_ptr[row_ptr_idx] = row_ptr_val;
        }

        dev_macrotiles[valid_idx].matrix.num_nonzeros = row_ptr_val;
        dev_macrotiles[valid_idx].matrix.num_rows = min(TILE_HEIGHT, mat->num_rows - start_row);
        dev_macrotiles[valid_idx].matrix.num_cols = min(TILE_WIDTH, mat->num_cols - start_col);

        dev_macrotiles[valid_idx].row_base = start_row;
        dev_macrotiles[valid_idx].col_base = start_col;
        dev_macrotiles[valid_idx].num_elements = row_ptr_val;
        dev_macrotiles[valid_idx].num_microtiles = 0;

        dev_macrotiles[valid_idx].res_matrix.num_rows = min(TILE_HEIGHT, mat->num_rows - start_row);
        dev_macrotiles[valid_idx].res_matrix.num_cols = -1;
        dev_macrotiles[valid_idx].res_matrix.num_nonzeros = 0;
   
    }
}


__global__ void count_valid_macrotiles(const csr_matrix *mat, int *d_valid_tile_count, int *d_valid_tile_idx, int n_macrotiles_col, int n_macrotiles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n_macrotiles)
    {
        int row = idx / n_macrotiles_col;
        int col = idx % n_macrotiles_col;

        int start_row = row * TILE_HEIGHT;
        int start_col = col * TILE_WIDTH;

        int valid_tile = 0;
        for(int i = start_row; i < min(start_row + TILE_HEIGHT, mat->num_rows); i++)
        {
            for(int j = mat->row_ptr[i]; j < mat->row_ptr[i + 1]; j++)
            {
                int col_idx = mat->col_idx[j];
                if(col_idx >= start_col && col_idx < min(start_col + TILE_WIDTH, mat->num_cols))
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

 
int macroscheduler(const csr_matrix *mat, macrotile_metadata **dev_macrotiles)
{
    int h_rows, h_cols;
    HANDLE_ERROR(cudaMemcpy(&h_rows, &mat->num_rows, sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&h_cols, &mat->num_cols, sizeof(int), cudaMemcpyDeviceToHost));

    int n_macrotiles_row = ceil((float)h_rows / TILE_HEIGHT);
    int n_macrotiles_col = ceil((float)h_cols / TILE_WIDTH);
    int n_macrotiles = n_macrotiles_row * n_macrotiles_col;

    int *d_valid_tile_count, h_valid_tile_count = 0;
    int *d_valid_tile_idx;

    HANDLE_ERROR(cudaMalloc((void **)&d_valid_tile_count, sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&d_valid_tile_idx, n_macrotiles * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_valid_tile_count, &h_valid_tile_count, sizeof(int), cudaMemcpyHostToDevice));

    int block_size_cvmt = 1024;
    int grid_size_cvmt = ceil((float)n_macrotiles / block_size_cvmt);
    count_valid_macrotiles<<<grid_size_cvmt, block_size_cvmt>>>(mat, d_valid_tile_count, d_valid_tile_idx, n_macrotiles_col, n_macrotiles);
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaMemcpy(&h_valid_tile_count, d_valid_tile_count, sizeof(int), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaMalloc((void **)dev_macrotiles, h_valid_tile_count * sizeof(macrotile_metadata)));
    macrotile_metadata *h_macrotiles = new macrotile_metadata[h_valid_tile_count];

    for(int i = 0; i < h_valid_tile_count; i++)
    {
        csr_matrix temp; 

        int *d_rowptr, *d_colidx;
        float *d_values;
        HANDLE_ERROR(cudaMalloc((void **)&d_rowptr, (TILE_HEIGHT + 1) * sizeof(int)));
        HANDLE_ERROR(cudaMemset(d_rowptr, 0, (TILE_HEIGHT + 1) * sizeof(int)));
        HANDLE_ERROR(cudaMalloc((void **)&d_colidx, TILE_HEIGHT * TILE_WIDTH * sizeof(int)));
        HANDLE_ERROR(cudaMalloc((void **)&d_values, TILE_HEIGHT * TILE_WIDTH * sizeof(float)));
        
        temp.num_rows = TILE_HEIGHT;
        temp.num_cols = TILE_WIDTH;
        temp.num_nonzeros = 0;
        temp.row_ptr = d_rowptr;
        temp.col_idx = d_colidx;
        temp.values = d_values;

        h_macrotiles[i].matrix = temp;
    }

    HANDLE_ERROR(cudaMemcpy(*dev_macrotiles, h_macrotiles, h_valid_tile_count * sizeof(macrotile_metadata), cudaMemcpyHostToDevice));
    delete[] h_macrotiles;

    int pitch = n_macrotiles_col;
    int batch = pow(2,16);

    for(int batch_start = 0; batch_start < h_valid_tile_count; batch_start += batch)
    {
        int batch_end = min(batch_start + batch, h_valid_tile_count);
        int batch_size = batch_end - batch_start;

        int block_dim = 1024;
        int grid_dim = ceil((float)batch_size / block_dim);
        macroscheduler_init <<<grid_dim, block_dim>>> (mat, *dev_macrotiles, d_valid_tile_idx, batch_start, batch_size, h_valid_tile_count, n_macrotiles_col);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    HANDLE_ERROR(cudaFree(d_valid_tile_count));
    HANDLE_ERROR(cudaFree(d_valid_tile_idx));   

    return h_valid_tile_count;
}
#endif