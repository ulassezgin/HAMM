#ifndef __MACROTILE_2D_CUH__
#define __MACROTILE_2D_CUH__

#include "../CPU_includes/dtypes.hpp"
#include "../commons/handle.cuh"
#include "../CPU_includes/tile_size.hpp"

__global__ void count_valid_macrotiles(const csr_matrix *mat, int *d_valid_tile_count, int *d_valid_tile_idx, int n_macrotiles_col, int n_macrotiles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_macrotiles) {
        int row = idx / n_macrotiles_col;
        int col = idx % n_macrotiles_col;

        int start_row = row * TILE_HEIGHT;
        int start_col = col * TILE_WIDTH;

        int valid_tile = 0;
        for (int i = start_row; i < min(start_row + TILE_HEIGHT, mat->num_rows); i++) {
            for (int j = mat->row_ptr[i]; j < mat->row_ptr[i + 1]; j++) {
                int col_idx = mat->col_idx[j];
                if (col_idx >= start_col && col_idx < min(start_col + TILE_WIDTH, mat->num_cols)) {
                    valid_tile++;
                }
            }
        }

        if (valid_tile > 0) {
            int old = atomicAdd(d_valid_tile_count, 1);
            d_valid_tile_idx[old] = idx;
        }
    }
}

__global__ void macroscheduler_init(const csr_matrix *mat, macrotile_metadata **dev_macrotiles, int *d_valid_tile_idx, int batch_start, int batch_size, int d_valid_tile_count, int n_macrotiles_col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int valid_idx = idx + batch_start;

    if (valid_idx < batch_start + batch_size && valid_idx < d_valid_tile_count) {
        int index = d_valid_tile_idx[valid_idx];
        int row = index / n_macrotiles_col;
        int col = index % n_macrotiles_col;

        int start_row = row * TILE_HEIGHT;
        int start_col = col * TILE_WIDTH;

        // Populate the macrotile
        macrotile_metadata *tile = &dev_macrotiles[row][col];
        tile->matrix.row_ptr[0] = 0;
        int row_ptr_idx = 0;
        int row_ptr_val = 0;

        for (int i = start_row; i < min(start_row + TILE_HEIGHT, mat->num_rows); i++) {
            for (int j = mat->row_ptr[i]; j < mat->row_ptr[i + 1]; j++) {
                int col_idx = mat->col_idx[j];
                if (col_idx >= start_col && col_idx < min(start_col + TILE_WIDTH, mat->num_cols)) {
                    tile->matrix.col_idx[row_ptr_val] = col_idx - start_col;
                    tile->matrix.values[row_ptr_val] = mat->values[j];
                    row_ptr_val++;
                }
            }
            row_ptr_idx++;
            tile->matrix.row_ptr[row_ptr_idx] = row_ptr_val;
        }

        tile->matrix.num_nonzeros = row_ptr_val;
        tile->matrix.num_rows = min(TILE_HEIGHT, mat->num_rows - start_row);
        tile->matrix.num_cols = min(TILE_WIDTH, mat->num_cols - start_col);

        tile->row_base = start_row;
        tile->col_base = start_col;
        tile->num_elements = row_ptr_val;
        tile->res_matrix.num_rows = min(TILE_HEIGHT, mat->num_rows - start_row);
        tile->res_matrix.num_cols = min(TILE_WIDTH, mat->num_cols - start_col);
        tile->res_matrix.num_nonzeros = 0;
    }
}

int macroscheduler(const csr_matrix *mat, macrotile_2d *dev_macrotiles_2d) {
    int h_rows, h_cols;
    HANDLE_ERROR(cudaMemcpy(&h_rows, &mat->num_rows, sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&h_cols, &mat->num_cols, sizeof(int), cudaMemcpyDeviceToHost));

    int n_macrotiles_row = ceil((float) h_rows / TILE_HEIGHT);
    int n_macrotiles_col = ceil((float) h_cols / TILE_WIDTH);
    int n_macrotiles = n_macrotiles_row * n_macrotiles_col;

    int *d_valid_tile_count, h_valid_tile_count = 0;
    int *d_valid_tile_idx;

    HANDLE_ERROR(cudaMalloc((void**)&d_valid_tile_count, sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&d_valid_tile_idx, n_macrotiles * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_valid_tile_count, &h_valid_tile_count, sizeof(int), cudaMemcpyHostToDevice));

    int block_size_cvmt = 1024;
    int grid_size_cvmt = ceil((float)n_macrotiles / block_size_cvmt);
    count_valid_macrotiles<<<grid_size_cvmt, block_size_cvmt>>>(mat, d_valid_tile_count, d_valid_tile_idx, n_macrotiles_col, n_macrotiles);
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaMemcpy(&h_valid_tile_count, d_valid_tile_count, sizeof(int), cudaMemcpyDeviceToHost));

    // Allocate 2D array on host and device
    macrotile_metadata **h_macrotiles = new macrotile_metadata*[n_macrotiles_row];
    for (int i = 0; i < n_macrotiles_row; i++) {
        h_macrotiles[i] = new macrotile_metadata[n_macrotiles_col];
    }
    HANDLE_ERROR(cudaMalloc((void**)&dev_macrotiles_2d->tiles, n_macrotiles_row * sizeof(macrotile_metadata*)));
    for (int i = 0; i < n_macrotiles_row; i++) {
        HANDLE_ERROR(cudaMalloc((void**)&dev_macrotiles_2d->tiles[i], n_macrotiles_col * sizeof(macrotile_metadata)));
    }

    // Copy host 2D array to device
    for (int i = 0; i < n_macrotiles_row; i++) {
        HANDLE_ERROR(cudaMemcpy(dev_macrotiles_2d->tiles[i], h_macrotiles[i], n_macrotiles_col * sizeof(macrotile_metadata), cudaMemcpyHostToDevice));
    }

    dev_macrotiles_2d->num_rows = n_macrotiles_row;
    dev_macrotiles_2d->num_cols = n_macrotiles_col;

    int batch = pow(2, 13);
    int block_dim = 1024;

    for (int batch_start = 0; batch_start < h_valid_tile_count; batch_start += batch) {
        int batch_end = min(batch_start + batch, h_valid_tile_count);
        int batch_size = batch_end - batch_start;
        int grid_dim = ceil((float)batch_size / block_dim);

        macroscheduler_init<<<grid_dim, block_dim>>>(mat, dev_macrotiles_2d->tiles, d_valid_tile_idx, batch_start, batch_size, h_valid_tile_count, n_macrotiles_col);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    HANDLE_ERROR(cudaFree(d_valid_tile_count));
    HANDLE_ERROR(cudaFree(d_valid_tile_idx));

    // Free host 2D array
    for (int i = 0; i < n_macrotiles_row; i++) {
        delete[] h_macrotiles[i];
    }
    delete[] h_macrotiles;

    return h_valid_tile_count;
}


#endif // __MACROTILE_2D_CUH__