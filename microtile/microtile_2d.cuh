#ifndef __MICROTILE_2D_CUH__
#define __MICROTILE_2D_CUH__

#include "../CPU_includes/dtypes.hpp"
#include "../commons/handle.cuh"
#include "../CPU_includes/tile_size.hpp"
#include "../CPU_includes/prints.hpp"
#include "../commons/print_dev.cuh"



__global__ void count_valid_microtiles(const csr_matrix *mat, int *d_valid_tile_count, int *d_valid_tile_idx, int n_microtiles_col, int n_microtiles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_microtiles) {
        int row = idx / n_microtiles_col;
        int col = idx % n_microtiles_col;

        int start_row = row * T_TILE_HEIGHT;
        int start_col = col * T_TILE_WIDTH;

        int valid_tile = 0;
        for (int i = start_row; i < min(start_row + T_TILE_HEIGHT, mat->num_rows); i++) {
            for (int j = mat->row_ptr[i]; j < mat->row_ptr[i + 1]; j++) {
                int col_idx = mat->col_idx[j];
                if (col_idx >= start_col && col_idx < min(start_col + T_TILE_WIDTH, mat->num_cols)) {
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

__global__ void microscheduler_init(const csr_matrix *mat, microtile_metadata **dev_microtiles, int *d_valid_tile_idx, int batch_start, int batch_size, int d_valid_tile_count, int n_microtiles_col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int valid_idx = idx + batch_start;

    if (valid_idx < batch_start + batch_size && valid_idx < d_valid_tile_count) {
        int index = d_valid_tile_idx[valid_idx];
        int row = index / n_microtiles_col;
        int col = index % n_microtiles_col;

        int start_row = row * T_TILE_HEIGHT;
        int start_col = col * T_TILE_WIDTH;

        // Access the correct microtile
        microtile_metadata *tile = dev_microtiles[row] + col;

        tile->matrix.row_ptr[0] = 0;
        int row_ptr_idx = 0;
        int row_ptr_val = 0;

        for (int i = start_row; i < min(start_row + T_TILE_HEIGHT, mat->num_rows); i++) {
            for (int j = mat->row_ptr[i]; j < mat->row_ptr[i + 1]; j++) {
                int col_idx = mat->col_idx[j];
                if (col_idx >= start_col && col_idx < min(start_col + T_TILE_WIDTH, mat->num_cols)) {
                    tile->matrix.col_idx[row_ptr_val] = col_idx - start_col;
                    tile->matrix.values[row_ptr_val] = mat->values[j];
                    row_ptr_val++;
                }
            }
            row_ptr_idx++;
            tile->matrix.row_ptr[row_ptr_idx] = row_ptr_val;
        }

        tile->matrix.num_nonzeros = row_ptr_val;
        tile->matrix.num_rows = min(T_TILE_HEIGHT, mat->num_rows - start_row);
        tile->matrix.num_cols = min(T_TILE_WIDTH, mat->num_cols - start_col);

        tile->row_base = start_row;
        tile->col_base = start_col;
        tile->num_elements = row_ptr_val;
        tile->res_matrix.num_rows = min(T_TILE_HEIGHT, mat->num_rows - start_row);
        tile->res_matrix.num_cols = min(T_TILE_WIDTH, mat->num_cols - start_col);
        tile->res_matrix.num_nonzeros = 0;
    }
}

int microscheduler(const csr_matrix *mat, microtile_2d *dev_microtiles_2d) {
    int h_micro_rows, h_micro_cols;
    HANDLE_ERROR(cudaMemcpy(&h_micro_rows, &mat->num_rows, sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&h_micro_cols, &mat->num_cols, sizeof(int), cudaMemcpyDeviceToHost));

    int n_microtiles_col = (h_micro_cols + T_TILE_WIDTH - 1) / T_TILE_WIDTH;
    int n_microtiles_row = (h_micro_rows + T_TILE_HEIGHT - 1) / T_TILE_HEIGHT;
    int n_microtiles = n_microtiles_col * n_microtiles_row;

    
}

#endif  // __MICROTILE_2D_CUH__