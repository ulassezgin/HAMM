#ifndef __MATRIX_OPS_HPP__
#define __MATRIX_OPS_HPP__

#include "dtypes.hpp"

void transpose(const coo_matrix& mat, coo_matrix* transpose_mat)
{
	transpose_mat->num_rows = mat.num_cols;
	transpose_mat->num_cols = mat.num_rows;
	transpose_mat->num_nonzeros = mat.num_nonzeros;
    transpose_mat->row_idx = new int[mat.num_nonzeros];
    transpose_mat->col_idx = new int[mat.num_nonzeros];
    transpose_mat->values = new float[mat.num_nonzeros];
    for (int i = 0; i < mat.num_nonzeros; i++)
    {
        transpose_mat->row_idx[i] = mat.col_idx[i];
        transpose_mat->col_idx[i] = mat.row_idx[i];
        transpose_mat->values[i] = mat.values[i];
    }
}

void convert(const coo_matrix* coo, csr_matrix* csr)
{
    csr->num_rows = coo->num_rows;
    csr->num_cols = coo->num_cols;
    csr->num_nonzeros = coo->num_nonzeros;
    csr->row_ptr = new int[coo->num_rows + 1];
    csr->col_idx = new int[coo->num_nonzeros];
    csr->values = new float[coo->num_nonzeros];
    for (int i = 0; i < coo->num_rows + 1; i++)
    {
        csr->row_ptr[i] = 0;
    }
    for (int i = 0; i < coo->num_nonzeros; i++)
    {
        csr->row_ptr[coo->row_idx[i]]++;
    }

    for (int i = 0; i < coo->num_rows; i++)
    {
        csr->row_ptr[i + 1] += csr->row_ptr[i];
    }

    int *temp = new int[coo->num_rows];
    for (int i = 0; i < coo->num_rows; i++)
    {
        temp[i] = 0;
    }

    for (int i = 0; i < coo->num_nonzeros; i++)
    {
        int row = coo->row_idx[i] - 1;
        int dest = csr->row_ptr[row] + temp[row];
        csr->col_idx[dest] = coo->col_idx[i] - 1;
        csr->values[dest] = coo->values[i];
        temp[row]++;
    }
    delete[] temp;
}


void convert(const csr_matrix* csr, coo_matrix* coo)
{
    coo->num_rows = csr->num_rows;
    coo->num_cols = csr->num_cols;
    coo->num_nonzeros = csr->num_nonzeros;
    coo->row_idx = new int[csr->num_nonzeros];
    coo->col_idx = new int[csr->num_nonzeros];
    coo->values = new float[csr->num_nonzeros];
    int idx = 0;
    for (int i = 0; i < csr->num_rows; i++)
    {
        for (int j = csr->row_ptr[i]; j < csr->row_ptr[i + 1]; j++)
        {
            coo->row_idx[idx] = i;
            coo->col_idx[idx] = csr->col_idx[j];
            coo->values[idx] = csr->values[j];
            idx++;
        }
    }
}


#endif // __MATRIX_OPS_HPP__