#ifndef __FREE_HPP__
#define __FREE_HPP__

#include "dtypes.hpp"

void free_coo_matrix(coo_matrix* mat)
{
    delete[] mat->row_idx;
    delete[] mat->col_idx;
    delete[] mat->values;
}

void free_csr_matrix(csr_matrix* mat)
{
    delete[] mat->row_ptr;
    delete[] mat->col_idx;
    delete[] mat->values;
}

void free_microtile_metadata(microtile_metadata* microtile)
{
    free_csr_matrix(&microtile->matrix);
    free_csr_matrix(&microtile->res_matrix);
}

void free_macrotile_metadata(macrotile_metadata* macrotile)
{
    for (int i = 0; i < macrotile->num_microtiles; i++)
    {
        free_microtile_metadata(macrotile->microtiles[i]);
    }
    delete[] macrotile->microtiles;
    free_csr_matrix(&macrotile->matrix);
    free_csr_matrix(&macrotile->res_matrix);
}

#endif // __FREE_HPP__