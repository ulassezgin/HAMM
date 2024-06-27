#ifndef __PRINTS_HPP__
#define __PRINTS_HPP__

#include <iostream>
#include "dtypes.hpp"

using namespace std;

void print_csr(const csr_matrix &mat)
{
    cout << "CSR Matrix" << endl;
    cout << "Number of rows: " << mat.num_rows << endl;
    cout << "Number of columns: " << mat.num_cols << endl;
    cout << "Number of nonzeros: " << mat.num_nonzeros << endl;
    cout << "Row pointer: ";
    for (int i = 0; i < mat.num_rows + 1; i++)
    {
        cout << mat.row_ptr[i] << " ";
    }
    cout << endl;
    cout << "Column index: ";
    for (int i = 0; i < mat.num_nonzeros; i++)
    {
        cout << mat.col_idx[i] << " ";
    }
    cout << endl;
    cout << "Values: ";
    for (int i = 0; i < mat.num_nonzeros; i++)
    {
        cout << mat.values[i] << " ";
    }
    cout << endl;
}

#endif // __PRINTS_HPP__