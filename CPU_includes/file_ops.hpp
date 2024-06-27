#ifndef __FILE_OPS_HPP__
#define __FILE_OPS_HPP__

#include <iostream>
#include <fstream>
#include <sstream>


#include "dtypes.hpp"

void load_mtx_coo(const char *filename, coo_matrix &coo)
{
    std::string line;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file for reading");
    }
    getline (file, line);
    std::istringstream iss(line);
    iss >> coo.num_rows >> coo.num_cols >> coo.num_nonzeros;
    coo.row_idx = new int[coo.num_nonzeros];
    coo.col_idx = new int[coo.num_nonzeros];
    coo.values = new float[coo.num_nonzeros];
    int idx = 0;
    while (getline(file, line))
    {
        std::istringstream iss(line);
        iss >> coo.row_idx[idx] >> coo.col_idx[idx];
        if(coo.is_bin_matrix)
        {
            coo.values[idx] = 1;
        }
        else
        {
            iss >> coo.values[idx];
        }
        idx++;
    }
    file.close();
}

#endif // __FILE_OPS_HPP__