#include "CPU_includes/dtypes.hpp"
#include "CPU_includes/file_ops.hpp"
#include "CPU_includes/matrix_ops.hpp"
#include "CPU_includes/tile_size.hpp"
#include "CPU_includes/prints.hpp"
#include "CPU_includes/free.hpp"

#include "commons/spgemm.cuh"

int main(int argc, char **argv)
{
    clock_t start, end;
    duration_metadata *duration = new duration_metadata;

    // Check if the input is correct
    if(argc != 3)
    {
        cout << "Usage: " << argv[0] << " <input file> <output file>" << endl;
        return EXIT_FAILURE;
    }

    string filepath = "./mtx_files/" + string(argv[1]) + ".mtx";
    coo_matrix coo_mat, coo_T_mat;
    csr_matrix csr_mat, csr_T_mat, csr_res_mat;

    bool is_binary = (string(argv[2]) == "true") ? true : false;
    coo_mat.is_bin_matrix = is_binary;


    printf("Starting...\n");

    // Read the matrix from the file
    start = clock();
    load_mtx_coo(filepath.c_str(), coo_mat);
    end = clock();
    duration->reading_duration = (double)(end - start) / CLOCKS_PER_SEC;

    // Transpose the matrix and convert it to csr format
    start = clock();

    transpose(coo_mat, &coo_T_mat);

    convert(&coo_mat, &csr_mat);
    convert(&coo_T_mat, &csr_T_mat);
    end = clock();
    duration->preparing_duration = (double)(end - start) / CLOCKS_PER_SEC;

    // Multiply the matrices

    spgemm(&csr_mat, &csr_T_mat, &csr_res_mat, duration);
    HANDLE_ERROR(cudaDeviceSynchronize());

    delete duration;
    free_coo_matrix(&coo_mat);
    free_coo_matrix(&coo_T_mat);

    printf("Done\n");
    return EXIT_SUCCESS;
}

// int main(int argc, char **argv)
// {
//     clock_t start, end;
//     duration_metadata *duration = new duration_metadata;

//     // Check if the input is correct
//     if(argc != 3)
//     {
//         cout << "Usage: " << argv[0] << " <input file> <output file>" << endl;
//         return EXIT_FAILURE;
//     }

//     string filepath = "./mtx_files/" + string(argv[1]) + ".mtx";
//     coo_matrix coo_mat, coo_T_mat;
//     csr_matrix csr_mat, csr_T_mat, csr_res_mat;

//     bool is_binary = (string(argv[2]) == "true") ? true : false;
//     coo_mat.is_bin_matrix = is_binary;


//     printf("Starting...\n");

//     // Read the matrix from the file
//     start = clock();
//     load_mtx_coo(filepath.c_str(), coo_mat);
//     end = clock();
//     duration->reading_duration = (double)(end - start) / CLOCKS_PER_SEC;

//     // Transpose the matrix and convert it to csr format
//     start = clock();
//     coo_T_mat.is_bin_matrix = true;
//     load_mtx_coo("./mtx_files/B.mtx", coo_T_mat);

//     convert(&coo_mat, &csr_mat);
//     convert(&coo_T_mat, &csr_T_mat);
//     end = clock();
//     duration->preparing_duration = (double)(end - start) / CLOCKS_PER_SEC;

//     // Multiply the matrices

//     spgemm(&csr_mat, &csr_T_mat, &csr_res_mat, duration);


//     delete duration;
//     free_coo_matrix(&coo_mat);
//     free_coo_matrix(&coo_T_mat);

//     printf("Done\n");
//     return EXIT_SUCCESS;
// }