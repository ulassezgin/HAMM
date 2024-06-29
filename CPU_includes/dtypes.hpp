#ifndef __DTYPES_HPP__
#define __DTYPES_HPP__

struct csr_matrix
{
    int *row_ptr;
    int *col_idx;
    float *values;
    int num_rows;
    int num_cols;
    int num_nonzeros;
};

struct coo_matrix
{
    int *row_idx;
    int *col_idx;
    float *values;
    int num_rows;
    int num_cols;
    int num_nonzeros;
    bool is_bin_matrix;
};

struct duration_metadata
{
    double reading_duration;
    double preparing_duration;
	double allocating_duration;
	double prediction_duration;
	double kernel_duration;
	double microtile_duration;
    double microtile_hash_duration;
    double macrotile_hash_duration;
    double microtile_hash_duration_c;
    double macrotile_hash_duration_c;
	double macrotile_duration;
	double match_duration;
    double micromap_duration;
    double esc_duration;
};

struct microtile_metadata
{
    int num_elements;
    unsigned int row_base;
    unsigned int col_base;
    csr_matrix matrix;
    csr_matrix res_matrix;
};

struct macrotile_metadata
{
    int num_microtiles;
    int num_elements;
    unsigned int row_base;
    unsigned int col_base;
    microtile_metadata **microtiles;
    csr_matrix matrix;
    csr_matrix res_matrix;
};


struct microtile_hash_node
{
    microtile_metadata *data;
    microtile_hash_node *next;
};

struct microtile_hash_table
{
    size_t count;
    microtile_hash_node **entries;
    microtile_hash_node *pool;
    unsigned int *elm_per_bucket;
};

struct macrotile_hash_node
{
    macrotile_metadata *data;
    macrotile_hash_node *next;
};

struct macrotile_hash_table
{
    size_t count;
    macrotile_hash_node **entries;
    macrotile_hash_node *pool;
    unsigned int *elm_per_bucket;
};

#endif // __DTYPES_HPP__