#ifndef __DTYPES_HASH_HPP__
#define __DTYPES_HASH_HPP__

#include "dtypes.hpp"

struct microtile_hash_node
{
    microtile_metadata *data;
    microtile_hash_node *next;
};
#endif // __DTYPES_HASH_HPP__