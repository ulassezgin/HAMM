#ifndef __LOCK_CUH__
#define __LOCK_CUH__

#include "../commons/handle.cuh"

struct Lock {
    int *mutex;
    Lock( void ) {
        HANDLE_ERROR( cudaMalloc( (void**)&mutex, sizeof(int) ) );
        HANDLE_ERROR( cudaMemset( mutex, 0, sizeof(int) ) );
    }
    ~Lock( void ) {
        cudaFree( mutex );
    }
    __device__ void lock( void ) {
        while( atomicCAS( mutex, 0, 1 ) != 0 );
	__threadfence();
    }
    __device__ void unlock( void ) {
        __threadfence();
        atomicExch( mutex, 0 );
    }
};

#endif