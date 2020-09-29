#ifndef __MANAGED_ALLOCATOR_HOST_H__
#define __MANAGED_ALLOCATOR_HOST_H__

#include <vector>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cufft.h>

#include "utils.h"

template<class T>
class managed_allocator_host {
   public:
      using value_type = T;
      using reference = T&;
      using const_reference = const T&;

      managed_allocator_host():
         is_allocated( false ) {}

      template<class U>
      managed_allocator_host(const managed_allocator_host<U>&) {
         is_allocated = false;
      }
  
      value_type* allocate(size_t n) {
         value_type* result = nullptr;
         if ( !is_allocated ) {
  
            cudaError_t error = cudaMallocManaged( &result, n*sizeof(T), cudaMemAttachHost );
     
            if(error != cudaSuccess) {
               throw std::runtime_error( "managed_allocator_host::allocate(): cudaMallocManaged failed" );
            }
            is_allocated = true;
         }
  
         return result;
      }
  
      void deallocate(value_type* ptr, size_t) {}
   private:
      bool is_allocated;
};

template<class T1, class T2>
bool operator==(const managed_allocator_host<T1>&, const managed_allocator_host<T2>&) {
   return true;
}

template<class T1, class T2>
bool operator!=(const managed_allocator_host<T1>& lhs, const managed_allocator_host<T2>& rhs) {
   return !(lhs == rhs);
}

typedef std::vector<cufftComplex, managed_allocator_host<cufftComplex>> managed_host_vector;
#endif // end of #ifndef __MANAGED_ALLOCATOR_HOST_H__
