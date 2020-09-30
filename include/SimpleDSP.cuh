#pragma once

#include <cufft.h>

#include "my_cuda_utils.hpp"
#include "my_cufft_utils.hpp"

#include "simple_dsp_kernels.cuh"

class SimpleDSP {
   public:
      SimpleDSP():
         samples( nullptr ),
         frequencies( nullptr ),
         con_sqrs( nullptr ),
         psds( nullptr ),
         num_bits(0),
         num_samples(0),
         log10num_con_sqrs(0) {}

      SimpleDSP( int new_num_bits );
      void operator()();
      ~SimpleDSP();

   private:
      cufftComplex* samples;
      cufftComplex* frequencies;
      cufftComplex* con_sqrs;
      float* psds;

      int num_bits;
      int num_samples;
      float log10num_con_sqrs;
};
