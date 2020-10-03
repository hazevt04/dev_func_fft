#include "my_cufft_utils.hpp"

void print_cufftComplexes(const cufftComplex* vals,
   const int num_vals,
   const char* prefix,
   const char* delim,
   const char* suffix ) {

   std::cout << prefix;
   for (int index = 0; index < num_vals; ++index) {
      std::cout << "{" << vals[index].x << ", " << vals[index].y << "}" << ((index == num_vals - 1) ? "\n" : delim);
   }
   std::cout << suffix;
}

// end of C++ file for my_cufft_utils
