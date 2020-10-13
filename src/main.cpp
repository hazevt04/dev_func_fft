// C++ File for main

#include "my_utils.hpp"
#include "my_file_io_funcs.hpp"

#include "SimpleDSP.cuh"

int main() {
   try {
      bool debug = true;
      int num_samples = 2 * FFT_SIZE;

      SimpleDSP simple_dsp( num_samples, debug );
      simple_dsp.run();

      return EXIT_SUCCESS;

   } catch (std::exception& ex) {
      std::cout << "ERROR: " << ex.what() << "\n";
      return EXIT_FAILURE;
   }
}
