// C++ File for main
#include "my_utils.hpp"
#include "SimpleDSP.cuh"

#ifndef NUM_FFT_SIZE_BITS
#define NUM_FFT_SIZE_BITS 6
#endif

#ifndef FFT_SIZE
#define FFT_SIZE (1u << (NUM_FFT_SIZE_BITS))
#endif

int get_num_samples( const char* input_string, const bool debug ) {
   try {
      dout << "input_string = " << std::string{input_string} << "\n"; 
      char* end_ptr = nullptr;
      int num_samples = (int)strtoul( input_string, &end_ptr, 10 );
      if ( *end_ptr != '\0' ) {
         throw std::runtime_error{ std::string{"Invalid input: "} + std::string{input_string} };
      }      
      if ( num_samples < 256 ) {
         throw std::runtime_error{ std::string{"Input for num_samples, "} + std::to_string(FFT_SIZE) + std::string{input_string} + 
            std::string{", must be greater than 256 (at least four 64 sample FFTs)."} };
      }
      if ( (num_samples & (FFT_SIZE-1)) != 0 ) {
         throw std::runtime_error{ std::string{"Input for num_samples, "} + std::to_string(FFT_SIZE) + std::string{input_string} + 
            std::string{", must be a multiple of FFT_SIZE (="} + std::to_string(FFT_SIZE) + std::string{")"} };
      }

      return num_samples;

   } catch (std::exception& ex) {
      throw std::runtime_error{ std::string{__func__} + std::string{"(): ERROR: "} + ex.what() };
   }
}


int main( int argc, char* argv[] ) {
   try {
      bool debug = false;
      int num_samples = 4*FFT_SIZE;

      if ( argc > 1 ) {
         num_samples = get_num_samples( argv[1], debug );
      }

      SimpleDSP simple_dsp( num_samples, debug );
      simple_dsp.run();

      return EXIT_SUCCESS;

   } catch (std::exception& ex) {
      std::cout << "ERROR: " << ex.what() << "\n";
      return EXIT_FAILURE;
   }
}
