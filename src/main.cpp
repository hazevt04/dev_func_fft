// C++ File for main

#include "my_utils.hpp"
#include "my_file_io_funcs.hpp"

#include "SimpleDSP.cuh"
#include <string>

int get_num_samples( const char* input_string, const bool debug ) {
   try {
      dout << "input_string = " << std::string{input_string} << "\n"; 
      char* end_ptr = nullptr;
      int num_samples = (int)strtoul( input_string, &end_ptr, 10 );
      if ( *end_ptr != '\0' ) {
         throw std::runtime_error{ std::string{"Invalid input: "} + std::string{input_string} };
      }      
      if ( ( ( num_samples & 0x1f ) != 0 ) || ( num_samples < 1 ) ) {
         throw std::runtime_error{ std::string{"Input "} + std::string{input_string} + std::string{" must be greater than 0 and a multiple of FFTSIZE:"} + std::to_string(FFT_SIZE) };
      }
      return num_samples;
   } catch (std::exception& ex) {
      throw std::runtime_error{ std::string{__func__} + std::string{"(): ERROR: "} + ex.what() };
   }
}


int main( int argc, char* argv[] ) {
   try {
      bool debug = true;
      int num_samples = FFT_SIZE;

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
