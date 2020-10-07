// C++ File for main

#include "my_utils.hpp"
#include "my_file_io_funcs.hpp"

#include "SimpleDSP.cuh"


// 64 Input samples
// [ 28.-20.j  16.-28.j   8.-16.j   8.-16.j  12.-12.j  -8.-20.j  -8.-28.j
  // -4.-16.j -32. -8.j -24. -8.j -28. -8.j -24. +8.j -24. +8.j  -4. +4.j
 // -16.+20.j -12. +8.j  -8. -4.j  12. +4.j  20.+16.j  -4.+12.j   8. +8.j
  // 20. -4.j  16. +0.j   8. -4.j  20. -8.j  12.-20.j  12.-12.j  -4.-28.j
  // 12.-12.j   4.-16.j  -4. -8.j -12. +0.j -24.-20.j -24. -4.j -16. +4.j
 // -16. +4.j -28. +0.j -24. +0.j -12. +0.j  -8. +8.j  -4.+12.j -12.+20.j
   // 4.+12.j  12.+16.j  16.+20.j  20.+20.j  28. +0.j  32. -8.j  16. -4.j
  // 16.-16.j  20. +4.j   8.-16.j  16.-28.j  12.-20.j  -8. -8.j -12.-28.j
 // -20.-12.j -20.-16.j -16.-12.j -24. -4.j -16.+12.j -16. +4.j -16. +4.j
 // -16.+24.j]

// Expected FFT Outputs (numpy.fft.fft() and numpy.fft.fftshift())
// [-132.        -240.j          -85.94402282 -36.09659629j
  // -81.35411222  -0.91108112j   84.94564024-187.26012091j
    // 6.83723648 -27.00435032j  -24.07475218-150.84853419j
  // -50.59636575 +29.90044203j   -4.80805541 -73.01125682j
  // -48.76955262 -69.85786438j  -51.8021418  -59.20386129j
  // -23.61138123 -64.37262007j  104.73821491 -43.24309453j
  // -16.74151197-175.16605276j  -27.61689685 +76.52360648j
 // -125.05633239 +34.02646551j   80.79678323 -48.6777785j
  // -72.         -84.j          -45.52931596 +54.15275963j
  // 125.79232289 +56.25780044j   57.22641    -15.69231809j
  // -83.99643814 -80.9538002j   -18.54727802-155.81469514j
   // 13.03395819 -97.52749052j   27.10334384  +5.76524363j
  // -53.45584412 +17.74011537j  -38.93389753  -6.62313144j
  // -43.226213  -105.64023583j   18.01908495 +45.23527691j
   // -2.87496529 -25.8473992j    55.45470935 -16.56101907j
  // 123.45218242-103.60809949j  -11.39947084 -14.98496058j
   // 44.         +40.j          -10.92177239 +15.03216534j
  // -23.93041029  +3.57839443j  115.6990709  -37.44050401j
   // 25.73130601 -43.05452418j   54.0160334   -1.24250993j
   // 29.06745814 -36.73234781j   36.88721264  -0.69300137j
   // 24.76955262 -98.14213562j  -44.02764984 +42.79225072j
   // 89.15365045 -14.57386059j  118.94062029 -10.8452387j
  // 108.29071849 -41.16699172j   67.17722805 +66.17455353j
  // 129.83575362 -19.3857114j    25.53325289 +72.75019775j
   // 48.         +12.j          141.75599176  +5.25427001j
   // 59.49219962 +24.95713725j   64.05621207 +49.31729165j
    // 3.42789564 -56.9873253j   127.79222818 -14.7950111j
  // 100.14170239 +13.37873228j   39.96257108 +40.42186716j
   // -2.54415588 -89.74011537j   76.22705162 -26.95460967j
   // 57.68394378+132.70446549j  146.53116761 +75.81095868j
  // 343.32575878+194.18044368j  656.97448503+592.2103624j
 // -443.87835664-364.05199061j  -72.23205842-241.45256226j]

constexpr float max_diff = 1e-3;

int main() {
   try {
      bool debug = true;
      int num_samples = 64;

      SimpleDSP simple_dsp( num_samples, debug );

      simple_dsp.run();

      return EXIT_SUCCESS;

   } catch (std::exception& ex) {
      std::cout << "ERROR: " << ex.what() << "\n";
      return EXIT_FAILURE;
   }
}
