#pragma once

#include <cmath>
#include <complex>
#include <iostream>
#include <iterator>

#include "my_utils.hpp"

// FFT Implementation from C++ Cookbook:
// https://www.oreilly.com/library/view/c-cookbook/0596007612/ch11s18.html#cplusplusckbk-CHP-11-EX-33

unsigned int bit_reverse(unsigned int x, int log2n) {
   int n = 0;
   for (int i = 0; i < log2n; i++) {
      n <<= 1;
      n |= (x & 1);
      x >>= 1;
   }
   return n;
}

constexpr double PI = 3.1415926536;

template <typename T>
using complex_vec = std::vector<std::complex<T>>;

template <typename T>
void cookbook_fft(complex_vec<T>& a, complex_vec<T>& b, int log2n) {

   const std::complex<T> J(0, 1);
   unsigned int fft_size = 1 << log2n;
   
   for (unsigned int i = 0; i != fft_size; ++i) {
      unsigned int br_index = bit_reverse(i, log2n);
      b[br_index] = a[i];
   }

   for (int s = 1; s <= log2n; ++s) {
      unsigned int m = 1 << s;
      unsigned int m2 = m >> 1;
      std::complex<T> w(1, 0);
      std::complex<T> wm = exp(-J * (T(PI) / m2));
      for (unsigned int j = 0; j != m2; ++j) {
         for (int k = j; k < (int)fft_size; k += m) {
            std::complex<T> t = w * b[k + m2];
            std::complex<T> u = b[k];
            b[k] = u + t;
            b[k + m2] = u - t;
         }
         w *= wm;
      }
   } // end of for (int s = 1; s <= log2n; ++s) {

}


