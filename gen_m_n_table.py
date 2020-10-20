#!/usr/bin/env python3

import math

# Script for gen_m_n_table

# Numpy and math don't have a sincos() function
def my_sincos(val):
   return math.cos(val), math.sin(val)

def gen_m_n_table():
   index = 0
   print("#pragma once\n")
   print("#include <cufft.h>\n")

   print("__device__ __constant__ cufftComplex w_values[] = {")
   for m in range(64):
      for n in [1 << j for j in range(2,7)]:
         result = my_sincos(-2*math.pi*(m/n))
         print( "\t{{{},{}}}{}".format( result[0], result[1],
            (",", "")[index>=319]) )
         index += 1

   print("};")

if __name__ == '__main__':
   gen_m_n_table() 
