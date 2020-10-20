#!/usr/bin/env python3

import numpy as np
import pathlib

# Script for try_fft

def get_samples( filepath, num_samples, debug=False ):
    samples = np.fromfile( filepath, dtype = np.complex64, count = num_samples, sep=""  )
    #  if debug: print( "num_samples is {}".format( num_samples ) )
    #  if debug: print( "filepath is is {}".format( filepath ) )
    #  if debug: print( "Size of samples is {}".format( len(samples) ) )
    return samples

def try_fft( filename, num_samples, fft_size, debug=False ):
    filepath = str( pathlib.Path().absolute()) + "/" + filename
    samples = get_samples( filename, num_samples, debug )
    start_index = 0
   
    if debug: print( "fft_size = {}".format( fftshift ) )
    all_frequencies = []
    while start_index < num_samples:
        frequencies = np.fft.fft( samples[start_index: start_index + fft_size], fft_size )
        frequencies = np.fft.fftshift( frequencies )
        if debug: print( "start_index: {}: samples is {}".format( start_index,
            samples[ start_index: start_index + fft_size] ) )
        if debug: print( "start_index: {}: frequencies is {}".format( start_index, frequencies ) )
        all_frequencies.append( frequencies )
        start_index += fft_size
    
    all_frequencies = np.array( all_frequencies )
    all_frequencies = all_frequencies.flatten()
    #all_frequencies = all_frequencies.view( np.float32 )

    print( "#pragma once\n" )
    print( "\n#include <cufft.h>\n" )
    print( "constexpr cufftComplex expected_frequencies[] = {" )
    for frequency in all_frequencies:
        print( "{}, ".format( frequency ) )
    print( "}\n" )
    

if __name__ == '__main__':
    filename = 'testdataBPSKcomplex.bin'
    num_samples = 32768
    fft_size = 64
    debug = False
    try_fft( filename, num_samples, fft_size, debug ) 
