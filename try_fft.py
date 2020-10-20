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
    all_con_sqrs = []
    all_psds = []
    while start_index < num_samples:
        frequencies = np.fft.fft( samples[start_index: start_index + fft_size], fft_size )
        frequencies = np.fft.fftshift( frequencies )
        con_sqrs = np.square(np.real(frequencies)) + np.square(np.imag(frequencies))
        psds = 10*np.log10(con_sqrs) - 10*np.log10(num_samples)
        if debug: 
            print( "start_index: {}: samples is {}".format( start_index,
                samples[ start_index: start_index + fft_size] ) )
            print( "start_index: {}: frequencies is {}".format( start_index, frequencies ) )
            print( "start_index: {}: conjugate squares is {}".format( start_index, con_sqrs ) )
            print( "start_index: {}: psds is {}".format( start_index, psds ) )
        all_frequencies.append( frequencies )
        all_con_sqrs.append( con_sqrs )
        all_psds.append( psds )
        start_index += fft_size
    
    all_frequencies = np.array( all_frequencies )
    all_frequencies = all_frequencies.flatten()
    all_con_sqrs = np.array( all_con_sqrs )
    all_con_sqrs = all_con_sqrs.flatten()
    all_psds = np.array( all_psds )
    all_psds = all_psds.flatten()
    #all_frequencies = all_frequencies.view( np.float32 )

    print( "#pragma once\n" )
    # print( "\n#include <cufft.h>\n" )
    # print( "constexpr cufftComplex expected_frequencies[] = {" )
    # index = 0
    # for frequency in all_frequencies:
    #     print( "\t{}{} ".format( frequency, (",","")[index >= num_samples-1] ) )
    #     index += 1
    # print( "};\n\n" )
    # print( "constexpr float expected_con_sqrs[] = {" )
    # index = 0
    # for con_sqr in all_con_sqrs:
    #     print( "\t{}{} ".format( con_sqr, (",","")[index >= num_samples-1] ) )
    #     index += 1
    # print( "};\n" )
    print( "constexpr float expected_psds[] = {" )
    index = 0
    for psd in all_psds:
        print( "\t{}{} ".format( psd, (",","")[index >= num_samples-1] ) )
        index += 1
    print( "};\n" )
    

if __name__ == '__main__':
    filename = 'testdataBPSKcomplex.bin'
    num_samples = 256
    fft_size = 64
    debug = False
    try_fft( filename, num_samples, fft_size, debug ) 
