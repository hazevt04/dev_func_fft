#!/usr/bin/env python3

import numpy as np
import pathlib

# Script for try_fft

def get_samples( filepath, num_samples, debug=False ):
    samples = np.fromfile( filepath, dtype = np.complex64, count = num_samples, sep=""  )
    if debug: print( "num_samples is {}".format( num_samples ) )
    if debug: print( "filepath is is {}".format( filepath ) )
    if debug: print( "Size of samples is {}".format( len(samples) ) )
    return samples

def write_expected_file( filename, exp_type, exp_name, exp_data, debug=False ):
    with open(filename,'w') as exp_file:
        exp_data_len = len(exp_data)
        exp_file.write( "#pragma once\n" )
        if exp_type == 'cufftComplex': exp_file.write( "\n#include <cufft.h>\n" )
        exp_file.write( "\nconstexpr {} {}[] = {{".format( exp_type, exp_name ) )
        index = 0
        for exp in exp_data:
            if exp_type == 'cufftComplex':
                comp_exp = complex(exp)
                exp_file.write( "\t{{{},{}}}{}\n".format( exp.real, exp.imag, (",","")[index >= exp_data_len-1] ) )
            else:
                if exp == float('-Inf'): exp = 0
                exp_file.write( "\t{}{}\n".format( exp, (",","")[index >= exp_data_len-1] ) )
            index += 1
        exp_file.write( "};\n\n" )



def try_fft( filename, num_samples, fft_size, debug=False ):
    filepath = str( pathlib.Path().absolute()) + "/" + filename
    samples = get_samples( filename, num_samples, debug )
    start_index = 0
   
    if debug: print( "fft_size = {}".format( fft_size ) )
    if debug: print( "num_samples = {}".format( num_samples ) )
    all_frequencies = []
    all_con_sqrs = []
    all_psds = []
    while start_index < num_samples:
        frequencies = np.fft.fft( samples[start_index: start_index + fft_size], fft_size )
        frequencies = np.fft.fftshift( frequencies )
        con_sqrs = np.square(np.real(frequencies)) + np.square(np.imag(frequencies))
        fft_sizes = np.full( (1, fft_size), fft_size)
        psds = 10*np.log10(con_sqrs) - 10*np.log10(fft_sizes)
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

    freq_filename = '/home/glenn/Programming/CUDA/dev_func_fft/include/expected_sfrequencies.hpp.in'
    con_sqrs_filename = '/home/glenn/Programming/CUDA/dev_func_fft/include/expected_con_sqrs.hpp.in'
    psds_filename = '/home/glenn/Programming/CUDA/dev_func_fft/include/expected_psds.hpp.in'

    write_expected_file( freq_filename, 'cufftComplex', 'expected_sfrequencies', all_frequencies, debug )
    write_expected_file( con_sqrs_filename, 'float', 'expected_con_sqrs', all_con_sqrs, debug )
    write_expected_file( psds_filename, 'float', 'expected_psds', all_psds, debug )    


if __name__ == '__main__':
    filename = 'testdataBPSKcomplex.bin'
    num_samples = 32768
    fft_size = 64
    debug = True
    try_fft( filename, num_samples, fft_size, debug ) 
