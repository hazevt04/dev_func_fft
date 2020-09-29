# dev_func_fft
An experiment to see about calling an FFT device function from a CUDA kernel.
The overall goal is to have the FFT in a DSP pipeline where the FFT is is the middle.
cuFFT is only callable from the host. This will have an FFT that is callable from a CUDA kernel.
