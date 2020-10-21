#pragma once

#include <cufft.h>

// Table only valid for case of 64 sample FFT
//#include "sincos_table.cuh"
// 'extern' here is to fix nvlink linker error
extern __device__ __constant__ cufftComplex w_values[];

#include "FFT_params.cuh"

// From SMFFT, Shared Memory-Based CUDA FFT library, by Karel Adamek (THANKS!!!)
// Found at: https://github.com/KAdamek/SMFFT

// https://www.geeksforgeeks.org/number-of-leading-zeros-in-binary-representation-of-a-given-number/
__device__ __inline__ int num_leading_zeros(unsigned int x) {
    unsigned y;
    int n = 32;
    y = x >>16; if (y != 0) {n = n -16; x = y;}
    y = x >> 8; if (y != 0) {n = n - 8; x = y;}
    y = x >> 4; if (y != 0) {n = n - 4; x = y;}
    y = x >> 2; if (y != 0) {n = n - 2; x = y;}
    y = x >> 1; if (y != 0) return n - 2;
    return n - x;
}

__device__ __inline__ int my_log2n(int x) {
   return 31 - num_leading_zeros((unsigned int)x);
}

// Valid only for case of 64 sample FFT
__device__ __inline__ int calc_w_index(int N, int m) {
   return ((m * 5) + (my_log2n(N) - 2));
}

// Replace sincosf with table lookup
__device__ __inline__ float2 Get_W_value_lookup(int N, int m){
   return w_values[calc_w_index(N,m)];
}

__device__ __inline__ float2 Get_W_value(int N, int m){
   float2 ctemp;
   sincosf ( -6.283185308f*fdividef( (float) m, (float) N), &ctemp.y, &ctemp.x);
   return(ctemp);
}

__device__ __inline__ float shfl(float *value, int par){
   #if (CUDART_VERSION >= 9000)
      return(__shfl_sync(0xffffffff, (*value), par));
   #else
      return(__shfl((*value), par));
   #endif
}

__device__ __inline__ float shfl_xor(float *value, int par){      
   #if (CUDART_VERSION >= 9000)
      return(__shfl_xor_sync(0xffffffff, (*value), par));
   #else
      return(__shfl_xor((*value), par));
   #endif
}

__device__ __inline__ float shfl_down(float *value, int par){
   #if (CUDART_VERSION >= 9000)
      return(__shfl_down_sync(0xffffffff, (*value), par));
   #else
      return(__shfl_down((*value), par));
   #endif
}

__device__ __inline__ void reorder_32_register(float2 *A_DFT_value, float2 *B_DFT_value, float2 *C_DFT_value, float2 *D_DFT_value){
   float2 Af2temp, Bf2temp, Cf2temp, Df2temp;
   unsigned int target = ((unsigned int) __brev( threadIdx.x ))>>(27);
   Af2temp.x = shfl(&(A_DFT_value->x), target);
   Af2temp.y = shfl(&(A_DFT_value->y), target);
   Bf2temp.x = shfl(&(B_DFT_value->x), target);
   Bf2temp.y = shfl(&(B_DFT_value->y), target);
   Cf2temp.x = shfl(&(C_DFT_value->x), target);
   Cf2temp.y = shfl(&(C_DFT_value->y), target);
   Df2temp.x = shfl(&(D_DFT_value->x), target);
   Df2temp.y = shfl(&(D_DFT_value->y), target);
   __syncwarp();
   (*A_DFT_value) = Af2temp;
   (*B_DFT_value) = Bf2temp;
   (*C_DFT_value) = Cf2temp;
   (*D_DFT_value) = Df2temp;
}

template<class const_params>
__device__ __inline__ void reorder_32(float2 *s_input, float2 *A_DFT_value, float2 *B_DFT_value, float2 *C_DFT_value, float2 *D_DFT_value){
   reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
}


template<class const_params>
__device__ __inline__ void reorder_64(float2 *s_input, float2 *A_DFT_value, float2 *B_DFT_value, float2 *C_DFT_value, float2 *D_DFT_value){
   int local_id = threadIdx.x & (const_params::warp - 1);
   int warp_id = threadIdx.x/const_params::warp;
   
   // reorder elements within warp so we can save them in semi-transposed manner into shared memory
   reorder_32_register(A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value);
   
   // reorder elements within warp so we can save them in semi-transposed manner into shared memory
   __syncthreads();
   unsigned int sm_store_pos = (local_id>>4) + 2*(local_id&15) + warp_id*132;
   s_input[sm_store_pos]          = *A_DFT_value;
   s_input[sm_store_pos + 33]     = *B_DFT_value;
   s_input[66 + sm_store_pos]     = *C_DFT_value;
   s_input[66 + sm_store_pos +33] = *D_DFT_value;
   
   // Read shared memory to get reordered input
   unsigned int sm_read_pos = (local_id&1)*32 + local_id + warp_id*132;
   __syncthreads();
   *A_DFT_value = s_input[sm_read_pos];
   *B_DFT_value = s_input[sm_read_pos + 1];
   *C_DFT_value = s_input[sm_read_pos + 66];
   *D_DFT_value = s_input[sm_read_pos + 66 + 1];
}


// From https://github.com/KAdamek/SMFFT/blob/master/SMFFT_CooleyTukey_C2C/FFT-GPU-32bit.cu
// with extra code for swapping the halves of the output ('Shift FFT')
template<class const_params>
__device__ void sm_fft(cufftComplex *sh_samples) {

   cufftComplex A_DFT_value, B_DFT_value, C_DFT_value, D_DFT_value;
   cufftComplex A_shift_value, B_shift_value, C_shift_value, D_shift_value;
   cufftComplex W;
   cufftComplex Aftemp, Bftemp, Cftemp, Dftemp;

   int j, m_param;
   int parity, itemp;
   int A_read_index, B_read_index, C_read_index, D_read_index;
   int A_shift_index, B_shift_index, C_shift_index, D_shift_index;
   int PoT, PoTp1, q;

   int local_id = threadIdx.x & (const_params::warp - 1);
   int warp_id = threadIdx.x/const_params::warp;
   A_DFT_value = sh_samples[local_id + (warp_id<<2)*const_params::warp];
   B_DFT_value = sh_samples[local_id + (warp_id<<2)*const_params::warp + const_params::warp];
   C_DFT_value = sh_samples[local_id + (warp_id<<2)*const_params::warp + 2*const_params::warp];
   D_DFT_value = sh_samples[local_id + (warp_id<<2)*const_params::warp + 3*const_params::warp];
   
   reorder_64<const_params>(sh_samples, &A_DFT_value, &B_DFT_value, &C_DFT_value, &D_DFT_value);
   
   //----> FFT
   PoT=1;
   PoTp1=2;   

   //--> First iteration
   itemp=local_id&1;
   parity=(1-itemp*2);
   
   A_DFT_value.x = parity*A_DFT_value.x + shfl_xor(&A_DFT_value.x, 1);
   A_DFT_value.y = parity*A_DFT_value.y + shfl_xor(&A_DFT_value.y, 1);
   B_DFT_value.x = parity*B_DFT_value.x + shfl_xor(&B_DFT_value.x, 1);
   B_DFT_value.y = parity*B_DFT_value.y + shfl_xor(&B_DFT_value.y, 1);
   C_DFT_value.x = parity*C_DFT_value.x + shfl_xor(&C_DFT_value.x, 1);
   C_DFT_value.y = parity*C_DFT_value.y + shfl_xor(&C_DFT_value.y, 1);
   D_DFT_value.x = parity*D_DFT_value.x + shfl_xor(&D_DFT_value.x, 1);
   D_DFT_value.y = parity*D_DFT_value.y + shfl_xor(&D_DFT_value.y, 1);
   
   //--> Second through Fifth iteration (no synchronization)
   PoT=2;
   PoTp1=4;
   for(q=1;q<5;q++){
      m_param = (local_id & (PoTp1 - 1));
      itemp   = m_param>>q;
      parity  = ((itemp<<1)-1);
      
      W = Get_W_value_lookup(PoTp1, itemp*m_param);
   
      Aftemp.x = W.x*A_DFT_value.x - W.y*A_DFT_value.y;
      Aftemp.y = W.x*A_DFT_value.y + W.y*A_DFT_value.x;
      Bftemp.x = W.x*B_DFT_value.x - W.y*B_DFT_value.y;
      Bftemp.y = W.x*B_DFT_value.y + W.y*B_DFT_value.x;
      Cftemp.x = W.x*C_DFT_value.x - W.y*C_DFT_value.y;
      Cftemp.y = W.x*C_DFT_value.y + W.y*C_DFT_value.x;
      Dftemp.x = W.x*D_DFT_value.x - W.y*D_DFT_value.y;
      Dftemp.y = W.x*D_DFT_value.y + W.y*D_DFT_value.x;
      
      A_DFT_value.x = Aftemp.x + parity*shfl_xor(&Aftemp.x,PoT);
      A_DFT_value.y = Aftemp.y + parity*shfl_xor(&Aftemp.y,PoT);
      B_DFT_value.x = Bftemp.x + parity*shfl_xor(&Bftemp.x,PoT);
      B_DFT_value.y = Bftemp.y + parity*shfl_xor(&Bftemp.y,PoT);
      C_DFT_value.x = Cftemp.x + parity*shfl_xor(&Cftemp.x,PoT);
      C_DFT_value.y = Cftemp.y + parity*shfl_xor(&Cftemp.y,PoT);
      D_DFT_value.x = Dftemp.x + parity*shfl_xor(&Dftemp.x,PoT);
      D_DFT_value.y = Dftemp.y + parity*shfl_xor(&Dftemp.y,PoT);   
      
      PoT=PoT<<1;
      PoTp1=PoTp1<<1;
   }
   
   itemp = local_id + (warp_id<<2)*const_params::warp;
   sh_samples[itemp]                        = A_DFT_value;
   sh_samples[itemp + const_params::warp]   = B_DFT_value;
   sh_samples[itemp + 2*const_params::warp] = C_DFT_value;
   sh_samples[itemp + 3*const_params::warp] = D_DFT_value;
   
   __syncthreads();
   q = 5;
   m_param = threadIdx.x & (PoT - 1);
   j = threadIdx.x>>q;
   
   W = Get_W_value_lookup(PoTp1,m_param);
   
   A_read_index=j*(PoTp1<<1) + m_param;
   B_read_index=j*(PoTp1<<1) + m_param + PoT;
   C_read_index=j*(PoTp1<<1) + m_param + PoTp1;
   D_read_index=j*(PoTp1<<1) + m_param + 3*PoT;
   
   Aftemp = sh_samples[A_read_index];
   Bftemp = sh_samples[B_read_index];
   A_DFT_value.x=Aftemp.x + W.x*Bftemp.x - W.y*Bftemp.y;
   A_DFT_value.y=Aftemp.y + W.x*Bftemp.y + W.y*Bftemp.x;      
   B_DFT_value.x=Aftemp.x - W.x*Bftemp.x + W.y*Bftemp.y;
   B_DFT_value.y=Aftemp.y - W.x*Bftemp.y - W.y*Bftemp.x;
   
   Cftemp = sh_samples[C_read_index];
   Dftemp = sh_samples[D_read_index];
   C_DFT_value.x=Cftemp.x + W.x*Dftemp.x - W.y*Dftemp.y;
   C_DFT_value.y=Cftemp.y + W.x*Dftemp.y + W.y*Dftemp.x;      
   D_DFT_value.x=Cftemp.x - W.x*Dftemp.x + W.y*Dftemp.y;
   D_DFT_value.y=Cftemp.y - W.x*Dftemp.y - W.y*Dftemp.x;
   
   sh_samples[A_read_index]=A_DFT_value;
   sh_samples[B_read_index]=B_DFT_value;
   sh_samples[C_read_index]=C_DFT_value;
   sh_samples[D_read_index]=D_DFT_value;
   
   PoT=PoT<<1;
   PoTp1=PoTp1<<1;
   
   // Swap the halves of each 64 sample FFT
   A_read_index = itemp;
   B_read_index = itemp + const_params::warp;
   C_read_index = itemp + 2*const_params::warp;
   D_read_index = itemp + 3*const_params::warp;

   if (( (A_read_index >= 0 ) && ( A_read_index < const_params::warp )) ||
      ( A_read_index >= 2*const_params::warp ) && ( A_read_index < 3*const_params::warp )) {

      A_shift_index = A_read_index + const_params::warp;
   } else if ((( A_read_index >= const_params::warp ) && ( A_read_index < 2*const_params::warp )) ||
      ( A_read_index >= 3*const_params::warp ) && ( A_read_index < 4*const_params::warp )) {

      A_shift_index = A_read_index - const_params::warp;
   }
   if (( (B_read_index >= 0 ) && ( B_read_index < const_params::warp )) ||
      ( B_read_index >= 2*const_params::warp ) && ( B_read_index < 3*const_params::warp )) {

      B_shift_index = B_read_index + const_params::warp;
   } else if ((( B_read_index >= const_params::warp ) && ( B_read_index < 2*const_params::warp )) ||
      ( B_read_index >= 3*const_params::warp ) && ( B_read_index < 4*const_params::warp )) {

      B_shift_index = B_read_index - const_params::warp;
   }
   if (( (C_read_index >= 0 ) && ( C_read_index < const_params::warp )) ||
      ( C_read_index >= 2*const_params::warp ) && ( C_read_index < 3*const_params::warp )) {

      C_shift_index = C_read_index + const_params::warp;
   } else if ((( C_read_index >= const_params::warp ) && ( C_read_index < 2*const_params::warp )) ||
      ( C_read_index >= 3*const_params::warp ) && ( C_read_index < 4*const_params::warp )) {

      C_shift_index = C_read_index - const_params::warp;
   }
   if (( (D_read_index >= 0 ) && ( D_read_index < const_params::warp )) ||
      ( D_read_index >= 2*const_params::warp ) && ( D_read_index < 3*const_params::warp )) {

      D_shift_index = D_read_index + const_params::warp;
   } else if ((( D_read_index >= const_params::warp ) && ( D_read_index < 2*const_params::warp )) ||
      ( D_read_index >= 3*const_params::warp ) && ( D_read_index < 4*const_params::warp )) {

      D_shift_index = D_read_index - const_params::warp;
   }

   A_shift_value = sh_samples[A_shift_index]; 
   B_shift_value = sh_samples[B_shift_index]; 
   C_shift_value = sh_samples[C_shift_index]; 
   D_shift_value = sh_samples[D_shift_index]; 
   __syncthreads();

   sh_samples[A_read_index] = A_shift_value;
   sh_samples[B_read_index] = B_shift_value;
   sh_samples[C_read_index] = C_shift_value;
   sh_samples[D_read_index] = D_shift_value;

}

