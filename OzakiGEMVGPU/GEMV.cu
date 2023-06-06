#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>
#include <random>
#include <iostream>
#include <random>
#include <iomanip>
#include <cuda.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda.h>
#include <inttypes.h>
#include <sys/time.h>
#include <float.h>
#include <cuda_fp16.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <bits/stdc++.h>

#include <cuda_fp16.h>
#include<algorithm>
#include<vector>

#include <random>
#include <iostream>
#include <random>
#include <iomanip>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>

#include <cublas_v2.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <cassert>
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>

#define MAX_CUDA_THREADS_PER_BLOCK 1024

#include <iostream>

using namespace std;


#include <cuda.h>
#define BLOCK 32
/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include <cuComplex.h>
#include <cublas_api.h>
#include <cuda_runtime_api.h>
#include <library_types.h>


#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

// memory alignment
#define ALIGN_TO(A, B) (((A + B - 1) / B) * B)

// device memory pitch alignment
static const size_t device_alignment = 32;

// type traits
template <typename T> struct traits;

template <> struct traits<float> {
    // scalar type
    typedef float T;
    typedef T S;

    static constexpr T zero = 0.f;
    static constexpr cudaDataType cuda_data_type = CUDA_R_32F;

    inline static S abs(T val) { return fabs(val); }

    template <typename RNG> inline static T rand(RNG &gen) { return (S)gen(); }

    inline static T add(T a, T b) { return a + b; }

    inline static T mul(T v, double f) { return v * f; }
};

template <> struct traits<double> {
    // scalar type
    typedef double T;
    typedef T S;

    static constexpr T zero = 0.;
    static constexpr cudaDataType cuda_data_type = CUDA_R_64F;

    inline static S abs(T val) { return fabs(val); }

    template <typename RNG> inline static T rand(RNG &gen) { return (S)gen(); }

    inline static T add(T a, T b) { return a + b; }

    inline static T mul(T v, double f) { return v * f; }
};

template <> struct traits<cuFloatComplex> {
    // scalar type
    typedef float S;
    typedef cuFloatComplex T;

    static constexpr T zero = {0.f, 0.f};
    static constexpr cudaDataType cuda_data_type = CUDA_C_32F;

    inline static S abs(T val) { return cuCabsf(val); }

    template <typename RNG> inline static T rand(RNG &gen) {
        return make_cuFloatComplex((S)gen(), (S)gen());
    }

    inline static T add(T a, T b) { return cuCaddf(a, b); }
    inline static T add(T a, S b) { return cuCaddf(a, make_cuFloatComplex(b, 0.f)); }

    inline static T mul(T v, double f) { return make_cuFloatComplex(v.x * f, v.y * f); }
};

template <> struct traits<cuDoubleComplex> {
    // scalar type
    typedef double S;
    typedef cuDoubleComplex T;

    static constexpr T zero = {0., 0.};
    static constexpr cudaDataType cuda_data_type = CUDA_C_64F;

    inline static S abs(T val) { return cuCabs(val); }

    template <typename RNG> inline static T rand(RNG &gen) {
        return make_cuDoubleComplex((S)gen(), (S)gen());
    }

    inline static T add(T a, T b) { return cuCadd(a, b); }
    inline static T add(T a, S b) { return cuCadd(a, make_cuDoubleComplex(b, 0.)); }

    inline static T mul(T v, double f) { return make_cuDoubleComplex(v.x * f, v.y * f); }
};

template <typename T> void print_matrix(const int &m, const int &n, const T *A, const int &lda);

template <> void print_matrix(const int &m, const int &n, const float *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <> void print_matrix(const int &m, const int &n, const double *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <> void print_matrix(const int &m, const int &n, const cuComplex *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
        }
        std::printf("\n");
    }
}

template <>
void print_matrix(const int &m, const int &n, const cuDoubleComplex *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
        }
        std::printf("\n");
    }
}

template <typename T> void print_vector(const int &m, const T *A);

template <> void print_vector(const int &m, const float *A) {
    for (int i = 0; i < m; i++) {
        std::printf("%0.2f ", A[i]);
    }
    std::printf("\n");
}

template <> void print_vector(const int &m, const double *A) {
    for (int i = 0; i < m; i++) {
        std::printf("%0.2f ", A[i]);
    }
    std::printf("\n");
}

template <> void print_vector(const int &m, const cuComplex *A) {
    for (int i = 0; i < m; i++) {
        std::printf("%0.2f + %0.2fj ", A[i].x, A[i].y);
    }
    std::printf("\n");
}

template <> void print_vector(const int &m, const cuDoubleComplex *A) {
    for (int i = 0; i < m; i++) {
        std::printf("%0.2f + %0.2fj ", A[i].x, A[i].y);
    }
    std::printf("\n");
}

template <typename T> void generate_random_matrix(int m, int n, T **A, int *lda) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<typename traits<T>::S> dis(-1.0, 1.0);
    auto rand_gen = std::bind(dis, gen);

    *lda = n;

    size_t matrix_mem_size = static_cast<size_t>(*lda * m * sizeof(T));
    // suppress gcc 7 size warning
    if (matrix_mem_size <= PTRDIFF_MAX)
        *A = (T *)malloc(matrix_mem_size);
    else
        throw std::runtime_error("Memory allocation size is too large");

    if (*A == NULL)
        throw std::runtime_error("Unable to allocate host matrix");

    // random matrix and accumulate row sums
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            T *A_row = (*A) + *lda * i;
            A_row[j] = traits<T>::rand(rand_gen);
        }
    }
}

// Makes matrix A of size mxn and leading dimension lda diagonal dominant
template <typename T> void make_diag_dominant_matrix(int m, int n, T *A, int lda) {
    for (int i = 0; i < std::min(m, n); ++i) {
        T *A_row = A + lda * i;
        auto row_sum = traits<typename traits<T>::S>::zero;
        for (int j = 0; j < n; ++j) {
            row_sum += traits<T>::abs(A_row[j]);
        }
        A_row[i] = traits<T>::add(A_row[i], row_sum);
    }
}

// Returns cudaDataType value as defined in library_types.h for the string
// containing type name
cudaDataType get_cuda_library_type(std::string type_string) {
    if (type_string.compare("CUDA_R_16F") == 0)
        return CUDA_R_16F;
    else if (type_string.compare("CUDA_C_16F") == 0)
        return CUDA_C_16F;
    else if (type_string.compare("CUDA_R_32F") == 0)
        return CUDA_R_32F;
    else if (type_string.compare("CUDA_C_32F") == 0)
        return CUDA_C_32F;
    else if (type_string.compare("CUDA_R_64F") == 0)
        return CUDA_R_64F;
    else if (type_string.compare("CUDA_C_64F") == 0)
        return CUDA_C_64F;
    else if (type_string.compare("CUDA_R_8I") == 0)
        return CUDA_R_8I;
    else if (type_string.compare("CUDA_C_8I") == 0)
        return CUDA_C_8I;
    else if (type_string.compare("CUDA_R_8U") == 0)
        return CUDA_R_8U;
    else if (type_string.compare("CUDA_C_8U") == 0)
        return CUDA_C_8U;
    else if (type_string.compare("CUDA_R_32I") == 0)
        return CUDA_R_32I;
    else if (type_string.compare("CUDA_C_32I") == 0)
        return CUDA_C_32I;
    else if (type_string.compare("CUDA_R_32U") == 0)
        return CUDA_R_32U;
    else if (type_string.compare("CUDA_C_32U") == 0)
        return CUDA_C_32U;
    else
        throw std::runtime_error("Unknown CUDA datatype");
}








__device__ void device_strcpy(char *dst, const char *src) {
    while (*dst++ = *src++);
}



__global__ void kernel(char *A) {
    device_strcpy(A, "Hello, World!");
}



typedef struct
{
  int NumRow;
  int NumNonZeros;
  int *iCol;
  int *iRow;
  double *dVal;
}Coo;

typedef struct
{
  long int NumRow;
  long int NumNonZeros;
  int *iColCsr;
  int *iRowPtrCsr;
  double *dValCsr;
}Csr;

typedef struct
{
  long int NumRow;
  long int NumNonZeros;
  long int iMaxColEll;
  int *iColEll;
  double *dValEll;
  
}Ell;






typedef struct
{
  int NumRow;
  int NumNonZeros;
  int NumEle;
  double *dVal;
  int *iNumPerRow;
  
}RunLen;





#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define NOT 16
//__constant__ int *Con_iTable;

#define CUDA_SAFE_CALL(func) \
do { \
     cudaError_t err = (func); \
if (err != cudaSuccess) { \
fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
exit(err); \
     } \
} while(0)



#ifndef bool         /* Boolean が定義されていなかったら */

#define bool int
#endif

#ifndef true            /* TRUE が定義されていなかったら */
#define true 1
#endif

#ifndef false           /* FALSE が定義されていなかったら */
#define false 0
#endif









int GetRandom(int min, int max)
{
    return min + (int)(rand() * (max - min + 1.0) / (1.0 + RAND_MAX));
}




Coo MakeSparseMatrix(int MAXRow, int MAXColumm,  float Percent , Coo sCoo){

  int num_of_Non_zero = MAXRow*MAXColumm*(Percent/100);

        sCoo.NumNonZeros = num_of_Non_zero;

        sCoo.dVal = (double *)malloc(sizeof(double)*sCoo.NumNonZeros);
        sCoo.iCol = (int *)malloc(sizeof(int)*sCoo.NumNonZeros);
        sCoo.iRow = (int *)malloc(sizeof(int)*sCoo.NumNonZeros);


        for(int i = 0; i < sCoo.NumNonZeros; i++)
          {
            sCoo.dVal[i]=0.0;
            sCoo.iRow[i]=0;
            sCoo.iCol[i]=0;
          }        


          
  using std::cout;
  using std::endl;
  using std::setprecision;

  // Modify as needed
    std::setprecision(100);
  constexpr int MIN = 0;
  constexpr int MAX = 100000000;
  std::random_device rd;

  std::default_random_engine eng(rd());
  std::uniform_real_distribution<double> distr(MIN, MAX);



            double tempVal = distr(eng);
            int temperROW = GetRandom(0,sCoo.NumNonZeros);
            int temperCOL = GetRandom(0,sCoo.NumNonZeros);

        for(int i = 0; i < sCoo.NumNonZeros; i++)
          {

          tempVal = distr(eng);
          //printf("Val=%lf\n",tempVal);
          temperROW = GetRandom(0,sCoo.NumNonZeros);
          //printf("ROW=%d\n",temperROW);
          temperCOL = GetRandom(0,sCoo.NumNonZeros);
          //printf("COL=%d\n",temperCOL);

            sCoo.dVal[i]=tempVal*tempVal/10000000000000;
            sCoo.iRow[i]=temperROW;
            sCoo.iCol[i]=temperCOL;
          }

          return sCoo;

}


Coo ReadMatrix( char *cFilename, Coo sCoo)
{
  FILE *fp;
  int iTmpRow, iTmpCol;
  int ret;
  double dTmpVal;

  int count = 0;

  fp = fopen( cFilename, "r" );

  if( fp == NULL ){
    printf( "%sファイルが開けません¥n", cFilename );
  }

  while( ( ret = fscanf( fp, "%d %d %lf", &iTmpCol, &iTmpRow,&dTmpVal) ) != EOF )
    {
      if(count == 0){
        sCoo.NumRow = iTmpCol;
        sCoo.NumNonZeros = (int)dTmpVal;
        sCoo.dVal = (double *)malloc(sizeof(double)*sCoo.NumNonZeros);
        sCoo.iCol = (int *)malloc(sizeof(int)*sCoo.NumNonZeros);
        sCoo.iRow = (int *)malloc(sizeof(int)*sCoo.NumNonZeros);

        for(int i = 0; i < sCoo.NumNonZeros; i++)
          {
            sCoo.dVal[i]=0.0;
            sCoo.iRow[i]=0;
            sCoo.iCol[i]=0;
          }
     
        count++;
      }else{
        // if(double(dTmpVal)==0.0)
        //   continue;
        sCoo.dVal[count-1]=(double)dTmpVal;
        sCoo.iRow[count-1]=iTmpRow-1;
        sCoo.iCol[count-1]=iTmpCol-1;
        count++;
      }
    }
  fclose(fp);
  return sCoo;
  
}



void FreeCsr( Csr sCsr )
{
  free( sCsr.iColCsr );
  free( sCsr.iRowPtrCsr );
  free( sCsr.dValCsr );
  
}



RunLen Coo2RunLen(Coo sCoo, RunLen sRunLen)
{

  sRunLen.NumRow = sCoo.NumRow;
  sRunLen.NumNonZeros = sCoo.NumNonZeros;

  sRunLen.dVal = (double *) malloc (sizeof(double) * sRunLen.NumNonZeros * 3);
  sRunLen.iNumPerRow = (int *) malloc (sizeof(int) * (sRunLen.NumRow+1));
  
  int iCnt = 0;
  int CurrentRow = -1;
  int iCntPtr = 0;
  int iCntVal = 0;

  printf("Nz:%d\n", sRunLen.NumNonZeros);
  
  for (int i = 0; i < sRunLen.NumNonZeros; i++){
    // if( i % 100000 == 0)
    //   printf("%d\n", i);
    iCnt++;
        
    if (CurrentRow != sCoo.iRow[i]){
      CurrentRow = sCoo.iRow[i];
      sRunLen.iNumPerRow[iCntPtr] = iCntVal;   
      iCntPtr++;
      iCnt = 0;
    }

    if( sCoo.iCol[i] != iCnt ){
      sRunLen.dVal[iCntVal++] = 0.0;
      sRunLen.dVal[iCntVal++] = double(sCoo.iCol[i] - iCnt);
      iCnt = sCoo.iCol[i];     
    }

    sRunLen.dVal[iCntVal++] = sCoo.dVal[i];

    
  }
  printf("iCntPtr:%d\n", iCntPtr);
  sRunLen.iNumPerRow[iCntPtr] = iCntVal;

  sRunLen.NumEle = iCntVal;
  
  return sRunLen;
}


Csr Coo2Csr( Coo sCoo, Csr sCsr )
{

  int TmpNonZeros = 0;
  int Count = 0;
  int Diff = -1;
  struct timeval s, e;
    
  sCsr.NumRow = sCoo.NumRow;
  sCsr.NumNonZeros = sCoo.NumNonZeros;
  sCsr.iRowPtrCsr = (int *) malloc (sizeof(int)
                                    * (sCsr.NumRow + 1));
  sCsr.iColCsr = (int *) malloc (sizeof(int) * sCsr.NumNonZeros);
  sCsr.dValCsr = (double *) malloc (sizeof(double)
                                    * sCsr.NumNonZeros);
  //printf("NumNonzeros:%d\n", sCsr.NumNonZeros);


  for(int i = 0; i <= sCsr.NumNonZeros; i++)
    {
      //if(i%1000000==0) printf("%d\n",i);
      if ( i != sCsr.NumNonZeros )
        {
          sCsr.iColCsr[i] = sCoo.iCol[i];
          sCsr.dValCsr[i] = sCoo.dVal[i];
        }

      if ( Diff != sCoo.iRow[i]){
        Diff = sCoo.iRow[i];
        sCsr.iRowPtrCsr[Count] = TmpNonZeros;   
        Count++;
      }

      
      if(Diff == sCoo.iRow[i])
        {
          TmpNonZeros++;
        }

      
    }

  printf("CSR Convert time = %lf (msec)\n", ((e.tv_sec - s.tv_sec) + (e.tv_usec - s.tv_usec)*1.0E-6)*1000);

  
  return sCsr;

}



double *MakeAns( int NumRow, double *dAns )
{

  dAns = ( double *) malloc ( sizeof(double) * NumRow );

  for( int i = 0; i < NumRow; i++)
    dAns[i] = 0.0;

  
  return dAns;
  
}

double *MakeVector( int NumRow, double *dVector )
{

  dVector = ( double *) malloc ( sizeof(double) * NumRow );

  srand((unsigned)time(NULL));

  for( int i = 0; i < NumRow; i++)
    //dVector[i] = 1; //i%10; //rand() % 10;
    dVector[i] = rand() % 10 / 100;
    //dVector[i] = 1000000000;
  return dVector;
  
}






__global__ void
device_CsrSpMVGPU( long int NumRow, int *iColCsr, int *iRowPtrCsr, 
double *dValCsr, double *dVector, double *dAns )
{

  long int idx = blockDim.x * blockIdx.x + threadIdx.x;
  long int idy = blockDim.y * blockIdx.y + threadIdx.y;
  long int offset = gridDim.x*blockDim.x;
  long int i = idx + idy*offset;

  if( i < NumRow )
    {
      double dTmpAns = 0.0;

      int iRowStart = iRowPtrCsr[i];
      int iRowEnd = iRowPtrCsr[i+1];

      for( int j = iRowStart; j < iRowEnd; j++){
        dTmpAns += dValCsr[j] * dVector[ iColCsr[j] ];
      }
      
      dAns[i] = dTmpAns;
      //printf(" dAns[%d] = %lf\n", i, dAns[i]);
    }
  
}



double *CsrSpMVGPU(Csr sCsr, double *dVector, double *dAns){
  
  int iBlocksPara;
  int *dev_iColCsr;
  int *dev_iRowPtrCsr;
  double *dev_dValCsr;
  double *dev_dVector;
  double *dev_dAns;

  long int MemUsageCsr = 0;
  cudaEvent_t start, stop;

  cudaDeviceReset();

  printf("NumRow:%ld, NumNonZeros:%ld\n", sCsr.NumRow, sCsr.NumNonZeros);

  if( sCsr.NumRow % (NOT*NOT) == 0 ){
    iBlocksPara = sCsr.NumRow / (NOT*NOT);
  } else 
    {
      iBlocksPara = sCsr.NumRow / (NOT*NOT) + 1;
  }

  dim3 iGrid(iBlocksPara,1,1);
  dim3 iBlock(NOT,NOT,1);

  CUDA_SAFE_CALL( cudaMalloc((void**)&dev_iColCsr, sizeof(int)*sCsr.NumNonZeros) ); 
   cudaDeviceSynchronize();
  CUDA_SAFE_CALL( cudaMalloc((void**)&dev_iRowPtrCsr, sizeof(int)*(sCsr.NumRow+1)) );
   cudaDeviceSynchronize();
  CUDA_SAFE_CALL( cudaMalloc((void**)&dev_dValCsr, sizeof(double)*sCsr.NumNonZeros) );
   cudaDeviceSynchronize();
  CUDA_SAFE_CALL( cudaMalloc((void**)&dev_dVector, sizeof(double)*sCsr.NumRow) );
   cudaDeviceSynchronize();
  CUDA_SAFE_CALL( cudaMalloc((void**)&dev_dAns, sizeof(double)*sCsr.NumRow) );
   cudaDeviceSynchronize();

  MemUsageCsr = 4 * sCsr.NumNonZeros + 8 * sCsr.NumNonZeros + 4 * (sCsr.NumRow + 1);
  printf("Memory usage of Csr: %ld (byte)\n", MemUsageCsr);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  CUDA_SAFE_CALL( cudaMemcpy(dev_iColCsr, sCsr.iColCsr, sizeof(int)*sCsr.NumNonZeros,
              cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL( cudaMemcpy(dev_iRowPtrCsr, sCsr.iRowPtrCsr, sizeof(int)*(sCsr.NumRow+1),
              cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL( cudaMemcpy(dev_dValCsr, sCsr.dValCsr, sizeof(double)*sCsr.NumNonZeros,
              cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL( cudaMemcpy(dev_dVector, dVector, sizeof(double)*sCsr.NumRow,
              cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL( cudaMemcpy(dev_dAns, dAns, sizeof(double)*sCsr.NumRow,
              cudaMemcpyHostToDevice) );              

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float Msec = 0.0;
  cudaEventElapsedTime(&Msec, start, stop);
  printf("CSR Transfer time: %lf msec\n", Msec);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  device_CsrSpMVGPU <<< iGrid, iBlock >>> (sCsr.NumRow, dev_iColCsr, dev_iRowPtrCsr, 
                                dev_dValCsr, dev_dVector, dev_dAns);
  cudaDeviceSynchronize();

  CUDA_SAFE_CALL( cudaMemcpy(dAns, dev_dAns, sizeof(double)*sCsr.NumRow,
              cudaMemcpyDeviceToHost) );
  cudaDeviceSynchronize();
                  
  return dAns;
}





void print_matrix(int m, int n , double *M ,int k){

int counter=0;
for(int i = 0 ; i < m*n ; i++){

 if(counter == n){
    printf("\n");
    counter=0;
 }
 printf("%lf ",M[i]);
 counter++;
}
}





void print_float_Vector( float C[] ,int n){

printf("\nPrint float Vector\n");

for (int j =0;j<n;j++){


printf("%f ",C[j]);



}
printf("\n\n\n");

}


void print_half_array( int m,int n,half C[] ){

printf("\nPrint half array \n");

int count =0;

if(n!=1){
for (int j =0;j<m*n;j++){


printf("%f ",C[j]);
count++;
if(count == n){

  printf("\n");
  count =0;
}

}
}
else{
for (int j =0;j<m;j++){


printf("%f ",C[j]);
count++;
if(count == m){

  printf("\n");
  count =0;
}

}


}
printf("\n\n\n");

}



/////////////////////////////////////////////////////////////////////////////////device ///////////////////////////////////////////////////
__global__ void hello(){
    printf("Hello CUDA World !!\n");
}






__global__ void Max_Sequential_Addressing_Shared(double* data, int data_size,
double *max){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ double  sdata[];

    if (idx < data_size){

        /*copy to shared memory*/
        sdata[threadIdx.x] = data[idx];
        __syncthreads();

        for(int stride=blockDim.x/2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
               double lhs = sdata[threadIdx.x];
               double rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] =  fabs(lhs) <  fabs(rhs) ? rhs : lhs;
            }
             __syncthreads();
        }


    }

 __syncthreads();
    if (idx == 0){
      sdata[0] = sdata[0];
    }
 __syncthreads();
  *max = sdata[0];
  __syncthreads();
  //printf("\n max in find max : %lf \n",*max);
}





///////////////////////////////////////////////////////////////////////////////////global////////////////////////////////////////////////////


__global__ void dev_Mat2SplitMat(double rho,int now,double *max,double *dev_Matrix,
double *SplitMatScaling,half *SplitMat,
int numCol, int numRow,int numSplitArray,double *d_only_TMPMat){
//printf("\n\n in Mat numSplit %d \n\n",numSplitArray);
 int j = blockDim.x * blockIdx.x + threadIdx.x;



//printf("\nnumSplit %d and max = %lf\n",numSplitArray,*max);
double tau,mu,sigma,tmp;

//printf("\n rho = %lf \n",rho);

// printf("Vec Max : %lf\n",*max);
mu = fabs(*max);
//printf("\n mu = %lf",mu);
tmp = log2(mu);
tau =ceil(tmp);
//printf("\n tau = %lf",tau);
SplitMatScaling[now]=tau;
//printf("\n devSplitMat ->Scaling = %lf\n ",SplitMatScaling[now]=tau);
tmp=rho+tau;
sigma = pow(2,(tmp));
//printf("\n sigma = %lf",sigma);

    if(j<numRow*numCol){

    d_only_TMPMat[j]=(double)(dev_Matrix[j]+sigma);
       __syncthreads();
    d_only_TMPMat[j]=(double)(dev_Matrix[j]-sigma);
       __syncthreads();
    //printf("\n d_only_TMPVec = %lf",d_only_TMPMat[j]);
    ///printf("\n dev_Vec = %lf",dev_Matrix[j]);
    dev_Matrix[j] = ((dev_Matrix[j]-d_only_TMPMat[j]));
       __syncthreads();
       tmp = pow(2,(-1*tau));
    d_only_TMPMat[j]=tmp*d_only_TMPMat[j];
     printf("\n  d_only_Mat>data[] = %f",(double)(d_only_TMPMat[j])); 
        SplitMat[j+(now*numRow*numCol)]= __double2half(d_only_TMPMat[j]); //dame
               __syncthreads();
    //  printf("\n                      devSplitMat = %f",(SplitMat[j+(now*numRow*numCol)]));

    
       }
       __syncthreads();
       // printf("\n dev_Vec = %lf",dev_Matrix[j]);
        //for(int k = 0;k<numRow*numRow;k++){
    // printf("\n  d_only_Mat>data[%d] = %f",k,(double)(d_only_TMPMat[k]));  
   // printf("\n               devMat[%d] = %f",k,(double)(dev_Matrix[k]));  
     //printf("\n                      devSplitMat->data[%d] = %f",k,(half)(SplitMat[now*j]));
    //}

    }
   

     


      //printf("\n in After kernel 11111111111111 %lf\n",devSplitMatrix[1].Scaling);











//////////////////////////////////////////////////////////////////Function4OzakiScheme/////////////////////////////////////////////////////////////////////////





////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void ozaki_Split_fp64tofp16_Mat(double *d_Matrix ,
int numCol,int numRow,int numSplitArray,double *TMP_MAT,double *SplitMatScaling,half *SplitMat){
  //printf("\n in the Vec \n");
dim3 Grid(numRow*numCol);
dim3  Block(numRow*numCol);

 if(2000 < numRow*numCol){
 if((numRow ) % BLOCK == 0){
 dim3  Block(BLOCK,BLOCK);
 dim3 Grid(int(numCol*numRow)/BLOCK,int(numCol*numRow)/BLOCK);
}
else{
  dim3  Block(BLOCK,BLOCK);
  dim3  Grid((int(numCol*numRow)/BLOCK)+1,(int(numCol*numRow)/BLOCK)+1);
}
 }

double *d_max;
cudaMalloc((void**)&d_max, sizeof(double));
double rho = ceil(log2(53)-(log2(23)-log2(numSplitArray))/2);

for(int i=0;i<numSplitArray;i++){

Max_Sequential_Addressing_Shared<<<1, numRow*numCol,
     numRow*numCol * sizeof(double)>>>(d_Matrix,numRow*numCol,d_max);


//printf("\n \n Max_Sequential_Addressing_Shared END \n");
cudaDeviceSynchronize();
 //cudaMemcpy(&max, d_max, sizeof(double), cudaMemcpyDeviceToHost);
 //printf("\n in Vec max :%lf\n",max);
dev_Mat2SplitMat<<<1, numRow*numCol,
     numRow*numCol* sizeof(double)>>>(rho,i,d_max,d_Matrix,SplitMatScaling,SplitMat,
     numCol,numRow, numSplitArray,TMP_MAT);
//printf("\n\n  dev_Vec2SplitVec EDN \n");
cudaDeviceSynchronize();

}


} 

__global__ void dev_Vec2SplitVec(double rho,int now ,double *max, double *dev_Vector,
double *SplitVecScaling,half *devSplitVector,
int numRow,int numSplitArray ,double *d_only_TMPVec ){//既存手法
int j = blockDim.x * blockIdx.x + threadIdx.x;

//printf("\nnumSplit %d and max = %lf\n",numSplitArray,*max);
double tau,mu,sigma,tmp;

//printf("\n rho = %lf \n",rho);

//printf("Vec Max : %lf\n",*max);
mu = fabs(*max);
//printf("\n mu = %lf",mu);
tmp = log2(mu);
tau =ceil(tmp);
//printf("\n tau = %lf",tau);
SplitVecScaling[now]=tau;
//printf("\n devSplitVector ->Scaling = %lf\n ",SplitVecScaling[now]);
tmp= rho+tau;
sigma = pow(2,tmp);
//printf("\n sigma = %lf",sigma);

    if(j<numRow){
    d_only_TMPVec[j]=(double)(dev_Vector[j]+sigma);
       __syncthreads();
    d_only_TMPVec[j]=(double)(dev_Vector[j]-sigma);
       __syncthreads();
    //printf("\n d_only_TMPVec = %lf",d_only_TMPVec[j]);
    //printf("\n dev_Vec = %lf",dev_Vector[j]);
    dev_Vector[j] = (double)((dev_Vector[j]-d_only_TMPVec[j]));
       __syncthreads();
       tmp =-1*tau;
    d_only_TMPVec[j] = pow(2,(tmp))*d_only_TMPVec[j];//double

    printf("\n d_only_TMPVec = %lf",d_only_TMPVec[j]);

    devSplitVector[j+(now*numRow)]=  __double2half(d_only_TMPVec[j]); //2回目で値が入らない nannde
    
       __syncthreads();
        //printf("\n dev_Vec = %lf",dev_Vector[j]);
        //for(int k = 0;k<numRow;k++){
    //printf("\n  d_only_TMPVec>data = %f",(double)(d_only_TMPVec[j]));  
    //printf("\n               devVector[%d] = %f",k,(double)(dev_Vector[k]));  
    
   // printf("\n                      devSplitVector->data = %f",(devSplitVector[j+(now*numRow)]));
    
    //}
    }
    __syncthreads();


}


void ozaki_Split_fp64tofp16_Vec(double *d_Vector,
int numRow,int numSplitArray,double *TMP_VEC , double *SplitVecScaling, half *SplitVec ){
  //printf("\n in the Vec \n");
dim3 Grid(numRow);
dim3  Block(numRow);

 if(4000 < numRow){
 if((numRow ) % BLOCK == 0){
 dim3  Block(BLOCK,BLOCK);
 dim3 Grid(int(numRow)/BLOCK,int(numRow)/BLOCK);
}
else{
  dim3  Block(BLOCK,BLOCK);
  dim3  Grid((int(numRow)/BLOCK)+1,(int(numRow)/BLOCK)+1);
}
 }

double *d_max;

cudaMalloc((void**)&d_max, sizeof(double));

double rho = ceil(log2(53)-(log2(23)-log2(numSplitArray))/2);
for(int i=0;i<numSplitArray;i++){

Max_Sequential_Addressing_Shared<<<1, numRow,
     numRow* sizeof(double)>>>(d_Vector,numRow,d_max);

//printf("\n \n Max_Sequential_Addressing_Shared END \n");
cudaDeviceSynchronize();
 //cudaMemcpy(&max, d_max, sizeof(double), cudaMemcpyDeviceToHost);
 //printf("\n in Vec max :%lf\n",max);
dev_Vec2SplitVec<<<1, numRow,
     numRow* sizeof(double)>>>(rho,i,d_max,d_Vector,SplitVecScaling,
     SplitVec,numRow , numSplitArray,TMP_VEC);
//printf("\n\n  dev_Vec2SplitVec EDN \n");
cudaDeviceSynchronize();

}

 
}



void ozaki_Split_fp64tofp16_VecPackMatrix( ){



}






__global__ void dev_SumVector32to64( int NowMat,int NowVec,int numCol,int numRow,
double *MatScaling, half  *d_SplitMat ,double *VecScaling,half  *d_SplitVec ,
double *d_D,float *TMP_C){
 int i = blockDim.x * blockIdx.x + threadIdx.x;

  double Scaling = pow(2,(MatScaling[NowMat]+ VecScaling[NowVec]));
printf("\nScaling = %lf",Scaling);
//TMP_[]
 if(i < numRow){
//TMP_C[i] += 0;
  __syncthreads();
  d_D[i] +=  (Scaling)*(TMP_C[i]);
    printf("\nTMP_C %lf",(TMP_C[i]));
  printf("\ns_D %lf",(Scaling)*(TMP_C[i]));
  //d_D[i] += (double)(TMP_C[i]);
 }
__syncthreads();
  //printf("\nin sum D %d in C %lf",i,d_D[i]);

}


void ozaki_fp16tofp64_sum(cublasHandle_t cublasH, cublasOperation_t transa, cublasOperation_t transb,
 int m, int  n, float *alpha, double *d_A, int lda, double *d_B, int  ldb, float *beta,
  double *d_D, int ldd,int numSplitArray,int numSplitArray2,float *TMP_C,double *SplitMatScaling,
 half *SplitMat,double *SplitVecScaling , half*SplitVec){


half MATH[m*n];
half VECH[n];

float N[n];


half *A =nullptr;
half *B=nullptr;




    for(int i=0;i<numSplitArray;i++){
        for(int j=0;j<numSplitArray2;j++){
         //  printf("\nProblem 1 at %d %d\n",i,j);

if(m<17 &&n<17){

          CUDA_CHECK(cudaMemcpy( MATH, SplitMat +( i*m*n) ,m*n*sizeof(half),cudaMemcpyDeviceToHost));//できる
          CUDA_CHECK(cudaMemcpy( VECH, SplitVec + (j*n) ,n*sizeof(half),cudaMemcpyDeviceToHost));///できる
            print_half_array(m,n,MATH);//できる
            print_half_array(n,1,VECH);//できる
}

   // printf("\nProblem 2 st %d %d\n",i,j);
 A=(SplitMat+(i*m*n));
 B=(SplitVec+(j*n));
CUBLAS_CHECK(cublasGemmEx(//16to32andtmpSegfo nannde
      cublasH, transa, transb, m, n, 1 , &alpha, 
       A, CUDA_R_16F, lda,
       B, CUDA_R_16F, ldb,
        &beta, TMP_C, CUDA_R_32F, ldd,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

                //  printf("\nProblem X st %d %d\n",i,j);



       CUDA_CHECK(cudaMemcpy( N, TMP_C ,n*sizeof(float),cudaMemcpyDeviceToHost));
      //print_float_Vector(N,n);
                 // printf("\nProblem 3 st %d %d\n",i,j);
    dev_SumVector32to64<<<1, n,
     n* sizeof(double)>>>(i,j,m,n,SplitMatScaling,SplitMat,SplitVecScaling,SplitVec,d_D,TMP_C );

      //CUDA_CHECK(cudaMemcpy( N1, TMP_C ,n*sizeof(float),cudaMemcpyDeviceToHost));
       //print_float_Vector(N,n);

       cudaDeviceSynchronize();
       
                  // printf("\nProblem 4 st %d %d\n",i,j);
        }

    }







}

void ozaki_fp16tofp64_sumPacked(){


}



///////////////////////////////////////////////////////////GEMV func//////////////////////////////////////////////////////////////////////////////////

void ozakiDgemv64to16(cublasHandle_t cublasH, cublasOperation_t transa, cublasOperation_t transb,
 int m, int  n, float *alpha, double *d_A, int lda, double *d_B, int  ldb,
  float *beta,
  double *d_D, int ldd,int numSplitArray,int numSplitArray2,
  double *TMP_MAT,double *SplitMatScaling,half *SplitMat,
  double *TMP_VEC,double *SplitVecScaling,half *SplitVec,
  float *TMP_C){

   


ozaki_Split_fp64tofp16_Mat(d_A, m ,n,numSplitArray,TMP_MAT,SplitMatScaling,SplitMat);
cudaDeviceSynchronize();
ozaki_Split_fp64tofp16_Vec(d_B,n,numSplitArray,TMP_VEC,SplitVecScaling,SplitVec);
cudaDeviceSynchronize();
ozaki_fp16tofp64_sum(cublasH,transa,transb,m,n,alpha,d_A,lda,d_B,ldb,beta,d_D,ldd,
numSplitArray,numSplitArray2,TMP_C,SplitMatScaling,SplitMat,SplitVecScaling,SplitVec);
cudaDeviceSynchronize();







}





///////////////////////////////////////////////////////////////////////GEMVNeo提案手法////////////////////////////////////////////////////////////////////////////////////////





void ozakiDgemvNeo64to16(cublasHandle_t cublasH, cublasOperation_t transa, cublasOperation_t transb,
 int m, int  n, half *alpha, double *d_A, int lda, double *d_B, int  ldb,
  half *beta,
  double *d_D, int ldd,int numSplitArray,int numSplitArray2,
  double *TMP_MAT,double *SplitMatScaling,half *SplitMat,
  double *TMP_VEC,double *SplitVecScaling,half *SplitVec,
  float *TMP_C){



//ozaki_Split_fp64tofp16_Mat(d_A,SplitMat , m , n,numSplitArray,TMP_MAT);

//ozaki_Split_fp64tofp16_Vec(d_B,SplitVec,n,numSplitArray,TMP_VEC);




//ozaki_fp16tofp64_sumPacked(cublasH,transa,transb,m,n,1,alpha,d_A,lda,d_B,ldb,beta,d_D,ldd,
//numSplitArray,numSplitArray2,*SplitMat,TMP_MAT,SplitVec,TMP_VEC);








}














///////////////////////////////////////////////////////////main func//////////////////////////////////////////////////////////////////////////////////







int main(int argc, char *argv[]){

printf("\nOzakiScheme DGEMV\n");
int Jousu=atoi(argv[1]);
int bunkatu = atoi(argv[2]);
int  ShowM;

if(Jousu <4){
 ShowM = 1;
}
else {
 ShowM = 0;
}






int nnnn=1;
for(int i=0 ;i<Jousu;i++){
nnnn=nnnn*2;
}


//nnnn=nnnn+nnnn/2;
  const int m = nnnn;
  const int n = m;
  const int k = 1;
const int lda = m;
const int ldb = n;
const int ldc = n;
const int ldd = n;


 
  /*

    const int k = m;
const int ldbb = 1;
  const int ldb = m;
 const int ldc = 1;
 const int ldd = m;
 int p=6;
   *   A = | 1.0 | 2.0 |
   *       | 3.0 | 4.0 |
   *
   *   B = | 5.0 | 6.0 |
   *       | 7.0 | 8.0 |
   */

  //std::vector<std::vector<double>>  A(m,std::vector<double>(n));
  //std::vector<std::vector<double>> B(1,std::vector<double>(n));
  using std::cout;
  using std::endl;
  using std::setprecision;

  // Modify as needed
  std::setprecision(10);
  constexpr int MIN = 0;
  constexpr int MAX = 100;
  std::random_device rd{};
  std::default_random_engine eng(rd());
  std::uniform_real_distribution<double> distr(MIN, MAX);
 //行列生成部分
//const   std::vector<double> A = {334.89};
//const   std::vector<double> B = {5.0};
// std::vector<double> A (m*n,0);
//std::vector<double> B (m,0);
  double *A;
  double *B;
        A=(double *)malloc(sizeof(double)*m*m);
        B=(double *)malloc(sizeof(double)*m);
     for (int i = 0; i < m*m; ++i)
  {
    double temp = distr(eng);
    A[i]=temp;
  }
     for (int i = 0; i < m; ++i)
  {
    double tempp = distr(eng);
    B[i]=tempp;
  }
//std::vector<double> C(k);
// std::vector<double> C(n,0);
//std::vector<double> D(n,0);
  double *C;
  double *D;
  C=(double *)malloc(sizeof(double)*m);
  D=(double *)malloc(sizeof(double)*m);
  const double alpha = 1.0;
  const double beta = 0.0;
  double *d_A = nullptr;
  double *d_B = nullptr;
  double *d_C = nullptr;
  double *d_D = nullptr;

  printf("Matrix size = %d * %d\n", m, m);
//intf(" asdasdsad ");

if(ShowM == 1){
printf("\nMAtrix:A\n");
  print_matrix(m, n, A, lda);
  printf("\n");



printf("\nVector:B\n");
   print_matrix(m, 1, B, ldb);
  printf("\n");
//真実の入力確認/*
}



  /* step 2: copy data to device */
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * m*m));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * n));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(double) * n));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_D), sizeof(double) * n));

  // printf("Cutlass\n");
  cudaEvent_t start, stop;

  float millisecondsTA = 0;
  float millisecondsTB = 0;
  // use GemmEx by Tensorcore (maybe lying)



  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);

  CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(double) * m*m, cudaMemcpyHostToDevice));


   cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&millisecondsTA, start, stop);


  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);

  CUDA_CHECK(cudaMemcpy(d_B, B, sizeof(double) * m, cudaMemcpyHostToDevice));

   cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&millisecondsTB, start, stop);



  CUDA_CHECK(cudaMemcpy(d_C, C, sizeof(double) * m, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy(d_D, D, sizeof(double) * m, cudaMemcpyHostToDevice));

  /* step 3: compute */



  // 初期化1
  cublasHandle_t cublasH = NULL;

  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;

  // 開始時間を記録
  // printf("Cutlass\n");
  float milliseconds1_1 = 0;
  float milliseconds1_2 = 0;
  // use GemmEx by Tensorcore (maybe lying)

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);

  CUBLAS_CHECK(cublasCreate(&cublasH));
  // CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  // CUBLAS_CHECK(cublasSetStream(cublasH, stream));
  cudaEventRecord(stop, NULL);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds1_1, start, stop);
  // cudaEventDestroy(start);
  // cudaEventDestroy(stop);

  cudaEventRecord(start, NULL);

  //CUBLAS_CHECK(cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH));
/*
  CUBLAS_CHECK(cublasGemmEx(
      cublasH, transa, transb, m, n, k, &alpha, d_A, traits<double>::cuda_double, lda, d_B,
      traits<double>::cuda_double, ldb, &beta, d_C, traits<double>::cuda_double, ldc,
      CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
*/
///////////////////////////////////////////////////////////////////////////////////////////////////ああああああああ///


CUBLAS_CHECK(cublasDgemv(cublasH, transa, m, n, &alpha, d_A, lda, d_B, 1, &beta, d_C, 1));
  cudaDeviceSynchronize();


  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds1_2, start, stop);
  // cudaEventDestroy(start);
  // cudaEventDestroy(stop);
  // 初期化2


float millisecondsTC = 0;
 cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);

  CUDA_CHECK(cudaMemcpy(C, d_C, sizeof(double) * n, cudaMemcpyDeviceToHost));

 cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&millisecondsTC, start, stop);



  CUBLAS_CHECK(cublasDestroy(cublasH));

//for Ozaki scheme
/*


VEctor Vec{
  .width=n,
  .max=0,
  .data=NULL,
};

Vec.data=(double*)malloc(n*sizeof(double));
memcpy(&Vec.data, &B ,sizeof(B));
Vec.max=findMaxinVec(Vec);


 MAtrix Mat{
 .data =NYLL,
 .height=m,
 .width=n,
 };

MatX.data=(double*)malloc(n*m*sizeof(double));
memcpy(&MatX.data, &A ,sizeof(A));


*/






CUDA_CHECK(cudaFree(d_C));




///////////////////////////////////////////////////////////////////OzakiAlreadyshuhou////////////////////////////////////////////////
  /// printf("OzakiScheme\n");
cublasHandle_t cublasHH;

  float milliseconds2_1 = 0;
  float milliseconds2_2 = 0;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);

 CUBLAS_CHECK(cublasCreate(&cublasHH));//初期化２SplitVector&SplitMat make

int bunkatu2 = bunkatu;


//test_host();


  double *TMP_MAT = nullptr;
  double *TMP_VEC = nullptr;

  float *TMP_C = nullptr;

  double *SplitVecScaling = nullptr;
  double *SplitMatScaling = nullptr;
  half *SplitMat = nullptr;
  half *SplitVec = nullptr;




  CUDA_CHECK(cudaMalloc((void **) (&TMP_VEC), sizeof(double) * n));
  CUDA_CHECK(cudaMalloc((void **)(&TMP_MAT), sizeof(double) *m*n));
  CUDA_CHECK(cudaMalloc((void **)(&TMP_C), sizeof(float) *n));

  CUDA_CHECK(cudaMalloc((void **) (&SplitMatScaling), sizeof(double) * bunkatu));
  CUDA_CHECK(cudaMalloc((void **) (&SplitVecScaling), sizeof(double) * bunkatu2));
  CUDA_CHECK(cudaMalloc((void **)(&SplitMat), sizeof(half) *m*n*bunkatu));
  CUDA_CHECK(cudaMalloc((void **)(&SplitVec), sizeof(half) *n*bunkatu));



/*] CUDA_CHECK(cudaMalloc((void**)(&d_SplitMat), bunkatu * sizeof(SplitMatrix)));
  CUDA_CHECK(cudaMalloc((void**)(&d_SplitVec), bunkatu * sizeof(SplitVector)));

        CUDA_CHECK(cudaMalloc((void**)(&dataMatPtr), m*n *bunkatu* sizeof(half)));
    CUDA_CHECK(cudaMalloc((void**)(&dataVecPtr), n *bunkatu2* sizeof(half)));

    CUDA_CHECK(cudaMemcpy((&d_SplitMat[i].data), (&dataMatPtr[i*m*n]), 
    sizeof(half *),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((&d_SplitVec[i].data), (&dataVecPtr[i*n]), 
    sizeof(half *), cudaMemcpyHostToDevice));
*/


  ////////////////////////////////////////////////////////////////






  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds2_1, start, stop);
  // cudaEventDestroy(start);
  // cudaEventDestroy(stop);


  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);


float alpha_H = 1;
float beta_H = 0;
ozakiDgemv64to16(cublasHH, transa, transb, m, n, &alpha_H , d_A, lda, d_B, ldb, &beta_H , d_D, ldd, 
bunkatu,bunkatu2,TMP_MAT,SplitMatScaling,SplitMat ,TMP_VEC,SplitVecScaling,SplitVec,TMP_C);



 //cuozblasDgemmDP(&cuozblasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_D, ldd);






  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds2_2, start, stop);




float millisecondsTD = 0;
 cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL);

cudaMemcpy(D, d_D, sizeof(double) * n, cudaMemcpyDeviceToHost);
 cudaDeviceSynchronize();

 cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&millisecondsTD, start, stop);




  CUBLAS_CHECK(cublasDestroy(cublasHH));





//////////////////////////////////////////////////////OzakiTeian/////////////////////////////////////////////////




 //////////////////////////////////////////////////////////////////////////////////////////////////////////
  /* step 4: copy data to host */
 

  //cuozblasDestroy(&cuozblasH);

  /*
  printf("Using TensorCore on GEMM_EX\n");
  printf("Handle\n");
  printf("%f miliseconds \n", milliseconds1_1);
  printf("Gemm\n");
  printf("%f miliseconds \n", milliseconds1_2);
  printf("FLOPS\n");
  printf("%f GFLOPS \n", ((double)a*a*a*1.0e-6f/(double)milliseconds1_2));
  printf("\n");

  printf("Using TensorCore on OzakiScheme\n");
  printf("Handle\n");
  printf("%f miliseconds \n", milliseconds2_1);
  printf("Gemm\n");
  printf("%f miliseconds \n", milliseconds2_2);
  printf("FLOPS\n");
  printf("%f GFLOPS \n", ((double)a*a*a*1.0e-6f/(double)milliseconds2_2));
*/


////////////////////////////////////////////////////////////////////RunResult////////////////////////////////////////////////////
printf("\nMAtrix A host to GPU: ");
printf("%.2f  \n", millisecondsTA);
 printf("Vector B host to GPU: ");
  printf("%.2f \n", millisecondsTB);
printf("\nVector CUDA result GPU to host: ");
printf("%.2f  \n", millisecondsTC);
 printf("Vector Ozaki result GPU to host: ");
  printf("%.2f \n", millisecondsTD);

 printf("\nUsing CUDACore on GEMV\n");
 printf("Handle: ");
 printf("%.2f  \n", milliseconds1_1);
 printf("Gemm: ");
 printf("%.2f \n", milliseconds1_2);

 double  GF1=(2.0*(m*(m))/(milliseconds1_2))*1e-9;
 printf("GFLOPS: ");
 printf("%.2f\n",GF1);//////////////////
 double  BW1=(m*((m +2)))  * sizeof(double) / milliseconds1_2 *1e-9 ;
 printf("BandWidth(GB/s): ");
 printf("%.2f\n",BW1);//////////////////
 printf("\n");


  printf("\nUsing TensorCore on OzakiScheme\n");
  printf("Handle: ");
  printf("%.2f  \n", milliseconds2_1);
  printf("Gemm: ");
  printf("%.2f \n", milliseconds2_2);
  printf("GFLOPS: ");
  double  GF2=(2.0*(m*(m))/milliseconds2_2)*1e-9;
  printf("%.2f \n", GF2);/////////////////////
  double BW2=(m*((m +2)* 1e-6)) * sizeof(double) / milliseconds2_2 *1e-9 ;
  printf("BandWidth(GB/s): ");
  printf("%.2f \n", BW2);/////////////////////

  printf("\n");

  printf("Speed Up rate: ");
  printf("%.2f\n", (double)milliseconds1_2/((double)milliseconds2_2) );

printf("\n");


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // CUDA_CHECK(cudaStreamSynchronize(stream));

  /*
   *   C = | 19.0 | 22.0 |
   *       | 43.0 | 50.0 |
   * 
   * 
   * 
   printf("CUDA\n");
   print_matrix(m, 1, C.data(), ldc);
   
   printf("=====\n");

   printf("Ozaki\n");
   print_matrix(m, 1, D.data(), ldd);

    printf("=====\n");


*/

if(m<16 && n < 16){

//大いなる真実の出力確認/*
printf("\nCUDA\n");
   print_matrix(m, m, C, ldc);
   
   printf("=====\n");

   printf("\nOzaki\n");
   print_matrix(m, m, D, ldd);

    printf("=====\n");
   


   printf("Difference\n");
double tempdif = 0;


for(int i = 0; i<m; i++){

tempdif = +(C[i] - D[i]);

printf("%f\n",tempdif);

}






}




//checkSpMV






















  /* free resources */
cudaFree(d_A);
cudaFree(TMP_MAT);
cudaFree(d_B);
cudaFree(d_D);
cudaFree(TMP_VEC);
cudaFree(TMP_C);
cudaFree(SplitMatScaling);
cudaFree(SplitVecScaling);
cudaFree(SplitVec);
cudaFree(SplitMat);

  // CUDA_CHECK(cudaStreamDestroy(stream));
cudaDeviceReset();

  // ms単位でstartとstopの差を計算する。

  // 終了処理

 printf("\nSpMV開始\n");

 return 0;
////////////////////////////////////////////////////////////////////////////////////////////////


    Coo sCoo;
    Csr sCsr;
   
    RunLen sRunLen;
    
    double *dVector = NULL;
    double *dAnsCsr = NULL;  
    double *dAnsRunLen = NULL;
    
    char *Filename;
    char *FlagSpMV;
    //char *BaseFilename;

    struct timeval s, e;
    FlagSpMV = argv[3];
    float percent =atoi(argv[4]);
    


    if (FlagSpMV == "1")
    {
      Filename = argv[4];

     //BaseFilename = argv[2];
    
    //行列ファイルをCOO形式で格納
   sCoo = ReadMatrix( Filename , sCoo );
    
      
    //CSR SpMV -----------------------------------------
    printf("\n");
    printf("START: CSR\n");

    sCsr = Coo2Csr( sCoo, sCsr );  
    
    dVector = MakeVector(sCsr.NumRow, dVector);
    dAnsCsr = MakeAns(sCsr.NumRow, dAnsCsr); 
    dAnsCsr = CsrSpMVGPU( sCsr, dVector, dAnsCsr );

    CsrSpMVGPU( sCsr, dVector, dAnsCsr );

    printf("CSR CPU time = %lf (msec)\n", ((e.tv_sec - s.tv_sec) + (e.tv_usec - s.tv_usec)*1.0E-6)*1000);
    FreeCsr(sCsr);
    
    }
    else if (FlagSpMV == "0")
    {
      sCoo = MakeSparseMatrix(m,m,percent,sCoo);
      sCsr = Coo2Csr( sCoo, sCsr );  
      dAnsCsr = MakeAns(sCsr.NumRow, dAnsCsr); 
      dAnsCsr = CsrSpMVGPU( sCsr, dVector, dAnsCsr );
      CsrSpMVGPU( sCsr, dVector, dAnsCsr );

  
    }
    
    
    



























  return EXIT_SUCCESS;





}

