#include "randutil.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <immintrin.h> // AVX, AVX2, FMA, AVX-512

// global variables to store the matrix

double* M = nullptr;
double* X = nullptr;
double* Y = nullptr;
double* YAVX = nullptr;
int N = 8;

#define ALIGN 64
#define VectorLength 4
   
// implementation of the matrix-vector multiply function
void AVXMatrixVectorMultiply(double* Y, const double* X)
{  
      const int n = (N/VectorLength);
   // __m256* Xavx = aligned_alloc(ALIGN, sizeof(__m256)*n);
   // __m256* Mavx = aligned_alloc(ALIGN, sizeof(__m256)*n);
   // __m256 MXavx = aligned_alloc(ALIGN, sizeof(__m256)*n);

   
   for (int i = 0; i < N; i++)
   {
      Y[i] = 0;
      for(int j=0; j<n; j++)
      {
         __m256d Mavx = _mm256_loadu_pd(M + (i*N+j*VectorLength));
        std::cout << "Mavx = [ " ;
        for(int k=0; k<4; k++)
        {
            std::cout << Mavx[k] <<" "; 
        }
        std::cout << " ]"<<std::endl;
        
         __m256d Xavx =  _mm256_loadu_pd((X +(j*VectorLength)));
         __m256d MXavx = _mm256_mul_pd(Mavx,Xavx);
         double* Ysum = (double*)&MXavx;
         Y[i] += Ysum[0]+Ysum[1]+MXavx[2]+MXavx[3];
      }
      
   }  

}
void MatrixVectorMultiply(double* Y, const double* X)
{
   for (int i = 0; i < N; ++i)
   {
      Y[i] = 0;
      for (int j = 0; j < N; ++j)
      {
         Y[i] += M[i*N+j] * X[j];
      }
   }
}


int main()
{
    randutil::seed(4);
    X = static_cast<double*>(malloc(N*sizeof(double)));
    
    std::cout << "X = [ " ;
    for (int i = 0; i < N; ++i)
    {
        X[i] = randutil::randn();
        std::cout << X[i] <<" "; 
    }
    std::cout << " ]"<<std::endl;
    M = static_cast<double*>(malloc(N*N*sizeof(double)));
   std::cout << "M = [ " ;
  
   for (int i = 0; i < N; ++i)
   {
       
      M[i*N+i] = std::sqrt(2.0) * randutil::randn();
      
      for (int j = 0; j < N; ++j)
      {
         M[i*N + j] = randutil::randn();
         std::cout << M[i*N + j] <<" "; 
      }
      std::cout << std::endl;
   }

    Y = static_cast<double*>(malloc(N*sizeof(double)));
    MatrixVectorMultiply(Y, X);
    std::cout << "Y = [ " ;
    for (int i = 0; i < N; ++i)
    {
        std::cout << Y[i] <<" "; 
    }
    std::cout << " ]"<<std::endl;

    YAVX = static_cast<double*>(malloc(N*sizeof(double)));
    AVXMatrixVectorMultiply(YAVX, X);
    std::cout << "YAVX = [ " ;
    for (int i = 0; i < N; ++i)
    {
        std::cout << YAVX[i] <<" "; 
    }
    std::cout << " ]"<<std::endl;



return 0;
}