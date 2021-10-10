#include "randutil.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <immintrin.h> // AVX, AVX2, FMA, AVX-512

// global variables to store the matrix

double* M = nullptr;
double* X = nullptr;
double* Y = nullptr;
double* Ycuda = nullptr;
int N = 8;


#define VectorLength 4
#define VectorLength512 4

void checkError(cudaError_t e)
{
   if (e != cudaSuccess)
   {
      std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
      abort();
   }
}

__global__
void kernel(float *X, float *M, float *Y, const int N){
    int tid=threadIdx.x+blockIdx.x*blockDim.x; //thread id
        float sum=0;
    if(tid<N){
        for(int i=0; i<N; i++)
            sum += X[i]*M[(i*N)+tid];
        Y[tid]=sum;
    }
}



// implementation of the matrix-vector multiply function
void cudaMatrixVectorMultiply(double* Ycuda, double* yDevice, double* mDevice, double* XDevice)
{  
   checkError(cudaMemcpy(xDevice, X, sizeof(double)*N, cudaMemcpyHostToDevice));
   checkError(cudaMemcpy(mDevice, M, sizeof(double)*N*N, cudaMemcpyHostToDevice));

   int Threads = 256;
   int Blocks = (N+Threads-1)/Threads;
   kernel<<Blocks, Threads>>(xDevice, mDevice, yDevice, N, N);
   checkError(cudaDeviceSynchronize());
   checkError(cudaMemcpy(Ycuda, yDevice, N*sizeof(double), cudaMemcpyDeviceToHost));
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


    Ycuda = static_cast<double*>(malloc(N*sizeof(double)));

  // allocate memory on the device
   double* xDevice;
   double* yDevice;
   double* mDevice;
   checkError(cudaMalloc(&xDevice, N*sizeof(double)));
   checkError(cudaMalloc(&yDevice, N*sizeof(double)));
   checkError(cudaMalloc(&mDevice, N*N*sizeof(double)));
 

   cudaMatrixVectorMultiply(Ycuda,yDevice, mDevice, xDevice);
   

    std::cout << "Ycuda = [ " ;
    for (int i = 0; i < N; ++i)
    {
        std::cout << Ycuda[i] <<" "; 
    }
    std::cout << " ]"<<std::endl;


    std::cout<<"sizeof double: "<<sizeof(double)<<std::endl;
return 0;
}