#include "eigensolver.h"
#include "randutil.h"
#include <string>
#include <iostream>
#include <iomanip>

// global variables to store the matrix

double* M = nullptr;
double* X = nullptr;
double* Y = nullptr;
double* Ycuda = nullptr;
int N = 9;


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
void kernel(float *X, float *M, float *Y, const int N)
{
    int tid=threadIdx.x+blockIdx.x*blockDim.x; //thread id
    float sum=0;
    if(tid<N)
    {
        for(int i=0; i<N; i++)
        {
            //sum += X[i]*M[(tid*N)+i]; 
            sum += X[i]*M[(i*N)+tid];//chec
        }
        Y[tid]=sum;
    }
}



// implementation of the matrix-vector multiply function
void cudaMatrixVectorMultiply(double* Y, const double* X)
{  
    float* xFloat = new float[N];
    float* mFloat = new float[N*N];
    float* yFloat = new float[N];
   
    for (int i = 0 ; i < N; i++)
    {
        xFloat[i] = (float) X[i];
    }
    
    for (int j = 0 ; j < N; j++)
    {
        for (int i = 0 ; i < N; i++)
        {
            //mFloat[i*N+j]  = (float) M[i*N+j];
            mFloat[j*N+i]  = (float) M[i*N+j];
        }
    }

   float* xDevice;
   float* yDevice;
   float* mDevice;
   checkError(cudaMalloc(&xDevice, N*sizeof(float)));
   checkError(cudaMalloc(&yDevice, N*sizeof(float)));
   checkError(cudaMalloc(&mDevice, N*N*sizeof(float)));

   checkError(cudaMemcpy(xDevice, xFloat, sizeof(float)*N, cudaMemcpyHostToDevice));
   checkError(cudaMemcpy(mDevice, mFloat, sizeof(float)*N*N, cudaMemcpyHostToDevice));

   int Threads = 256;
   int Blocks = (N+Threads-1)/Threads;
   kernel<<<Blocks, Threads>>>(xDevice, mDevice, yDevice, N);
   //checkError(cudaDeviceSynchronize());
   checkError(cudaMemcpy(yFloat, yDevice, N*sizeof(float), cudaMemcpyDeviceToHost));
   

    for (int i = 0 ; i < N; i++)
    {
        Y[i]  = (double) yFloat[i];
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


    Ycuda = static_cast<double*>(malloc(N*sizeof(double)));

  // allocate memory on the device

 

   cudaMatrixVectorMultiply(Ycuda,X);
   

    std::cout << "Ycuda = [ " ;
    for (int i = 0; i < N; ++i)
    {
        std::cout << Ycuda[i] <<" "; 
    }
    std::cout << " ]"<<std::endl;

    
    std::cout << "error = [ " ;
    for (int i = 0; i < N; ++i)
    {
        std::cout << Ycuda[i]-Y[i] <<" "; 
    }
    std::cout << " ]"<<std::endl;


    std::cout<<"sizeof double: "<<sizeof(double)<<std::endl;
return 0;
}