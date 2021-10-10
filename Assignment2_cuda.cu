// COSC3500, Semester 2, 2021
// Assignment 2
// Main file - serial version

#include "eigensolver.h"
#include "randutil.h"
#include <string>
#include <iostream>
#include <iomanip>

// global variables to store the matrix

double* M = nullptr;
int N = 0;
float* xDevice;
float* yDevice;
float* mDevice;
float* xFloat = nullptr;
float* mFloat = nullptr;
float* yFloat = nullptr;
int Threads;
int Blocks;


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
            sum += X[i]*M[(tid*N)+i]; //chec
        }
        Y[tid]=sum;
    }
}

// implementation of the matrix-vector multiply function
void MatrixVectorMultiply(double* Y, const double* X)
{  

   
    for (int i = 0 ; i < N; i++)
    {
        xFloat[i] = (float) X[i];
    }
    
    for (int i = 0 ; i < N; i++)
    {
        for (int j = 0 ; j < N; j++)
        {
            mFloat[i*N+j]  = (float) M[i*N+j];
        }
    }


   checkError(cudaMalloc(&xDevice, N*sizeof(float)));
   checkError(cudaMalloc(&yDevice, N*sizeof(float)));
   checkError(cudaMalloc(&mDevice, N*N*sizeof(float)));

   checkError(cudaMemcpy(xDevice, xFloat, sizeof(float)*N, cudaMemcpyHostToDevice));
   checkError(cudaMemcpy(mDevice, mFloat, sizeof(float)*N*N, cudaMemcpyHostToDevice));

   Threads = 1024;
   Blocks = (N+Threads-1)/Threads;
   kernel<<<Blocks, Threads>>>(xDevice, mDevice, yDevice, N);
   checkError(cudaDeviceSynchronize());
   checkError(cudaMemcpy(yFloat, yDevice, N*sizeof(float), cudaMemcpyDeviceToHost));
   

    for (int i = 0 ; i < N; i++)
    {
        Y[i]  = (double) yFloat[i];
    }

   checkError(cudaFree(xDevice));
   checkError(cudaFree(yDevice));
   checkError(cudaFree(mDevice));

}


int main(int argc, char** argv)
{
   // get the current time, for benchmarking
   auto StartTime = std::chrono::high_resolution_clock::now();

   // get the input size from the command line
   if (argc < 2)
   {
      std::cerr << "expected: matrix size <N>\n";
      return 1;
   }
   N = std::stoi(argv[1]);

   // Allocate memory for the matrix
   M = static_cast<double*>(malloc(N*N*sizeof(double)));
   mFloat = static_cast<float*>(malloc(N*N*sizeof(float)));
   xFloat = static_cast<float*>(malloc(N*sizeof(float)));
   yFloat = static_cast<float*>(malloc(N*sizeof(float)));
   // seed the random number generator to a known state
   randutil::seed(4);  // The standard random number.  https://xkcd.com/221/

   // Initialize the matrix.  This is a matrix from a Gaussian Orthogonal Ensemble.
   // The matrix is symmetric.
   // The diagonal entries are gaussian distributed with variance 2.
   // The off-diagonal entries are gaussian distributed with variance 1.
   for (int i = 0; i < N; ++i)
   {
      M[i*N+i] = std::sqrt(2.0) * randutil::randn();
      for (int j = i+1; j < N; ++j)
      {
         M[i*N + j] = M[j*N + i] = randutil::randn();
      }
   }
   auto FinishInitialization = std::chrono::high_resolution_clock::now();

   // Call the eigensolver
   EigensolverInfo Info = eigenvalues_arpack(N, 100);

   auto FinishTime = std::chrono::high_resolution_clock::now();

   auto InitializationTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishInitialization - StartTime);
   auto TotalTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishTime - StartTime);

   std::cout << "Obtained " << Info.Eigenvalues.size() << " eigenvalues.\n";
   std::cout << "The largest eigenvalue is: " << std::setw(16) << std::setprecision(12) << Info.Eigenvalues.back() << '\n';
   std::cout << "Total time:                             " << std::setw(12) << TotalTime.count() << " us\n";
   std::cout << "Time spent in initialization:           " << std::setw(12) << InitializationTime.count() << " us\n";
   std::cout << "Time spent in eigensolver:              " << std::setw(12) << Info.TimeInEigensolver.count() << " us\n";
   std::cout << "   Of which the multiply function used: " << std::setw(12) << Info.TimeInMultiply.count() << " us\n";
   std::cout << "   And the eigensolver library used:    " << std::setw(12) << (Info.TimeInEigensolver - Info.TimeInMultiply).count() << " us\n";
   std::cout << "Total serial (initialization + solver): " << std::setw(12) << (TotalTime - Info.TimeInMultiply).count() << " us\n";
   std::cout << "Number of matrix-vector multiplies:     " << std::setw(12) << Info.NumMultiplies << '\n';
   std::cout << "Time per matrix-vector multiplication:  " << std::setw(12) << (Info.TimeInMultiply / Info.NumMultiplies).count() << " us\n";

   // free memory
   free(M);

}
