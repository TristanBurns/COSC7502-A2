// COSC3500, Semester 2, 2021
// Assignment 2
// Main file - avx version

#include "eigensolver.h"
#include "randutil.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <immintrin.h> // AVX, AVX2, FMA, AVX-512

// global variables to store the matrix

double* M = nullptr;
int N = 0;
# define VectorLength 4 //int VectorLength = 256/(8*sizeof(double));

// implementation of the matrix-vector multiply function
void MatrixVectorMultiply(double* Y, const double* X)
{  
   const int n = (N/VectorLength);  
   for (int i = 0; i < N; i++)
   {
      Y[i] = 0;
      for(int j=0; j<n; j++)
      {
         __m256d Mavx = _mm256_loadu_pd(M+(i*N+j*VectorLength));
         __m256d Xavx =  _mm256_loadu_pd((X +(j*VectorLength)));
         __m256d MXavx = _mm256_mul_pd(Mavx,Xavx);
         Y[i] += MXavx[0]+MXavx[1]+MXavx[2]+MXavx[3];
      }
   }
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
