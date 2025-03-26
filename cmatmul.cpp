#include <immintrin.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
  #define RESTRICT __restrict__
#else
  #define RESTRICT restrict
#endif

// Block sizes used by the optimised kernel.
const int BLOCK_M = 3;   // Process 3 rows at a time.
const int BLOCK_N = 32;  // Process 32 columns at a time (4 vectors of 8 floats).

//   C_block += A_block * B_block,
// where A_block is 3 x K, B_block is K x 32.
inline void matmul_dot_inner(int k, const float* RESTRICT A, int lda,
                             const float* RESTRICT B, int ldb,
                             float* RESTRICT C, int ldc) {
    const int regsA = 3; 
    const int regsB = 4; 
    __m256 csum[regsA][regsB];

    for (int i = 0; i < regsA; i++) {
        for (int j = 0; j < regsB; j++) {
            csum[i][j] = _mm256_setzero_ps();
        }
    }

    for (int p = 0; p < k; p++) {
        for (int bi = 0; bi < regsB; bi++) {
        // Load vector (8 floats) from B: element (p, bi*8)
            const float* RESTRICT bp = B + p * ldb + bi * 8;
            __m256 b_vec = _mm256_loadu_ps(bp);
            for (int ai = 0; ai < regsA; ai++) {
                // Load scalar from A and broadcast it.
                __m256 a_vec = _mm256_set1_ps(A[ai * lda + p]);
                csum[ai][bi] = _mm256_fmadd_ps(a_vec, b_vec, csum[ai][bi]);
            }
        }
    }

    for (int ai = 0; ai < regsA; ai++) {
        for (int bi = 0; bi < regsB; bi++) {
            float* cp = C + ai * ldc + bi * 8;
            __m256 c_val = _mm256_loadu_ps(cp);
            c_val = _mm256_add_ps(c_val, csum[ai][bi]);
            _mm256_storeu_ps(cp, c_val);
        }
    }
}

// We assume matrices are stored in row-major order.
// A is M x K, B is K x N, and C is M x N.
void matmul_opt_full(const float* RESTRICT A, const float* RESTRICT B, float* RESTRICT C,
                     int M, int N, int K,
                     int lda, int ldb, int ldc) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i += BLOCK_M) {
        for (int j = 0; j < N; j += BLOCK_N) {
            matmul_dot_inner(K, A + i * lda, lda,
                            B + j, ldb,
                            C + i * ldc + j, ldc);
        }
    }
}

// If M and N are not multiples of BLOCK_M and BLOCK_N, allocate padded
// matrices, fill extra areas with zero, and compute with the optimised kernel. 
void matmul_opt(const float* RESTRICT A, const float* RESTRICT B, float* RESTRICT C,
                int M, int N, int K) {
    if ((M % BLOCK_M == 0) && (N % BLOCK_N == 0)) {
        matmul_opt_full(A, B, C, M, N, K, K, N, N);
    } else {
        int padded_M = ((M + BLOCK_M - 1) / BLOCK_M) * BLOCK_M;
        int padded_N = ((N + BLOCK_N - 1) / BLOCK_N) * BLOCK_N;
        std::vector<float> A_pad(padded_M * K, 0.0f);
        std::vector<float> B_pad(K * padded_N, 0.0f);
        std::vector<float> C_pad(padded_M * padded_N, 0.0f);

        for (int i = 0; i < M; i++) {
            memcpy(&A_pad[i * K], &A[i * K], K * sizeof(A[0]));
        }

        for (int i = 0; i < K; i++) {
            memcpy(&B_pad[i * padded_N], &B[i * N], N * sizeof(B[0]));
        }

        matmul_opt_full(A_pad.data(), B_pad.data(), C_pad.data(),
                        padded_M, padded_N, K, K, padded_N, padded_N);

        for (int i = 0; i < M; i++) {
            memcpy(&C[i * N], &C_pad[i * padded_N], N * sizeof(C[0]));
        }
    }
}

// Naïve matrix multiplication for correctness checking.
void matmul_naive(const float* A, const float* B, float* C,
                int M, int N, int K,
                int lda, int ldb, int ldc) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int p = 0; p < K; p++) {
                sum += A[i * lda + p] * B[p * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

int main() {
    const int M_dim = 384;  
    const int N_dim = 320;  
    const int K_dim = 384;  

    // const int M_dim = 768;
    // const int N_dim = 640;
    // const int K_dim = 768;

    // A is M x K, so lda = K.
    // B is K x N, so ldb = N.
    // C is M x N, so ldc = N.

    const int lda = K_dim;
    const int ldb = N_dim;
    const int ldc = N_dim;

    std::vector<float> A(M_dim * lda);
    std::vector<float> B(K_dim * ldb);
    std::vector<float> C(M_dim * ldc, 0.0f);
    std::vector<float> C_ref(M_dim * ldc, 0.0f);

    for (int i = 0; i < M_dim * lda; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K_dim * ldb; i++) {
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    matmul_naive(A.data(), B.data(), C_ref.data(),
                M_dim, N_dim, K_dim,
                lda, ldb, ldc);

    // Warm up
    matmul_opt(A.data(), B.data(), C.data(), M_dim, N_dim, K_dim);

    std::fill(C.begin(), C.end(), 0.0f);

    const int num_runs = 10;
    auto start_opt = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < num_runs; r++) {
        std::fill(C.begin(), C.end(), 0.0f);
        matmul_opt(A.data(), B.data(), C.data(), M_dim, N_dim, K_dim);
    }
    auto end_opt = std::chrono::high_resolution_clock::now();
    double elapsed_opt = std::chrono::duration_cast<std::chrono::duration<double>>(end_opt - start_opt).count() / num_runs;
    double opt_gflops = (2.0 * M_dim * N_dim * K_dim) / (elapsed_opt * 1e9);
    std::cout << "Optimised MatMul average time: " << elapsed_opt << " seconds\n";
    std::cout << "Optimised Achieved GFLOPS: " << opt_gflops << "\n";

    std::vector<float> C_naive(M_dim * ldc, 0.0f);
    const int num_runs_naive = 1;  
    auto start_naive = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < num_runs_naive; r++) {
        std::fill(C_naive.begin(), C_naive.end(), 0.0f);
        matmul_naive(A.data(), B.data(), C_naive.data(),
                    M_dim, N_dim, K_dim,
                    lda, ldb, ldc);
    }
    auto end_naive = std::chrono::high_resolution_clock::now();
    double elapsed_naive = std::chrono::duration_cast<std::chrono::duration<double>>(end_naive - start_naive).count() / num_runs_naive;
    double naive_gflops = (2.0 * M_dim * N_dim * K_dim) / (elapsed_naive * 1e9);
    std::cout << "Naïve MatMul average time: " << elapsed_naive << " seconds\n";
    std::cout << "Naïve Achieved GFLOPS: " << naive_gflops << "\n";

    double totalError = 0.0;
    double maxError = 0.0;
    double refNorm = 0.0;
    for (size_t i = 0; i < C.size(); i++) {
        double diff = std::abs(C[i] - C_ref[i]);
        totalError += diff;
        maxError = std::max(maxError, diff);
        refNorm += std::abs(C_ref[i]);
    }
    double relativeError = totalError / refNorm;
    double avgError = totalError / C.size();

    std::cout << "Total L1 error: " << totalError << "\n";
    std::cout << "Relative error: " << relativeError << "\n";
    std::cout << "Average error per element: " << avgError << "\n";
    std::cout << "Max absolute error: " << maxError << "\n";

    if (relativeError < 1e-5) {
        std::cout << "Result is acceptable within floating-point tolerance.\n";
    } else {
        std::cout << "Result is incorrect.\n";
    }

    return 0;
}
