# CMATMUL - Cache Based Matrix Multiplication Kernel

CMATMUL is an optimised C++ implementation of matrix multiplication that leverages modern CPU features such as AVX2/FMA SIMD vectorisation and OpenMP multi-threading for increased throughput. This project examines a naïve triple-loop algorithm and uses the findings, presented [here](docs.md), to implement a high-performance kernel through techniques like cache tiling, register blocking, and memory access optimisation.


## Overview

Matrix multiplication is fundamental to many applications in computing, data analysis, and machine learning. However, the naïve approach underutilises modern hardware's capabilities significantly. CMATMUL tackles this by:

- **Memory Access Optimisation:** Reordering accesses to maximise cache line usage.
- **Register Blocking:** Keeping small blocks of data in CPU registers for rapid reuse.
- **Tiling for Cache Locality:** Dividing matrices into cache-friendly blocks.
- **SIMD Vectorisation:** Using AVX2/FMA intrinsics to process multiple floats concurrently.
- **OpenMP Parallelisation:** Distributing work across multiple cores to scale performance.

By combining these techniques, the optimised kernel significantly outperforms a naïve implementation, achieving over 100× the throughput in tested configurations.

## Getting Started

```bash
git clone https://github.com/ollycassidy13/CMATMUL
cd CMATMUL
```

### Compilation

Compile the code using the following command:

```bash
g++ -O3 -mavx2 -mfma -fopenmp -std=c++11 cmatmul.cpp -o cmatmul
```

> Note: A modern C++ compiler with support for C++11 (or later) and OpenMP, and a CPU with AVX2 and FMA support are needed

This command enables high optimisation, AVX2, FMA, and OpenMP to fully exploit the hardware capabilities.

### Use

#### With OpenMP (Multi-threaded):

```bash
export OMP_NUM_THREADS=4  # Set number of threads (e.g., 4)
./cmatmul
```

#### Without OpenMP (Single-threaded):

```bash
export OMP_NUM_THREADS=1
./cmatmul
```

## Documentation

Full documentation on how the example implementation's methodology works is provided in [docs.md](docs.md)

---

*Note: The code provided in this repository is intended for educational purposes. Users are encouraged to experiment with and modify the code to suit their specific hardware configurations and application requirements.*
