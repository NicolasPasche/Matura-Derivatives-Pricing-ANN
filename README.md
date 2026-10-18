# Derivatives Pricing Engine

**European option pricing using analytical models, Monte Carlo simulation, and artificial neural networks.**

A C++ and Python project developed as part of my 2026 Matura thesis in Switzerland.

## Overview

This project investigates whether artificial neural networks can serve as
efficient surrogate models for pricing European call options under the
Merton Jump Diffusion model.

The implementation compares four pricing approaches:

- **Merton Jump Diffusion:** Semi-analytical pricing incorporating jumps.
- **Monte Carlo:** Stochastic simulation of the underlying asset price.
- **Artificial Neural Networks:** Neural network approximations trained
  on analytical and Monte Carlo pricing data.

The primary objective is to compare pricing accuracy and computational
performance.

## Technologies

- C++17
- Python
- PyTorch / LibTorch
- CUDA
- NumPy, Pandas, Matplotlib

## Benchmark Results

### Pricing Accuracy

Evaluated on 75,000 held-out European call options, using the
semi-analytical Merton price as reference.

| Method | MAE | RMSE |
|---|---:|---:|
| Monte Carlo (5,000 paths) | 0.3441 | 0.5601 |
| ANN trained on Monte Carlo | 0.1484 | 0.1967 |
| ANN trained on analytical prices | **0.1155** | **0.1512** |

The analytically trained ANN achieved the lowest prediction error
among the three approximations.

### Computational Performance

GPU-accelerated batched ANN inference was compared against a
sequential CPU implementation of the semi-analytical Merton pricer.

At 100,000 options, the GPU ANN achieved approximately **17.6x speedup**.

Performance depends on batch size, hardware, and implementation details.

## Benchmark Figures

### Accuracy Comparison

![Accuracy Comparison](results/figures/benchmarks/accuracy_comparison.pdf)

### Computational Speedup

![Speedup](results/figures/benchmarks/speedup.pdf)

Additional figures are available in
[`results/figures/benchmarks/`](results/figures/benchmarks/).

## Project Structure

```text
derivatives_pricing_eninge/
├── code/
│   ├── benchmarks/
│   ├── plots/
│   ├── src/
│   │   ├── derivatives/
│   │   ├── models/
│   │   ├── pricers/
│   │   └── utils/
│   ├── tests/
│   └── CMakeLists.txt
├── data/
├── documentation/
├── models/
└── results/
```

## Build

Requirements:

- C++17-compatible compiler
- CMake >= 3.18
- LibTorch
- CUDA Toolkit

The LibTorch installation path must currently be configured in
`code/CMakeLists.txt`.

```bash
cd code

cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release

cmake --build build-release --target main -j2

./build-release/main
```

## Tests

```bash
cmake --build build-release \
    --target test_analytical_pricer test_merton_analytical_pricer test_ann_pricer \
    -j2

./build-release/test_analytical_pricer
./build-release/test_merton_analytical_pricer

./build-release/test_ann_pricer ../models/best_model_analytical.pt analytical
./build-release/test_ann_pricer ../models/best_model_mc_comparison.pt
```

## Disclaimer

This project was developed for educational and research purposes.

**I do not take any responsibility for you losing money with this software.**

## Author

**Nicolas Pasche**

Matura Thesis — 2026

<img width="279" height="259" alt="grafik" src="https://github.com/user-attachments/assets/2faa4544-9d01-4fa0-ab4c-5f659391ef95" />
