# 🧠 Parallel Matrix Factorization for Recommender Systems (CUDA Accelerated)

This project implements a high-performance **Parallel Matrix Factorization (PMF)** algorithm designed for large-scale recommendation tasks. It optimizes **Stochastic Gradient Descent (SGD)** using **CuPy** to accelerate computations on the GPU, significantly improving convergence speed and scalability.

---

## 🚀 Key Features
- ⚡ **Parallelized SGD** with batch updates for speed and stability
- 🧠 **CuPy-based GPU acceleration** for high-throughput matrix ops
- 🧮 **Chunked RMSE computation** to fit in limited GPU memory
- 🤖 **Top-N Recommendations** for any user via dot-product scoring

---

## 🛠️ Tools & Technologies

- **Python 3.11**
- [CuPy](https://cupy.dev/) — CUDA array backend for GPU computation
- `pandas`, `numpy` — Data loading and preprocessing


---

## 📊 Dataset

- **MovieLens 32M** (`ratings.csv`)
- ~834 MB of user–movie interactions
- Fields: `userId`, `movieId`, `rating`
