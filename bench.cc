#include <mkl_spblas.h>
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Compute y = A * x
// A is M x N sparse matrix with BS x BS block sizes
int main() {
  using namespace std;

  omp_set_num_threads(1);

  // Generate random block sparse matrix
  constexpr int M = 1024;
  constexpr int N = 1024;
  constexpr int BS = 4;
  constexpr float SPARSITY = 0.95;

  int num_block_rows = (M + BS - 1) / BS;
  int num_block_cols = (N + BS - 1) / BS;
  int num_blocks = num_block_rows * num_block_cols;
  int num_nnz_blocks = num_blocks * (1 - SPARSITY);

  // Randomly pick non-zero blocks
  vector<int> block_ids(num_blocks);
  iota(block_ids.begin(), block_ids.end(), 0);
  unsigned seed = 0;
  default_random_engine gen;
  shuffle(block_ids.begin(), block_ids.end(), gen);
  sort(block_ids.begin(), block_ids.begin() + num_nnz_blocks);

  // Fill rowptr
  int current_rowptr = 0;
  vector<int> rowptr(num_block_rows + 1);
  for (int i = 0; i < num_block_rows; ++i) {
    rowptr[i] = current_rowptr;
    for (; block_ids[current_rowptr] / num_block_cols == i; ++current_rowptr)
      ;
  }
  rowptr[num_block_rows] = num_nnz_blocks;

  // Fill colidx and randomly generate values
  vector<int> colidx(num_nnz_blocks);
  vector<float> values(num_nnz_blocks * BS * BS);
  uniform_real_distribution<float> dist(-1.0f, 1.0f);
  int current_block_row = 0;
  for (int i = 0; i < num_nnz_blocks; ++i) {
    int block_id = block_ids[i];
    int block_row = block_id / num_block_cols;
    int block_col = block_id % num_block_cols;

    colidx[i] = block_col;
    generate_n(values.begin() + i * BS * BS, BS * BS, [&dist, &gen]() {
      return dist(gen);
    });
  }

  // Randomly generate x
  vector<float> x(N);
  generate_n(x.begin(), N, [&dist, &gen]() { return dist(gen); });
  vector<float> y_ref(M);
  generate_n(y_ref.begin(), M, [&dist, &gen]() { return dist(gen); });

  // MKL BSR
  constexpr int NUM_ITER = 16;
  constexpr int NUM_WARMUP = 1;
  chrono::time_point<chrono::system_clock> t_begin, t_end;
  for (int i = 0; i < NUM_ITER + NUM_WARMUP; ++i) {
    if (i == NUM_WARMUP) {
      t_begin = chrono::system_clock::now();
    }
    mkl_cspblas_sbsrgemv(
        "N",
        &num_block_rows,
        &BS,
        values.data(),
        rowptr.data(),
        colidx.data(),
        x.data(),
        y_ref.data());
  }
  t_end = chrono::system_clock::now();
  double dt = chrono::duration<double>(t_end - t_begin).count();
  cout << "MKL BCSR Effective GF/s " << (2. * M * N * NUM_ITER) / dt / 1e9 << endl;

  // MKL inspector-executor BSR
  vector<float> y(M);
  generate_n(y.begin(), M, [&dist, &gen]() { return dist(gen); });
  sparse_matrix_t a_handle;
  sparse_status_t ret = mkl_sparse_s_create_bsr(
      &a_handle,
      SPARSE_INDEX_BASE_ZERO,
      SPARSE_LAYOUT_ROW_MAJOR,
      num_block_rows,
      num_block_cols,
      BS,
      rowptr.data(),
      rowptr.data() + 1,
      colidx.data(),
      values.data());
  assert(ret == SPARSE_STATUS_SUCCESS);

  ret = mkl_sparse_set_mv_hint(
      a_handle,
      SPARSE_OPERATION_NON_TRANSPOSE,
      {.type = SPARSE_MATRIX_TYPE_GENERAL},
      100);
  assert(ret == SPARSE_STATUS_SUCCESS);

  ret = mkl_sparse_optimize(a_handle);
  assert(ret == SPARSE_STATUS_SUCCESS);

  for (int i = 0; i < NUM_ITER + NUM_WARMUP; ++i) {
    if (i == NUM_WARMUP) {
      t_begin = chrono::system_clock::now();
    }
    ret = mkl_sparse_s_mv(
        SPARSE_OPERATION_NON_TRANSPOSE,
        1.0f,
        a_handle,
        {.type = SPARSE_MATRIX_TYPE_GENERAL},
        x.data(),
        0.0f,
        y.data());
    assert(ret == SPARSE_STATUS_SUCCESS);
  }
  t_end = chrono::system_clock::now();
  dt = chrono::duration<double>(t_end - t_begin).count();
  cout << "MKL IE BCSR Effective GF/s " << (2. * M * N * NUM_ITER) / dt / 1e9
       << endl;

  float atol = 1e-5, rtol = 1e-5;
  for (int i = 0; i < y.size(); ++i) {
    float expected = y_ref[i];
    float actual = y[i];
    float error = fabs(actual - expected);
    if (error > atol && error / fabs(expected) > rtol) {
      cerr << "Correctness check failed at " << i << ": actual " << actual
           << " expected " << expected << endl;
    }
  }

  mkl_sparse_destroy(a_handle);

  return 0;
}
