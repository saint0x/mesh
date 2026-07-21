#ifndef __AMDGCN_WAVEFRONT_SIZE
#define __AMDGCN_WAVEFRONT_SIZE 32
#endif

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

extern "C" {
struct hipblasContext;
using hipblasHandle_t = hipblasContext*;
using hipblasStatus_t = int;
using hipblasOperation_t = int;
static constexpr hipblasStatus_t HIPBLAS_STATUS_SUCCESS = 0;
static constexpr hipblasOperation_t HIPBLAS_OP_N = 111;
hipblasStatus_t hipblasCreate(hipblasHandle_t* handle);
hipblasStatus_t hipblasSgemm(hipblasHandle_t handle, hipblasOperation_t transa,
                             hipblasOperation_t transb, int m, int n, int k,
                             const float* alpha, const float* A, int lda,
                             const float* B, int ldb, const float* beta,
                             float* C, int ldc);
}

namespace {

thread_local char g_last_error[512] = {0};

void set_error(const char* api, const char* message) {
  std::snprintf(g_last_error, sizeof(g_last_error), "%s: %s", api, message);
}

bool check_hip(const char* api, hipError_t status) {
  if (status == hipSuccess) {
    return true;
  }
  set_error(api, hipGetErrorString(status));
  return false;
}

bool check_hipblas(const char* api, hipblasStatus_t status) {
  if (status == HIPBLAS_STATUS_SUCCESS) {
    return true;
  }
  std::snprintf(g_last_error, sizeof(g_last_error), "%s: hipBLAS status %d", api,
                static_cast<int>(status));
  return false;
}

hipblasHandle_t blas_handle() {
  static hipblasHandle_t handle = [] {
    hipblasHandle_t value = nullptr;
    if (hipblasCreate(&value) != HIPBLAS_STATUS_SUCCESS) {
      return static_cast<hipblasHandle_t>(nullptr);
    }
    return value;
  }();
  return handle;
}

std::size_t blocks_for(std::size_t len) { return (len + 255) / 256; }

bool launch_ok(const char* api) {
  return check_hip(api, hipGetLastError()) &&
         check_hip("hipDeviceSynchronize", hipDeviceSynchronize());
}

__global__ void fill_kernel(float* out, float value, std::size_t len) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) out[idx] = value;
}

__global__ void add_kernel(const float* lhs, const float* rhs, float* out,
                           std::size_t len) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) out[idx] = lhs[idx] + rhs[idx];
}

__global__ void mul_kernel(const float* lhs, const float* rhs, float* out,
                           std::size_t len) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) out[idx] = lhs[idx] * rhs[idx];
}

__global__ void silu_kernel(const float* input, float* out, std::size_t len) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    const float x = input[idx];
    out[idx] = x / (1.0f + expf(-x));
  }
}

__global__ void embedding_kernel(const float* table, const uint32_t* tokens,
                                 float* out, std::size_t token_count,
                                 std::size_t hidden_dim) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t len = token_count * hidden_dim;
  if (idx < len) {
    const std::size_t token_row = idx / hidden_dim;
    const std::size_t col = idx - token_row * hidden_dim;
    out[idx] = table[static_cast<std::size_t>(tokens[token_row]) * hidden_dim + col];
  }
}

__global__ void rms_norm_kernel(const float* input, const float* gamma,
                                float* out, std::size_t rows,
                                std::size_t cols, float eps) {
  const std::size_t row = blockIdx.x;
  if (row >= rows) return;

  extern __shared__ float scratch[];
  float sum = 0.0f;
  for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = input[row * cols + col];
    sum += value * value;
  }
  scratch[threadIdx.x] = sum;
  __syncthreads();

  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }

  const float inv_rms = rsqrtf((scratch[0] / static_cast<float>(cols)) + eps);
  for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
    out[row * cols + col] = input[row * cols + col] * inv_rms * gamma[col];
  }
}

__global__ void rope_kernel(const float* input, const uint32_t* positions,
                            float* out, std::size_t rows, std::size_t cols,
                            std::size_t head_dim, float base) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t half_dim = head_dim / 2;
  const std::size_t num_heads = cols / head_dim;
  const std::size_t pairs = rows * num_heads * half_dim;
  if (idx >= pairs) return;

  const std::size_t dim = idx % half_dim;
  const std::size_t head_linear = idx / half_dim;
  const std::size_t head = head_linear % num_heads;
  const std::size_t row = head_linear / num_heads;
  const std::size_t offset = row * cols + head * head_dim;
  const float theta = static_cast<float>(positions[row]) *
                      powf(base, -static_cast<float>(dim * 2) /
                                      static_cast<float>(head_dim));
  const float s = sinf(theta);
  const float c = cosf(theta);
  const float left = input[offset + dim];
  const float right = input[offset + half_dim + dim];
  out[offset + dim] = left * c - right * s;
  out[offset + half_dim + dim] = left * s + right * c;
}

__global__ void copy_columns_kernel(const float* input, float* out,
                                    std::size_t rows, std::size_t input_cols,
                                    std::size_t col_start,
                                    std::size_t col_count) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t len = rows * col_count;
  if (idx < len) {
    const std::size_t row = idx / col_count;
    const std::size_t col = idx - row * col_count;
    out[idx] = input[row * input_cols + col_start + col];
  }
}

__global__ void copy_rows_kernel(const float* input, float* out,
                                 std::size_t rows, std::size_t cols,
                                 std::size_t dst_row_start) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t len = rows * cols;
  if (idx < len) {
    const std::size_t row = idx / cols;
    const std::size_t col = idx - row * cols;
    out[(dst_row_start + row) * cols + col] = input[row * cols + col];
  }
}

__global__ void copy_rows_range_kernel(const float* input, float* out,
                                       std::size_t row_count,
                                       std::size_t cols,
                                       std::size_t src_row_start,
                                       std::size_t dst_row_start) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t len = row_count * cols;
  if (idx < len) {
    const std::size_t row = idx / cols;
    const std::size_t col = idx - row * cols;
    out[(dst_row_start + row) * cols + col] =
        input[(src_row_start + row) * cols + col];
  }
}

__global__ void add_range_kernel(float* dst, const float* update,
                                 std::size_t offset, std::size_t len) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) dst[offset + idx] += update[idx];
}

__global__ void attention_kernel(const float* q, const float* k_cache,
                                 const float* v_cache,
                                 const uint32_t* local_kv_indices,
                                 float* out, std::size_t q_rows,
                                 std::size_t q_cols,
                                 std::size_t kv_cols,
                                 std::size_t cache_prefix_len,
                                 std::size_t cache_seq_len,
                                 std::size_t head_dim) {
  const std::size_t row = blockIdx.x;
  const std::size_t q_head = blockIdx.y;
  if (row >= q_rows) return;

  const std::size_t local_q_heads = q_cols / head_dim;
  if (q_head >= local_q_heads) return;
  const std::size_t kv_head = static_cast<std::size_t>(local_kv_indices[q_head]);
  const std::size_t visible = min(cache_prefix_len + row + 1, cache_seq_len);
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const float* q_head_base = q + row * q_cols + q_head * head_dim;

  extern __shared__ float shared_attention[];
  float* reduce = shared_attention;
  float* scores = shared_attention + blockDim.x;

  float max_score = -INFINITY;
  for (std::size_t seq = 0; seq < visible; ++seq) {
    const float* k_head_base = k_cache + seq * kv_cols + kv_head * head_dim;
    float score = 0.0f;
    for (std::size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      score += q_head_base[d] * k_head_base[d];
    }
    reduce[threadIdx.x] = score;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reduce[threadIdx.x] += reduce[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      const float scaled = reduce[0] * scale;
      scores[seq] = scaled;
      max_score = fmaxf(max_score, scaled);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) reduce[0] = max_score;
  __syncthreads();
  max_score = reduce[0];

  float denom_part = 0.0f;
  for (std::size_t seq = threadIdx.x; seq < visible; seq += blockDim.x) {
    denom_part += expf(scores[seq] - max_score);
  }
  reduce[threadIdx.x] = denom_part;
  __syncthreads();

  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float denom = reduce[0];

  for (std::size_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float accum = 0.0f;
    for (std::size_t seq = 0; seq < visible; ++seq) {
      const float weight = expf(scores[seq] - max_score);
      accum += weight * v_cache[seq * kv_cols + kv_head * head_dim + dim];
    }
    out[row * q_cols + q_head * head_dim + dim] = accum / denom;
  }
}

}  // namespace

extern "C" const char* meshnet_rocm_last_error() { return g_last_error; }

extern "C" int meshnet_rocm_device_count(int* count) {
  return check_hip("hipGetDeviceCount", hipGetDeviceCount(count)) ? 0 : 1;
}

extern "C" int meshnet_rocm_malloc(float** ptr, std::size_t len) {
  return check_hip("hipMalloc", hipMalloc(reinterpret_cast<void**>(ptr),
                                          len * sizeof(float)))
             ? 0
             : 1;
}

extern "C" int meshnet_rocm_malloc_u32(uint32_t** ptr, std::size_t len) {
  return check_hip("hipMalloc", hipMalloc(reinterpret_cast<void**>(ptr),
                                          len * sizeof(uint32_t)))
             ? 0
             : 1;
}

extern "C" int meshnet_rocm_free(void* ptr) {
  return check_hip("hipFree", hipFree(ptr)) ? 0 : 1;
}

extern "C" int meshnet_rocm_upload_f32(float* dst, const float* src,
                                       std::size_t len) {
  return check_hip("hipMemcpyH2D",
                   hipMemcpy(dst, src, len * sizeof(float),
                             hipMemcpyHostToDevice))
             ? 0
             : 1;
}

extern "C" int meshnet_rocm_download_f32(float* dst, const float* src,
                                         std::size_t len) {
  return check_hip("hipMemcpyD2H",
                   hipMemcpy(dst, src, len * sizeof(float),
                             hipMemcpyDeviceToHost))
             ? 0
             : 1;
}

extern "C" int meshnet_rocm_upload_u32(uint32_t* dst, const uint32_t* src,
                                       std::size_t len) {
  return check_hip("hipMemcpyH2D",
                   hipMemcpy(dst, src, len * sizeof(uint32_t),
                             hipMemcpyHostToDevice))
             ? 0
             : 1;
}

extern "C" int meshnet_rocm_fill(float* out, float value, std::size_t len) {
  fill_kernel<<<blocks_for(len), 256>>>(out, value, len);
  return launch_ok("fill_kernel") ? 0 : 1;
}

extern "C" int meshnet_rocm_add(const float* lhs, const float* rhs, float* out,
                                std::size_t len) {
  add_kernel<<<blocks_for(len), 256>>>(lhs, rhs, out, len);
  return launch_ok("add_kernel") ? 0 : 1;
}

extern "C" int meshnet_rocm_mul(const float* lhs, const float* rhs, float* out,
                                std::size_t len) {
  mul_kernel<<<blocks_for(len), 256>>>(lhs, rhs, out, len);
  return launch_ok("mul_kernel") ? 0 : 1;
}

extern "C" int meshnet_rocm_silu(const float* input, float* out,
                                 std::size_t len) {
  silu_kernel<<<blocks_for(len), 256>>>(input, out, len);
  return launch_ok("silu_kernel") ? 0 : 1;
}

extern "C" int meshnet_rocm_embedding(const float* table, const uint32_t* tokens,
                                      float* out, std::size_t token_count,
                                      std::size_t hidden_dim) {
  const std::size_t len = token_count * hidden_dim;
  embedding_kernel<<<blocks_for(len), 256>>>(table, tokens, out, token_count,
                                             hidden_dim);
  return launch_ok("embedding_kernel") ? 0 : 1;
}

extern "C" int meshnet_rocm_rms_norm(const float* input, const float* gamma,
                                     float* out, std::size_t rows,
                                     std::size_t cols, float eps) {
  const unsigned threads = 256;
  rms_norm_kernel<<<rows, threads, threads * sizeof(float)>>>(input, gamma, out,
                                                              rows, cols, eps);
  return launch_ok("rms_norm_kernel") ? 0 : 1;
}

extern "C" int meshnet_rocm_rope(const float* input, const uint32_t* positions,
                                 float* out, std::size_t rows,
                                 std::size_t cols, std::size_t head_dim,
                                 float base) {
  const std::size_t pairs = rows * (cols / head_dim) * (head_dim / 2);
  rope_kernel<<<blocks_for(pairs), 256>>>(input, positions, out, rows, cols,
                                          head_dim, base);
  return launch_ok("rope_kernel") ? 0 : 1;
}

extern "C" int meshnet_rocm_matmul(const float* a, const float* b, float* c,
                                   std::size_t m, std::size_t k,
                                   std::size_t n) {
  hipblasHandle_t handle = blas_handle();
  if (handle == nullptr) {
    set_error("hipblasCreate", "failed to initialize hipBLAS handle");
    return 1;
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  return check_hipblas(
             "hipblasSgemm",
             hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                          static_cast<int>(n), static_cast<int>(m),
                          static_cast<int>(k), &alpha, b, static_cast<int>(n),
                          a, static_cast<int>(k), &beta, c,
                          static_cast<int>(n)))
             ? 0
             : 1;
}

extern "C" int meshnet_rocm_copy_columns(const float* input, float* out,
                                         std::size_t rows,
                                         std::size_t input_cols,
                                         std::size_t col_start,
                                         std::size_t col_count) {
  const std::size_t len = rows * col_count;
  copy_columns_kernel<<<blocks_for(len), 256>>>(input, out, rows, input_cols,
                                                col_start, col_count);
  return launch_ok("copy_columns_kernel") ? 0 : 1;
}

extern "C" int meshnet_rocm_copy_rows(const float* input, float* out,
                                      std::size_t rows, std::size_t cols,
                                      std::size_t dst_row_start) {
  const std::size_t len = rows * cols;
  copy_rows_kernel<<<blocks_for(len), 256>>>(input, out, rows, cols,
                                             dst_row_start);
  return launch_ok("copy_rows_kernel") ? 0 : 1;
}

extern "C" int meshnet_rocm_copy_rows_range(const float* input, float* out,
                                            std::size_t row_count,
                                            std::size_t cols,
                                            std::size_t src_row_start,
                                            std::size_t dst_row_start) {
  const std::size_t len = row_count * cols;
  copy_rows_range_kernel<<<blocks_for(len), 256>>>(
      input, out, row_count, cols, src_row_start, dst_row_start);
  return launch_ok("copy_rows_range_kernel") ? 0 : 1;
}

extern "C" int meshnet_rocm_add_range(float* dst, const float* update,
                                      std::size_t offset, std::size_t len) {
  add_range_kernel<<<blocks_for(len), 256>>>(dst, update, offset, len);
  return launch_ok("add_range_kernel") ? 0 : 1;
}

extern "C" int meshnet_rocm_attention(const float* q, const float* k_cache,
                                      const float* v_cache,
                                      const uint32_t* local_kv_indices,
                                      float* out, std::size_t q_rows,
                                      std::size_t q_cols,
                                      std::size_t kv_cols,
                                      std::size_t cache_prefix_len,
                                      std::size_t cache_seq_len,
                                      std::size_t head_dim) {
  const dim3 grid(static_cast<unsigned>(q_rows),
                  static_cast<unsigned>(q_cols / head_dim));
  const unsigned threads = 256;
  const std::size_t shared = (threads + cache_seq_len) * sizeof(float);
  attention_kernel<<<grid, threads, shared>>>(
      q, k_cache, v_cache, local_kv_indices, out, q_rows, q_cols, kv_cols,
      cache_prefix_len, cache_seq_len, head_dim);
  return launch_ok("attention_kernel") ? 0 : 1;
}
