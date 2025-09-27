#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>

namespace {

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;

struct FastTopKParams {
  const float *__restrict__ input; // [B, max_seq_len]
  int32_t *__restrict__ indices;   // [B, TopK]
  int32_t *__restrict__ lengths;   // [B]
  size_t max_seq_len;
};

// when length <= TopK, we can directly write the indices
__device__ void naive_topk_cuda(const float *__restrict__ score,
                                int32_t *__restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

static constexpr float kNegInf = std::numeric_limits<float>::lowest();

__device__ __forceinline__ uint32_t float_to_uint32(float x) {
  const uint32_t bits = __float_as_uint(x);
  // flip for descending order
  return (bits & 0x80000000u) ? bits : ~(bits | 0x80000000u);
}

__device__ uint32_t find_first_pos(uint32_t *histogram, uint32_t target) {
  constexpr auto kChunkSize = 256;
  constexpr auto kNumChunks = (1 << 15) / kChunkSize;
  static_assert(kThreadsPerBlock % kChunkSize == 0);

  __shared__ uint32_t s_cnt[kNumChunks];
  __shared__ uint32_t s_chunk;
  __shared__ uint32_t s_remain;
  __shared__ uint32_t s_return;

  const auto tid = threadIdx.x;
  if (tid < kNumChunks) {
    s_cnt[tid] = 0;
  }
  __syncthreads();

  // step 1: load to shared memory
  {
    uint32_t sum = 0;
    constexpr auto kStep = (1 << 15) / kThreadsPerBlock;
    for (int i = 0; i < kStep; ++i) {
      const auto idx = tid * kStep + i;
      sum += histogram[idx];
    }
    // write to shared memory
    const auto chunk_id = (tid * kStep) / kChunkSize;
    ::atomicAdd(s_cnt + chunk_id, sum);
    // s_cnt[i] is the sum of the i-th chunk
  }
  __syncthreads();

  // step 2: find the chunk
  static_assert(kThreadsPerBlock > kChunkSize, "need 1 thread to reduce");
  if (tid == kThreadsPerBlock - 1) {
    uint32_t prefix = 0;
    for (auto i = 0; i < kNumChunks; ++i) {
      const auto value = s_cnt[i];
      if (prefix + value > target) {
        s_chunk = i;
        s_remain = target - prefix;
        break;
      }
      prefix += value;
    }
  } else if (tid < kNumChunks) {
    // perform in-chunk prefix sum
    uint32_t prefix = 0;
    for (auto i = 0; i < kChunkSize; ++i) {
      const auto idx = tid * kChunkSize + i;
      prefix += histogram[idx];
      histogram[idx] = prefix;
    }
  }
  __syncthreads();

  // step 3: find the exact position
  const auto chunk = s_chunk;
  if (tid == chunk) {
    const auto remain = s_remain;
    // binary search on the chunk, find first >= remain
    auto i = tid * kChunkSize;
#pragma unroll
    for (auto j = kChunkSize >> 1; j > 0; j >>= 1) {
      if (histogram[i + j] <= remain)
        i += j;
    }
    s_return = i;
  }
  __syncthreads();

  return s_return;
}

__device__ void fast_topk_cuda(const float *__restrict__ score,
                               int32_t *__restrict__ indice, int32_t length) {
  extern __shared__ uint32_t s_histogram[]; // 32K items, 128KB

  // 64 registers, to avoid costly global memory access
  float local_scores[64];

  alignas(128) __shared__ int32_t s_counter;
  __shared__ uint32_t s_pos; // position of TopK
  constexpr auto kNumLoop = (1 << 15) / kThreadsPerBlock;
  const auto tid = threadIdx.x;
#pragma unroll kNumLoop
  for (auto i = 0; i < kNumLoop; ++i) {
    s_histogram[tid + i * kThreadsPerBlock] = 0;
  }
  s_counter = 0;
  __syncthreads();

  // step 1: histogram
#pragma unroll 64
  for (auto i = 0; i < 64; ++i) {
    const auto idx = tid + i * kThreadsPerBlock;
    const auto value = (idx < length) ? score[idx] : kNegInf;
    local_scores[i] = value;
    const auto bin = float_to_uint32(value) >> 17; // top 15 bits
    ::atomicAdd(s_histogram + bin, 1);
  }
  __syncthreads();

  // step 2: add those strictly no larger than s_pos
  const auto pos = find_first_pos(s_histogram, TopK);
#pragma unroll 64
  for (auto i = 0; i < 64; ++i) {
    const auto value = local_scores[i];
    const auto bin = float_to_uint32(value) >> 17; // top 15 bits
    if (bin <= pos) {
      const auto offset = ::atomicAdd(&s_counter, 1);
      // assert(offset && "offset should be positive");
      indice[offset] = tid + i * kThreadsPerBlock;
    }
  }
  __syncthreads();

// step 3: add those remaining at s_pos + 1
#pragma unroll 64
  for (auto i = 0; i < 64; ++i) {
    const auto value = local_scores[i];
    const auto bin = float_to_uint32(value) >> 17; // top 15 bits
    if (bin == pos + 1) {
      const auto offset = ::atomicAdd(&s_counter, 1);
      if (offset < TopK) {
        indice[offset] = tid + i * kThreadsPerBlock;
      }
    }
  }

  __syncthreads();
}

__global__ __launch_bounds__(kThreadsPerBlock) // max 1024 threads
    void topk_kernel(const FastTopKParams params) {
  const auto &[input, indices, lengths, max_seq_len] = params;
  const auto bid = blockIdx.x;
  const auto tid = threadIdx.x;
  const auto length = *(lengths + bid);
  const auto indice = indices + bid * TopK;
  const auto score = input + bid * max_seq_len;
  if (length <= TopK) {
    return naive_topk_cuda(score, indice, length);
  } else {
    return fast_topk_cuda(score, indice, length);
  }
}

auto fast_topk_interface(at::Tensor score, at::Tensor indices,
                         at::Tensor lengths) -> void {
  const auto B = score.size(0);
  const auto max_seq_len = score.size(1);
  TORCH_CHECK(score.dim() == 2);
  TORCH_CHECK(indices.dim() == 2);
  TORCH_CHECK(lengths.dim() == 1);
  TORCH_CHECK(indices.size(0) == B);
  TORCH_CHECK(indices.size(1) == TopK);
  TORCH_CHECK(lengths.size(0) == B);
  const auto params = FastTopKParams{
      score.data_ptr<float>(), indices.data_ptr<int32_t>(),
      lengths.data_ptr<int32_t>(), static_cast<size_t>(max_seq_len)};
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  constexpr auto smem = 32 * 1024 * sizeof(uint32_t); // 128KB
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  [[maybe_unused]]
  static const auto _ = [] {
    ::cudaFuncSetAttribute(topk_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    return 0;
  }();
  topk_kernel<<<grid, block, smem, stream>>>(params);
  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess,
              "topk kernel failed:", ::cudaGetErrorString(result));
}

} // namespace

PYBIND11_MODULE(topk_kernel, m) {
  m.def("fast_topk", &fast_topk_interface, "fast_topk");
}
