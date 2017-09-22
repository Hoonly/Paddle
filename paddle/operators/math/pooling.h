/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/framework/eigen.h"
#include "paddle/framework/tensor.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {

//////////////////////
#ifdef __NVCC__
#define HL_DEVICE __device__
#else
#define HL_DEVICE
#endif
#define FLT_MAX __FLT_MAX__
/////////////////////

namespace pool {
template <class T>
class maxPool {
 public:
  HL_DEVICE inline T initial() { return -(T)(FLT_MAX); }
  HL_DEVICE inline void process(T& y, const T& x) { y = y > x ? y : x; }
  HL_DEVICE inline void finalize(T& y, const T& poo_size) {}
  HL_DEVICE inline void gradProcess(const T& x, const T& y, const T& dy, T& dx,
                                    T scale) {
    dx += dy * (x == y);
  }
};

template <class T>
class avePool {
 public:
  HL_DEVICE inline T initial() { return 0; }
  HL_DEVICE inline void process(T& y, const T& x) { y += x; }
  HL_DEVICE inline void finalize(T& y, const T& poo_size) { y /= poo_size; }
  HL_DEVICE inline void gradProcess(const T& x, const T& y, const T& dy, T& dx,
                                    T scale) {
    dx += (scale * dy);
  }
};
}  // namespace pool

template <typename Place, typename PoolProcess, typename T>
class Pool2dForwardFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& output,
                  std::vector<int>& ksize, std::vector<int>& strides,
                  std::vector<int>& paddings, PoolProcess pool_process);
};

template <typename Place, typename PoolProcess, typename T>
class Pool2dBackwardFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings,
                  PoolProcess pool_process);
};

template <typename Place, typename PoolProcess, typename T>
class Pool3dForwardFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& output,
                  std::vector<int>& ksize, std::vector<int>& strides,
                  std::vector<int>& paddings, PoolProcess pool_process);
};

template <typename Place, typename PoolProcess, typename T>
class Pool3dBackwardFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings,
                  PoolProcess pool_process);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
