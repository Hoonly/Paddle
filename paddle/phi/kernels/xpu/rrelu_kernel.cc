// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/rrelu_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void RReluKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const float lower,
                 const float upper,
                 bool is_test,
                 DenseTensor* out ,
                 DenseTensor* noise) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const T* x_ptr = x.data<T>();
  T* o_ptr = dev_ctx.template Alloc<T>(out);
  //   T* n_ptr = dev_ctx.template Alloc<T>(noise); //   负斜率值，在训练场景为生成的随机数，测试场景为mid_val固定值

  PD_CHECK(is_test);  // 仅支持推理场景

#ifndef PADDLE_WITH_XPU_PLUGIN
  PADDLE_THROW(errors::Unavailable("rrelu-xpu only support in xpu plugin, add -DWITH_XPU_PLUGIN=ON to build."));
#else
  int r = xpu::plugin::rrelu(dev_ctx.x_context(),
                             reinterpret_cast<const XPUType*>(x.data<T>()),
                             reinterpret_cast<XPUType*>(out->data<T>()),
                             static_cast<int>(x.numel()),
                             lower,
                             upper);

#endif

}

}  // namespace phi

PD_REGISTER_KERNEL(rrelu,
                   XPU,
                   ALL_LAYOUT,
                   phi::RReluKernel,
                   float,
                   phi::dtype::float16) {}
