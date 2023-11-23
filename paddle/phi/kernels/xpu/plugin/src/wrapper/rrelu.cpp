// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
/*
 * copyright (C) 2023 KUNLUNXIN, Inc
 */

#include "xpu/plugin.h"
#include "xpu/refactor/impl_public/wrapper_check.h"

namespace xpu2 {
namespace plugin {
template <typename T>
__attribute__((global)) void rrelu(const T* x, T* y, int len, const float lower, const float upper);
}
}  // namespace xpu2

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

template <typename T>
static int cpu_wrapper(Context* ctx, const T* x, T* y, int len, float lower, float upper) {
  T mid_val = static_cast<T>((lower + upper) / 2.0);
  T zero = static_cast<T>(0);
  for (int i = 0; i < len; i++) {
    if (x[i] < zero) {
      y[i] = mid_val * x[i];
    } else {
      y[i] = x[i];
    }
  }
  return SUCCESS;
}

template <typename T>
static int xpu2_wrapper(Context* ctx, const T* x, T* y, int len, float lower, float upper) {
  ctx_guard RAII_GUARD(ctx);
  xpu2::plugin::rrelu<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(x, y, len, lower, upper);
  return api::SUCCESS;
}

template <typename T>
int rrelu(Context* ctx, const T* x, T* y, int len, float lower, float upper) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "rrelu", T);
  WRAPPER_DUMP_PARAM5(ctx, x, y, len, lower, upper);
  WRAPPER_DUMP(ctx);
  WRAPPER_ASSERT_GT(ctx, len, 0);
  WRAPPER_CHECK_2PTRS(ctx, T, len, x, y);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx, x, y, len, lower, upper);
  }
  if (ctx->dev().type() == api::kXPU2) {
    return xpu2_wrapper(ctx, x, y, len, lower, upper);
  }
  return NOT_IMPLEMENT;
}

template int rrelu(Context*, const float*, float*, int, float, float);

template int rrelu(Context*, const float16*, float16*, int, float, float);

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
