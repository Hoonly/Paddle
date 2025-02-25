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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"

namespace phi {
namespace fusion {

template <typename T_X,
          typename T_W,
          typename T_OUT,
          typename T_GEMM,
          typename Context>
void Conv3dXPUKernelImpl(const Context& ctx,
                         const DenseTensor& x,
                         const paddle::optional<DenseTensor>& x_max,
                         const DenseTensor& filter,
                         const DenseTensor& filter_max,
                         const paddle::optional<DenseTensor>& bias,
                         const paddle::optional<DenseTensor>& branch,
                         const paddle::optional<DenseTensor>& branch_max,
                         const std::vector<int>& paddings,
                         const std::vector<int>& dilations,
                         const std::vector<int>& strides,
                         const std::string& padding_algorithm,
                         int groups,
                         const std::string& data_format, // add data_format, default to 'NCDHW', option: 'NCDHW/OIDHW' or 'NDHWC/DHWOI'
                         int act_type,
                         float act_param,
                         DenseTensor* out,
                         DenseTensor* out_max) {
  using XPUTypeX = typename XPUTypeTrait<T_X>::Type;
  using XPUTypeW = typename XPUTypeTrait<T_W>::Type;
  using XPUTypeOut = typename XPUTypeTrait<T_OUT>::Type;
  auto input_dims = x.dims();
  auto filter_dims = filter.dims();
  // update paddings and dilations accoring to padding_algorithm
  std::vector<int> paddings_vec = paddings;
  std::vector<int> dilations_vec = dilations;
  DDim in_data_dims = phi::slice_ddim(input_dims, 2, input_dims.size());
  DDim filter_data_dims = phi::slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  phi::UpdatePaddingAndDilation(&paddings_vec,
                                &dilations_vec,
                                padding_algorithm,
                                in_data_dims,
                                strides,
                                ksize);

  // default is NCDHW
  int n = static_cast<int>(input_dims[0]);
  int c = static_cast<int>(input_dims[1]);
  int d = static_cast<int>(input_dims[2]);
  int h = static_cast<int>(input_dims[3]);
  int w = static_cast<int>(input_dims[4]);
  int out_c = static_cast<int>(out->dims()[1]);
  if(data_format == "NDHWC") {
    c = static_cast<int>(input_dims[4]);
    d = static_cast<int>(input_dims[1]);
    h = static_cast<int>(input_dims[2]);
    w = static_cast<int>(input_dims[3]);
    out_c = static_cast<int>(out->dims()[4]);
  }

  auto* input_data = reinterpret_cast<const XPUTypeX*>(x.data<T_X>());
  const float* input_max_data =
      x_max.get_ptr() == nullptr ? nullptr : x_max.get_ptr()->data<float>();
  auto* filter_data = reinterpret_cast<const XPUTypeW*>(filter.data<T_W>());
  auto* filter_max_data = filter_max.data<float>();

  const XPUTypeOut* branch_data = nullptr;
  auto* branch_tensor = branch.get_ptr();
  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  if (branch_tensor != nullptr) {
    if (branch_tensor->dtype() == out->dtype()) {
      branch_data =
          reinterpret_cast<const XPUTypeOut*>(branch_tensor->data<T_OUT>());
    } else {
      auto branch_data_temp =
          RAII_GUARD.alloc_l3_or_gm<XPUTypeOut>(branch_tensor->numel());
      int r = xpu::cast<XPUTypeX, XPUTypeOut>(
          ctx.x_context(),
          reinterpret_cast<const XPUTypeX*>(branch_tensor->data<T_X>()),
          branch_data_temp,
          branch_tensor->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
      branch_data = branch_data_temp;
    }
  }
  const float* branch_max_data = branch_max.get_ptr() == nullptr
                                     ? nullptr
                                     : branch_max.get_ptr()->data<float>();
  const float* bias_data =
      bias.get_ptr() == nullptr ? nullptr : bias.get_ptr()->data<float>();
  auto* out_data =
      reinterpret_cast<XPUTypeOut*>(ctx.template Alloc<T_OUT>(out));
  auto* out_max_data = ctx.template Alloc<float>(out_max);
  xpu::Activation_t act(static_cast<xpu::Activation_t::act_enum>(act_type));
  if (act_type == xpu::Activation_t::LEAKY_RELU) {
    act.leaky_alpha = act_param;
  } else if (act_type == xpu::Activation_t::HARD_SIGMOID) {
    act.hard_sigmoid_slope = act_param;
  }

  int r = xpu::
      conv3d_fusion<XPUTypeX, XPUTypeW, XPUTypeOut, T_GEMM>(  // TX/TW/TY/TGEMM
          ctx.x_context(),
          input_data,
          filter_data,
          out_data,
          n,
          c,
          d,
          h,
          w,
          out_c,
          ksize,
          strides,
          paddings_vec,
          dilations_vec,
          groups,
          input_max_data,
          filter_max_data,
          out_max_data,
          data_format=="NCDHW", // is_ncdhw
          bias_data,
          branch_data,
          act,
          branch_max_data);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv3d_xpu");
}

#define CONV3D_XPU_KERNEL_IMPL(x_dtype_, w_dtype_, out_dtype_, gemm_dtype_)  \
  Conv3dXPUKernelImpl<x_dtype_, w_dtype_, out_dtype_, gemm_dtype_, Context>( \
      ctx,                                                                   \
      x,                                                                     \
      x_max,                                                                 \
      filter,                                                                \
      filter_max,                                                            \
      bias,                                                                  \
      branch,                                                                \
      branch_max,                                                            \
      paddings,                                                              \
      dilations,                                                             \
      strides,                                                               \
      padding_algorithm,                                                     \
      groups,                                                                \
      data_format,                                                           \
      act_type,                                                              \
      act_param,                                                             \
      out,                                                                   \
      out_max);

template <typename T, typename Context>
void Conv3dXPUKernel(const Context& ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& x_max,
                     const DenseTensor& filter,
                     const DenseTensor& filter_max,
                     const paddle::optional<DenseTensor>& bias,
                     const paddle::optional<DenseTensor>& branch,
                     const paddle::optional<DenseTensor>& branch_max,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     const std::vector<int>& strides,
                     const std::string& padding_algorithm,
                     int groups,
                     const std::string& data_format,
                     int act_type,
                     float act_param,
                     DataType out_dtype,
                     DenseTensor* out,
                     DenseTensor* out_max) {
  if (out_dtype == DataType::FLOAT32) {
    CONV3D_XPU_KERNEL_IMPL(T, int16_t, float, int16_t);
  } else if (out_dtype == DataType::FLOAT16) {
    CONV3D_XPU_KERNEL_IMPL(T, int16_t, dtype::float16, int16_t);
  } else {
    PADDLE_THROW(phi::errors::Unimplemented("Not support out_dtype is %s.",
                                            DataTypeToString(out_dtype)));
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(conv3d_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::Conv3dXPUKernel,
                   float,
                   phi::dtype::float16) {}
