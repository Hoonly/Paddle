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

#include "paddle/phi/kernels/conv_transpose_kernel.h"

#include "glog/logging.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {

// target_len == 2 || target_len == 4
inline std::vector<int> vector_extend(const std::vector<int>& src,
                                      int target_len) {
  if (target_len == 2 && src.size() == 1) {
    return {src[0], src[0]};
  }
  if (target_len == 4 && src.size() == 1) {
    return {src[0], src[0], src[0], src[0]};
  }
  if (target_len == 4 && src.size() == 2) {
    return {src[0], src[0], src[1], src[1]};
  }
  return src;
}

template <typename T, typename Context>
void Conv2dTransposeKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& output_padding,
                           const IntArray& output_size,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format,
                           DenseTensor* out) {
  using XPUT = typename XPUTypeTrait<T>::Type;

  ctx.template Alloc<T>(out);

  PADDLE_ENFORCE_EQ(
      data_format == "NHWC" || data_format == "NDHWC",
      false,
      errors::InvalidArgument(
          ("XPU do support data_format is NCHW in conv_transpose op.")));

  DDim in_data_dims = slice_ddim(x.dims(), 2, x.dims().size());
  DDim filter_data_dims = slice_ddim(filter.dims(), 2, filter.dims().size());
  std::vector<int> ksize = vectorize<int>(filter_data_dims);

  std::vector<int> paddings_ = paddings;
  std::vector<int> dilations_ = dilations;
  UpdatePaddingAndDilation(
      &paddings_, &dilations_, padding_algorithm, in_data_dims, strides, ksize);

  const int batch_size = static_cast<int>(x.dims()[0]);
  const int img_yc = static_cast<int>(x.dims()[1]);
  const int img_xc = static_cast<int>(out->dims()[1]);
  const int img_xh = static_cast<int>(out->dims()[2]);
  const int img_xw = static_cast<int>(out->dims()[3]);

  int fccal_type = FCCalcType<XPUT>();
  if (fccal_type == XPUFCCalcType::FC_INT32) {
    int r = xpu::conv2d_transpose_v2<float, float, float, int32_t>(
        ctx.x_context(),
        x.data<float>(),
        filter.data<float>(),
        out->data<float>(),
        batch_size,
        img_yc,
        img_xh,
        img_xw,
        img_xc,
        ksize,
        strides,
        paddings_,
        dilations_,
        groups,
        nullptr,
        nullptr,
        nullptr,
        true);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose_v2");
  } else if (fccal_type == XPUFCCalcType::FC_FLOAT) {
    int r = xpu::conv2d_transpose_v2<float, float, float, float>(
        ctx.x_context(),
        x.data<float>(),
        filter.data<float>(),
        out->data<float>(),
        batch_size,
        img_yc,
        img_xh,
        img_xw,
        img_xc,
        ksize,
        strides,
        paddings_,
        dilations_,
        groups,
        nullptr,
        nullptr,
        nullptr,
        true);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose_v2");
  } else if (fccal_type == XPUFCCalcType::FC_INT32_WITH_LL) {
    if (output_size.size()) {
      VLOG(4) << "int_with_ll quantization is not supported when output_size "
                 "is specified, "
              << "use int31 instead";
      int r = xpu::conv2d_transpose_v2<float, float, float, int32_t>(
          ctx.x_context(),
          x.data<float>(),
          filter.data<float>(),
          out->data<float>(),
          batch_size,
          img_yc,
          img_xh,
          img_xw,
          img_xc,
          ksize,
          strides,
          paddings_,
          dilations_,
          groups,
          nullptr,
          nullptr,
          nullptr,
          true);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose_v2");
    } else {
      // xpu::conv2d_transpose_v2 do not support int_with_ll now
      // use xpu::conv2d_transpose
      int img_yh = static_cast<int>(x.dims()[2]);
      int img_yw = static_cast<int>(x.dims()[3]);
      int r = xpu::conv2d_transpose<float, float, float, int_with_ll_t>(
          ctx.x_context(),
          x.data<float>(),
          filter.data<float>(),
          out->data<float>(),
          batch_size,
          img_yc,
          img_yh,
          img_yw,
          img_xc,
          ksize,
          strides,
          paddings_,
          dilations_,
          groups,
          nullptr,
          nullptr,
          nullptr,
          true);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose");
    }
  } else {
    int r = xpu::conv2d_transpose_v2<XPUT, XPUT, XPUT, int16_t>(
        ctx.x_context(),
        reinterpret_cast<const XPUT*>(x.data<T>()),
        reinterpret_cast<const XPUT*>(filter.data<T>()),
        reinterpret_cast<XPUT*>(out->data<T>()),
        batch_size,
        img_yc,
        img_xh,
        img_xw,
        img_xc,
        ksize,
        strides,
        paddings_,
        dilations_,
        groups,
        nullptr,
        nullptr,
        nullptr,
        true);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv2d_transpose_v2");
  }
}
template <typename T, typename Context>
void DepthwiseConv2dTransposeKernel(const Context& ctx,
                                    const DenseTensor& x,
                                    const DenseTensor& filter,
                                    const std::vector<int>& strides,
                                    const std::vector<int>& paddings,
                                    const std::vector<int>& output_padding,
                                    const IntArray& output_size,
                                    const std::string& padding_algorithm,
                                    int groups,
                                    const std::vector<int>& dilations,
                                    const std::string& data_format,
                                    DenseTensor* out) {
  Conv2dTransposeKernel<T, Context>(ctx,
                                    x,
                                    filter,
                                    strides,
                                    paddings,
                                    output_padding,
                                    output_size,
                                    padding_algorithm,
                                    groups,
                                    dilations,
                                    data_format,
                                    out);
}

template <typename T, typename Context>
void Conv3dTransposeKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& output_padding,
                           const std::vector<int>& output_size,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format,
                           DenseTensor* out) {
  using XPUT = typename XPUTypeTrait<T>::Type;
  int fccal_type = FCCalcType<XPUT>();
  if (fccal_type != XPUFCCalcType::FC_INT16) {
    LOG(FATAL) << "Conv3dTransposeKernel only support FC_INT16";
  }

  std::string normed_data_format = data_format;
  if (data_format == "NCHW")
    normed_data_format = "NCDHW";
  else if (data_format == "NHWC")
    normed_data_format = "NDHWC";

  ctx.template Alloc<T>(out);

  PADDLE_ENFORCE_EQ(
      normed_data_format == "NDHWC" || normed_data_format == "NCDHW",
      true,
      errors::InvalidArgument(
          ("XPU conv3d_transpose supported data_format is NDHWC or NCDHW, but received [%s]", normed_data_format)));
  
  DDim in_data_dims = slice_ddim(x.dims(), 2, x.dims().size());
  DDim filter_data_dims = slice_ddim(filter.dims(), 2, filter.dims().size());
  std::vector<int> ksize = vectorize<int>(filter_data_dims);

  std::vector<int> paddings_ = paddings;
  std::vector<int> dilations_ = dilations;
  UpdatePaddingAndDilation(
      &paddings_, &dilations_, padding_algorithm, in_data_dims, strides, ksize);

  // 根据layout计算不同参数传入，
  // 默认laout是NCDHW, kernel对应OIDHW
  // layour NDHWC, kernel对应DHWOI
  int n = static_cast<int>(x.dims()[0]);
  int yc = static_cast<int>(x.dims()[1]);
  int yd = static_cast<int>(x.dims()[2]);
  int yh = static_cast<int>(x.dims()[3]);
  int yw = static_cast<int>(x.dims()[4]);
  int xc = static_cast<int>(out->dims()[1]);
  if(normed_data_format == "NDHWC") {
    yc = static_cast<int>(x.dims()[4]);
    yd = static_cast<int>(x.dims()[1]);
    yh = static_cast<int>(x.dims()[2]);
    yw = static_cast<int>(x.dims()[3]);
    xc = static_cast<int>(out->dims()[4]);
  }

  // support dtype
  // (float, float, float, int16_t);
  // (float, float, float, int);
  // (float, float, float, tfloat32);
  // (float, float, float, int8_t);
  // (float16, float16, float16, int16_t);
  // (float16, float16, float16, int8_t);
  // 当前实现逻辑，仅对(float, float, float, int16_t)和(float16, float16, float16, int16_t)进行支持
  int r = xpu::conv3d_transpose<XPUT, XPUT, XPUT, int16_t>(
      ctx.x_context(),
      reinterpret_cast<const XPUT*>(x.data<T>()),
      reinterpret_cast<const XPUT*>(filter.data<T>()),
      reinterpret_cast<XPUT*>(out->data<T>()),
      n,
      yc,
      yd,
      yh,
      yw,
      xc,
      ksize,
      strides,
      paddings_,
      dilations_,
      groups,
      nullptr,
      nullptr,
      nullptr,
      normed_data_format == "NDHWC");
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "conv3d_transpose");
}

}  // namespace phi
PD_REGISTER_KERNEL(depthwise_conv2d_transpose,
                   XPU,
                   ALL_LAYOUT,
                   phi::DepthwiseConv2dTransposeKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(conv2d_transpose,
                   XPU,
                   ALL_LAYOUT,
                   phi::Conv2dTransposeKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(conv3d_transpose,
                   XPU,
                   ALL_LAYOUT,
                   phi::Conv3dTransposeKernel,
                   float, // 支持的数据类型1
                   phi::dtype::float16) // 支持的数据类型2
                   {}
