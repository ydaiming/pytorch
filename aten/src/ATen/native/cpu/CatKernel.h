#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using cat_contig_fn = void(*)(Tensor &, TensorList, int64_t);
DECLARE_DISPATCH(cat_contig_fn, cat_contig_stub);

}}  // namespace at::native
