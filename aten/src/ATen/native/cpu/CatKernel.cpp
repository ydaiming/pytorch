#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/cpu/CatKernel.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/Parallel.h>

namespace at { namespace native {

namespace {

template <typename scalar_t>
void cat_contig_kernel_impl(Tensor& result, TensorList tensors, int64_t dim) {
  auto size = result.sizes().vec();
  int64_t outer = result.numel() / (result.size(dim) * result.stride(dim));
  scalar_t* result_data = result.data_ptr<scalar_t>();
  int64_t output_stride = result.stride(dim) * result.size(dim);

  auto loop_nd = [&](int64_t start, int64_t end) {
    using Vec = vec256::Vec256<scalar_t>;
    for (auto i = start; i < end; ++i) {
      scalar_t* output_ptr = result.data_ptr<scalar_t>();
      output_ptr += i * output_stride;
      for (const auto& t : tensors) {
        int64_t local_inner = t.size(dim) * t.stride(dim);
        scalar_t* input_ptr = t.data_ptr<scalar_t>() + i * local_inner;
        std::memcpy(output_ptr, input_ptr, local_inner * sizeof(scalar_t));
        output_ptr += local_inner;
      }
    }
  };
  at::parallel_for(0, outer, at::internal::GRAIN_SIZE, loop_nd);
}

void cat_contig_kernel(Tensor& result, TensorList tensors, int64_t dim) {
  AT_DISPATCH_FLOATING_TYPES(result.scalar_type(), "cat_contig_kernel", [&]() {
    cat_contig_kernel_impl<scalar_t>(result, tensors, dim);
  });
}

} // anonymous namespace

REGISTER_DISPATCH(cat_contig_stub, &cat_contig_kernel);

}} // at::native
