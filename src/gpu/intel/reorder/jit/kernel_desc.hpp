/*******************************************************************************
* Copyright 2026 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_INTEL_REORDER_JIT_KERNEL_DESC_HPP
#define GPU_INTEL_REORDER_JIT_KERNEL_DESC_HPP

#include "gpu/intel/jit/ir/kernel_desc.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/reorder/jit/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace reorder {
namespace jit {

// Describes a post-op-free reorder kernel, used as a kernel cache key.
class kernel_desc_t : public kernel_desc_base_t {
public:
    kernel_desc_t() = default;
    kernel_desc_t(const dsl::kernel::options_t &options, layout_t src,
            layout_t dst, std::string name);

    // A non-constant offset is baked into the kernel and cannot be serialized.
    static bool is_supported(const layout_t &src, const layout_t &dst) {
        return is_const(src.offset()) && is_const(dst.offset());
    }

    std::string kernel_name() const override { return name_; }
    dsl::kernel::options_t options(const impl::engine_t *engine) const override;
    compute::range_t local_range() const override { return local_range_; }
    void init_kernel_iface(dsl::kernel::iface_t &kernel_iface) const override;
    void init_kernel_info(kernel_info_t &kernel_info,
            const kernel_params_base_t &params,
            const impl::engine_t *engine) const override;
    status_t create_kernel(compute::kernel_t &kernel, primitive_t *primitive,
            impl::engine_t *engine) const override;
    status_t create_kernel(compute::kernel_t &kernel, primitive_t *primitive,
            impl::engine_t *engine, bool register_kernel) const;
    status_t create_generator(
            const intel::engine_t &engine, compute::kernel_t &kernel) const;
    serialization_stream_t serialize() const override;
    static kernel_desc_t deserialize(const serialization_stream_t &s);
#if __cplusplus >= 202002L
    bool operator==(const kernel_desc_t &) const = default;
#endif

private:
    int regs_ = 0;
    int simd_ = 0;
    bool dpas_ = false;
    layout_t src_;
    layout_t dst_;
    std::string name_;
    compute::range_t local_range_;
};

} // namespace jit
} // namespace reorder
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
