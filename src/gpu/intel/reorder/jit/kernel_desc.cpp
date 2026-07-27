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

#include "gpu/intel/reorder/jit/kernel_desc.hpp"

#include "gpu/intel/jit/ir/hw.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/reorder/jit/kernel.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace reorder {
namespace jit {

namespace {

void append(serialization_stream_t &s, const layout_t &layout) {
    s.append(layout.type().str());
    s.append(layout.ndims());
    s.append(layout.offset<int64_t>());
    s.append(layout.blocks());
}

layout_t pop_layout(deserializer_t &d) {
    std::string type_str;
    size_t ndims;
    int64_t offset;
    std::vector<layout_t::block_t> blocks;
    d.pop(type_str);
    d.pop(ndims);
    d.pop(offset);
    d.pop(blocks);
    dsl::type_t type;
    istringstream_t iss(type_str);
    type.parse(iss);
    return layout_t(type, blocks, offset, ndims, /*do_normalize=*/false);
}

} // namespace

kernel_desc_t::kernel_desc_t(const dsl::kernel::options_t &options,
        layout_t src, layout_t dst, std::string name)
    : regs_(options.regs())
    , simd_(options.simd())
    , dpas_(options.require_dpas())
    , src_(std::move(src))
    , dst_(std::move(dst))
    , name_(std::move(name)) {
    config_t cfg(options, src_, dst_);
    local_range_ = cfg.nd_range().local_range();
}

dsl::kernel::options_t kernel_desc_t::options(
        const impl::engine_t *engine) const {
    auto ret = dsl::kernel::options_t(make_ir_hw(engine), regs_, simd_);
    ret.set_require_dpas(dpas_);
    return ret;
}

void kernel_desc_t::init_kernel_iface(
        dsl::kernel::iface_t &kernel_iface) const {
    kernel_iface.register_arg("src", dsl::type_t::byte(dsl::type::attr_t::ptr));
    kernel_iface.register_arg("dst", dsl::type_t::byte(dsl::type::attr_t::ptr));
}

void kernel_desc_t::init_kernel_info(kernel_info_t &kernel_info,
        const kernel_params_base_t &params,
        const impl::engine_t *engine) const {}

status_t kernel_desc_t::create_kernel(compute::kernel_t &kernel,
        primitive_t *primitive, impl::engine_t *engine) const {
    return create_kernel(kernel, primitive, engine, /*register_kernel=*/true);
}

status_t kernel_desc_t::create_kernel(compute::kernel_t &kernel,
        primitive_t *primitive, impl::engine_t *engine,
        bool register_kernel) const {
    return primitive->create_kernel(
            engine, kernel, kernel_name().c_str(), *this, register_kernel);
}

status_t kernel_desc_t::create_generator(
        const intel::engine_t &engine, compute::kernel_t &kernel) const {
    config_t cfg(options(&engine), src_, dst_);
    dsl::kernel::iface_t iface(name_);
    init_kernel_iface(iface);
    kernel_t jit_kernel(cfg, iface, local_range_);
    return engine.create_kernel(&kernel, &jit_kernel);
}

serialization_stream_t kernel_desc_t::serialize() const {
    serialization_stream_t s(regs_, simd_, dpas_, name_, local_range_);
    append(s, src_);
    append(s, dst_);
    return s;
}

kernel_desc_t kernel_desc_t::deserialize(const serialization_stream_t &s) {
    deserializer_t d(s);
    kernel_desc_t desc;
    d.pop(desc.regs_);
    d.pop(desc.simd_);
    d.pop(desc.dpas_);
    d.pop(desc.name_);
    d.pop(desc.local_range_);
    desc.src_ = pop_layout(d);
    desc.dst_ = pop_layout(d);
    return desc;
}

} // namespace jit
} // namespace reorder
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
