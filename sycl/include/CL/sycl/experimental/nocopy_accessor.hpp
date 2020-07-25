//==----- nocopy_accessor.hpp - SYCL public experimental API header file -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/accessor_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace experimental {

template <typename DataT, int Dimensions = 1, access::mode AccessMode = access::mode::read_write>
class nocopy_accessor
#ifndef __SYCL_DEVICE_ONLY__
  : public detail::AccessorBaseHost,
    public detail::accessor_common<DataT, Dimensions, AccessMode,
                                   access::target::host_buffer,
                                   access::placeholder::false_t>
#endif // !__SYCL_DEVICE_ONLY__
{
protected:
  template <typename T, int Dims> static constexpr bool IsSameAsBuffer() {
    return std::is_same<T, DataT>::value && (Dims > 0) && (Dims == Dimensions);
  }
#ifndef __SYCL_DEVICE_ONLY__
  using AccessorCommonT = detail::accessor_common<DataT, Dimensions, AccessMode,
                                                  access::target::host_buffer,
                                                  access::placeholder::false_t>;
  constexpr static int AdjustedDim = Dimensions == 0 ? 1 : Dimensions;
  using AccessorCommonT::AS;
  using PtrType = detail::const_if_const_AS<AS, DataT> *;

  static access::mode getAdjustedMode(const property_list &PropertyList) {
    access::mode AdjustedMode = AccessMode;
    if (PropertyList.has_property<property::noinit>()) {
      if (AdjustedMode == access::mode::write) {
        AdjustedMode = access::mode::discard_write;
      } else if (AdjustedMode == access::mode::read_write) {
        AdjustedMode = access::mode::discard_read_write;
      }
    }
    return AdjustedMode;
  }

  PtrType getQualifiedPtr() const {
    return reinterpret_cast<PtrType>(AccessorBaseHost::getPtr());
  }

  template <int Dims = Dimensions> size_t getLinearIndex(id<Dims> Id) const {
    size_t Result = 0;
    for (int I = 0; I < Dims; ++I)
      Result = Result * getMemoryRange()[I] + getOffset()[I] + Id[I];
    return Result;
  }
#endif // !__SYCL_DEVICE_ONLY__
public:
template <typename T = DataT, int Dims = Dimensions, typename AllocatorT,
            typename = detail::enable_if_t<IsSameAsBuffer<T, Dims>()>>
  nocopy_accessor(buffer<T, Dims, AllocatorT> &BufferRef, const property_list &PropertyList = {})
#ifdef __SYCL_DEVICE_ONLY__
  {
    throw feature_not_supported("nocopy_accessor() is not implemented on target device", PI_INVALID_OPERATION);
  }
#else // __SYCL_DEVICE_ONLY__
  : AccessorBaseHost(/*Offset=*/{0, 0, 0},
                     detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
                     detail::convertToArrayOfN<3, 1>(BufferRef.get_range()),
                     getAdjustedMode(PropertyList),
                     detail::getSyclObjImpl(BufferRef).get(), Dimensions, sizeof(DataT),
                     BufferRef.OffsetInBytes, BufferRef.IsSubBuffer)
  {
    detail::AccessorImplHost *impl = AccessorBaseHost::impl.get();
    impl->nocopy = true;
    addHostAccessorAndWait(impl);
  }
#endif // __SYCL_DEVICE_ONLY__


  DataT *get_pointer() const {
#ifdef __SYCL_DEVICE_ONLY__
    throw feature_not_supported("nocopy_accessor.get_pointer() is not implemented on target device", PI_INVALID_OPERATION);
#else // __SYCL_DEVICE_ONLY__
    const size_t LinearIndex = getLinearIndex(id<AdjustedDim>());
    return getQualifiedPtr() + LinearIndex;
#endif // __SYCL_DEVICE_ONLY__
  }
};

} // namespace experimental
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
