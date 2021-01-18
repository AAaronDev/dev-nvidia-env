 /* Copyright 1993-2016 NVIDIA Corporation.  All rights reserved.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to NVIDIA intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and
  * conditions of a form of NVIDIA software license agreement by and
  * between NVIDIA and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of NVIDIA is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  * OF THESE LICENSED DELIVERABLES.
  *
  * U.S. Government End Users.  These Licensed Deliverables are a
  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  * 1995), consisting of "commercial computer software" and "commercial
  * computer software documentation" as such terms are used in 48
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  * U.S. Government End Users acquire the Licensed Deliverables with
  * only those rights set forth herein.
  *
  * Any use of the Licensed Deliverables in individual and commercial
  * software must include, in the user documentation and internal
  * comments to the code, the above Disclaimer and U.S. Government End
  * Users Notice.
  */

#ifndef _CG_ASYNC_H
#define _CG_ASYNC_H

#include <cuda_pipeline.h>
#include "info.h"

_CG_BEGIN_NAMESPACE

namespace details {
    // Groups supported by memcpy_async
    template <class TyGroup> struct _async_copy_group_supported : public _CG_STL_NAMESPACE::false_type {};

    template <unsigned int Sz, typename TyPar>
    struct _async_copy_group_supported<cooperative_groups::thread_block_tile<Sz, TyPar>> : public _CG_STL_NAMESPACE::true_type {};
    template <> struct _async_copy_group_supported<cooperative_groups::coalesced_group>  : public _CG_STL_NAMESPACE::true_type {};
    template <> struct _async_copy_group_supported<cooperative_groups::thread_block>     : public _CG_STL_NAMESPACE::true_type {};

    template <class TyGroup>
    using async_copy_group_supported = _async_copy_group_supported<_CG_STL_NAMESPACE::remove_cv_t<TyGroup>>;

    template <unsigned int N>
    struct _Segment {
        int _seg[N];
    };

    // Trivial layout guaranteed-aligned copy-async compatible segments
    template <unsigned int N> struct Segment;
    template<> struct __align__(4)  Segment<1> : public _Segment<1> {};
    template<> struct __align__(8)  Segment<2> : public _Segment<2> {};
    template<> struct __align__(16) Segment<4> : public _Segment<4> {};

    template <typename TyElem, size_t TyAlignment = alignof(TyElem), bool BypassAlignment = (TyAlignment >= 4)>
    struct memcpy_async_dispatch {};

    // Interleaved element by element copies from source to dest
    template <typename TyGroup, typename TyElem>
    _CG_STATIC_QUALIFIER void inline_copy(TyGroup& group, TyElem* dst, const TyElem* src, size_t count) {
        unsigned int rank = group.thread_rank();
        unsigned int stride = group.size();

        src += rank;
        dst += rank;
        for (size_t idx = rank; idx < count; idx += stride) {
            *dst = *src;
            src += stride;
            dst += stride;
        }
    }

    template <typename TyGroup, typename TyElem>
    _CG_STATIC_QUALIFIER size_t accelerated_async_copy(TyGroup& group, TyElem* dst, const TyElem* src, size_t count, nvcuda::experimental::pipeline& pipe) {
        static_assert(async_copy_group_supported<TyGroup>::value, 
            "Async copy is only supported for groups that represent private private shared memory");
            
        unsigned int stride = group.size();
        unsigned int rank = group.thread_rank();

        src += rank;
        dst += rank;
        for (unsigned int idx = rank; idx < count; idx += stride) {
            nvcuda::experimental::memcpy_async(*dst, *src, pipe);
            src += stride;
            dst += stride;
        }

        // Return the given count
        return count;
    }

    // Determine best possible alignment given an input and initial conditions
    // Attempts to generate as little code as possible, most likely should only be used with 1 and 2 byte alignments
    template <unsigned int MaxAlignment, typename TyElem>
    _CG_STATIC_QUALIFIER uint32_t find_best_alignment(TyElem* dst, const TyElem* src) {
        constexpr uint32_t alignmentBypass = alignof(TyElem);
        // Narrowing conversion intentional
        uint32_t base1 = (uint32_t)reinterpret_cast<uintptr_t>(src);
        uint32_t base2 = (uint32_t)reinterpret_cast<uintptr_t>(dst);

        // 0b1010 ^ 0b1001 == 0b0001
        uint32_t diff = ((base1) ^ (base2)) & (MaxAlignment-1);

        // range [MaxAlignment, alignof(elem)], step: x >> 1
        // over range of possible alignments, choose best available out of range
        uint32_t out = MaxAlignment;
        #pragma unroll
        for (uint32_t alignment = (MaxAlignment >> 1); alignment >= alignmentBypass; alignment >>= 1) {
            if (alignment & diff)
                out = alignment;
        }

        return out;
    }

    // Determine best possible alignment given an input and initial conditions
    // Attempts to generate as little code as possible, most likely should only be used with 1 and 2 byte alignments
    template <typename TyType, typename TyGroup, typename TyElem>
    _CG_STATIC_QUALIFIER size_t copy_like(const TyGroup &group, TyElem* dst, const TyElem* src, size_t count, nvcuda::experimental::pipeline &pipe) {
        constexpr size_t targetAlignment = alignof(TyType);
        constexpr size_t sourceAlignment = alignof(TyElem);
        constexpr uint32_t alignmentRatio = targetAlignment / sourceAlignment;

        uintptr_t base = (uintptr_t)reinterpret_cast<uintptr_t>(src);
        uint32_t alignOffset = ((~base) + 1) & (targetAlignment - 1);
        size_t init = count;

        inline_copy(group, dst, src, alignOffset);
        count -= alignOffset;
        src   += alignOffset;
        dst   += alignOffset;

        // Copy using the best available alignment 
        size_t asyncCount = accelerated_async_copy(group,
            reinterpret_cast<TyType*>(dst), reinterpret_cast<const TyType*>(src), 
            count / alignmentRatio, pipe);
        asyncCount *= alignmentRatio;
        count -= asyncCount;
        src   += asyncCount;
        dst   += asyncCount;

        inline_copy(group, dst, src, count);

        // this should always return the number of elements given, could be used as a debug check
        return init;
    }

    // Manually dispatching to proper alignment is required for 1 and 2 byte inputs
    template <typename TyElem, size_t TyAlignment>
    struct memcpy_async_dispatch<TyElem, TyAlignment, false> {
        template <typename TyGroup>
        _CG_STATIC_QUALIFIER size_t copy(
            TyGroup& group, TyElem* dst, const TyElem* src,
            size_t copyCount, nvcuda::experimental::pipeline& pipe) {
            // Dispatch to an inline copy for unsupported alignments
            uint32_t alignment = find_best_alignment<16>(dst, src);

            switch(alignment) {
                case 1:
                case 2:
                    inline_copy(group, dst, src, copyCount);
                    return copyCount;
                case 4:
                    return copy_like<Segment<1>>(group, dst, src, copyCount, pipe);
                case 8:
                    return copy_like<Segment<2>>(group, dst, src, copyCount, pipe);
                case 16:
                    return copy_like<Segment<4>>(group, dst, src, copyCount, pipe);
            }
            return copyCount;
        }
    };

    // Specialization for 4 byte alignments
    template <typename TyElem>
    struct memcpy_async_dispatch<TyElem, 4, true> {
        template <typename TyGroup>
        _CG_STATIC_QUALIFIER size_t copy(
            TyGroup& group, TyElem* dst, const TyElem* src,
            size_t copyCount, nvcuda::experimental::pipeline& pipe) {
            // Dispatch straight to aligned LDGSTS calls
            return accelerated_async_copy(group, dst, src, copyCount, pipe);
        }
    };

    // Specialization for 8 byte alignments
    template <typename TyElem>
    struct memcpy_async_dispatch<TyElem, 8, true> {
        template <typename TyGroup>
        _CG_STATIC_QUALIFIER size_t copy(
            TyGroup& group, TyElem* dst, const TyElem* src,
            size_t copyCount, nvcuda::experimental::pipeline& pipe) {
            // Dispatch straight to aligned LDGSTS calls
            return accelerated_async_copy(group, dst, src, copyCount, pipe);
        }
    };

    // Alignments over 16 are truncated to 16 and bypass alignment
    // This is the highest performing memcpy available
    template <typename TyElem>
    struct memcpy_async_dispatch<TyElem, 16, true> {
        template <typename TyGroup>
        _CG_STATIC_QUALIFIER size_t copy(
            TyGroup& group, TyElem* dst, const TyElem* src,
            size_t copyCount, nvcuda::experimental::pipeline& pipe) {
            // Dispatch straight to aligned LDGSTS calls
            return accelerated_async_copy(group, dst, src, copyCount, pipe);
        }
    };

    // Truncate internally to 16 byte alignment and dispatch again
    template <typename TyElem, size_t TyAlignment>
    struct memcpy_async_dispatch<TyElem, TyAlignment, true> {
        template <typename TyGroup>
        _CG_STATIC_QUALIFIER size_t copy(
            TyGroup& group, TyElem* dst, const TyElem* src,
            size_t copyCount, nvcuda::experimental::pipeline& pipe) {
            const Segment<4> *src16 = reinterpret_cast<const Segment<4> *>(src);
            Segment<4> *dst16 = reinterpret_cast<Segment<4> *>(dst);

            // Update copy count to reflect the change in element size
            // e.g. count * (32 / 16)
            copyCount *= (TyAlignment / 16);

            return memcpy_async_dispatch<Segment<4>>::copy(group, dst16, src16, copyCount, pipe);
        }
    };
}

/* Group submit batch of async-copy to cover of contiguous 1D array
   to a pipeline, commit the batch, and return pipeline stage. */
template <class TyGroup, class TyElem>
_CG_STATIC_QUALIFIER size_t memcpy_async(
        TyGroup& group, TyElem* dst, size_t dstCount,
        const TyElem* src, size_t srcCount, nvcuda::experimental::pipeline& pipe) {
    size_t count = details::memcpy_async_dispatch<TyElem>::copy(group, dst, src, min(dstCount, srcCount), pipe);
    pipe.commit();
    return count;
}

/* Group wait for prior Nth stage of memcpy_async to complete. */
template <unsigned int Stage, class TyGroup>
_CG_STATIC_QUALIFIER void wait_prior(TyGroup& group, nvcuda::experimental::pipeline& pipe) {
    pipe.wait_prior<Stage>();
    group.sync();
}

/* Group wait for stage-S of memcpy_async to complete. */
template <class TyGroup>
_CG_STATIC_QUALIFIER void wait(TyGroup& group, nvcuda::experimental::pipeline& pipe, size_t stage) {
    pipe.wait(stage);
    group.sync();
}

/* Group submit batch of async-copy to cover contiguous 1D array
and commit that batch to eventually wait for completion. */
template <class TyGroup, class TyElem>
_CG_STATIC_QUALIFIER size_t memcpy_async(TyGroup& group, TyElem* dst, size_t dstCount, const TyElem* src, size_t srcCount) {
    // Fake pipe for non-pipelined copies
    nvcuda::experimental::pipeline pipe;
    size_t count = memcpy_async(group, dst, dstCount, src, srcCount, pipe);
    return count;
}

/* Group wait for prior Nth stage of memcpy_async to complete. */
template <unsigned int Stage, class TyGroup>
_CG_STATIC_QUALIFIER void wait_prior(TyGroup& group) {
    nvcuda::experimental::pipeline pipe;
    wait_prior<Stage>(group, pipe);
}

/* Group wait all previously submitted memcpy_async to complete. */
template <class TyGroup>
_CG_STATIC_QUALIFIER void wait(TyGroup& group) {
    // Fake pipe for non-pipelined copies
    nvcuda::experimental::pipeline pipe;
    pipe.wait_prior<0>();
    group.sync();
}

_CG_END_NAMESPACE

#endif // _CG_ASYNC_H
