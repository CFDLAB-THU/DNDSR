#pragma once

#include "DNDS/Config/ConfigParam.hpp"
#include "DNDS/Core/CheckedSize.hpp"
#include "DNDS/Runtime/Array/RowLayout.hpp"

namespace DNDS::CFV
{
    /** @brief Immutable VR sweep representation with contiguous dense payloads.
     * @details Raw sweeps gather the original integrated B/mean/boundary terms
     * before applying the unchanged local SVD pseudoinverse. This changes
     * floating-point association, not the functional or singular cutoff.
     * Triangle packing is available only for the symmetric local inverse;
     * arbitrary cross blocks B are never assumed individually symmetric.
     * Every matrix, vector and injection row is contiguous. Experiments with
     * cross-row SoA/AoSoA matrix storage were slower and are retained in Git
     * history; this policy does not restrict the layouts of solution fields.
     */
    struct ReconstructionStorageSettings
    {
        std::string representation = "precomputed";
        bool inverseTriangle = false, faceCoupling = false;
        /// Device raw sweep storage. Host construction and sweep arithmetic remain FP64.
        std::string rawBPrecision = "fp64", inversePrecision = "fp64";
        /// AoS physical row capacity rounded in elements; logical width stays n*n.
        index aosRowPadding = 1;
        /// Base alignment is independent of row pitch; zero uses provider default.
        index allocationAlignmentBytes = 0;
        DNDS_DECLARE_CONFIG(ReconstructionStorageSettings)
        {
            DNDS_FIELD(representation, "Sweep representation: precomputed AInvB or raw B gathered before local inverse application");
            DNDS_FIELD(inverseTriangle, "Raw only: validated symmetric local inverse in row-packed lower triangle");
            DNDS_FIELD(faceCoupling, "Raw only: one full B per local mesh face in canonical parent orientation; reverse incidence transposes it");
            DNDS_FIELD(rawBPrecision, "Raw B device storage: fp64 or fp32. The FP64 pipelined sweep promotes the packed matrix before arithmetic");
            DNDS_FIELD(inversePrecision, "Raw local inverse device storage: fp64 or fp32. Independent of B precision and full/triangular packing");
            DNDS_FIELD(aosRowPadding, "Positive AoS physical-row pitch multiple in scalar elements, independent of logical width");
            DNDS_FIELD(allocationAlignmentBytes, "Optional power-of-two payload base alignment in bytes; zero uses provider default");
        }
        bool Raw() const { return representation == "raw"; }
        bool PackedRawB() const { return rawBPrecision == "fp32"; }
        bool PackedInverse() const { return inversePrecision == "fp32"; }
        void Validate() const
        {
            DNDS_check_throw_info(representation == "precomputed" || Raw(), "unknown VR sweep representation");
            DNDS_check_throw_info(Raw() || (!inverseTriangle && !faceCoupling), "triangle/face VR storage requires raw sweeps");
            DNDS_check_throw_info((rawBPrecision == "fp64" || PackedRawB()) &&
                                      (inversePrecision == "fp64" || PackedInverse()),
                                  "raw VR matrix storage precision must be fp64 or fp32");
            DNDS_check_throw_info(Raw() || (!PackedRawB() && !PackedInverse()),
                                  "independent matrix storage precision requires raw VR");
            DNDS_check_throw_info(aosRowPadding > 0, "invalid VR physical row padding");
            DNDS_check_throw_info(allocationAlignmentBytes >= 0 && (!allocationAlignmentBytes || !(allocationAlignmentBytes & (allocationAlignmentBytes - 1))),
                                  "VR allocation alignment must be zero or a power of two");
        }
        Runtime::AllocationOptions Allocation() const { return {size_t(allocationAlignmentBytes)}; }
        Runtime::RowLayout Layout(index rows, index width) const
        {
            DNDS_check_throw_info(rows >= 0 && width >= 0, "negative VR payload shape");
            DNDS_check_throw_info(aosRowPadding > 0, "invalid VR physical row padding");
            const index groups = width / aosRowPadding + (width % aosRowPadding != 0);
            const index stride = CheckedSize::Multiply(groups, aosRowPadding);
            return Runtime::RowLayout::Dense(size_t(rows), size_t(width), size_t(stride));
        }
    };
}
