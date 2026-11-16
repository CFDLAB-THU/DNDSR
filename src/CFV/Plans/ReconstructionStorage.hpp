#pragma once

#include "DNDS/Config/ConfigParam.hpp"
#include "DNDS/Core/CheckedSize.hpp"
#include "DNDS/Runtime/Array/RowLayout.hpp"

namespace DNDS::CFV
{
    /** @brief Independent immutable VR sweep representation and payload layouts.
     * @details Raw sweeps gather the original integrated B/mean/boundary terms
     * before applying the unchanged local SVD pseudoinverse. This changes
     * floating-point association, not the functional or singular cutoff.
     * Triangle packing is available only for the symmetric local inverse;
     * arbitrary cross blocks B are never assumed individually symmetric.
     */
    struct ReconstructionStorageSettings
    {
        std::string representation = "precomputed";
        std::string matrixLayout = "aos", inverseLayout = "aos";
        std::string vectorLayout = "aos", boundaryLayout = "aos";
        bool inverseTriangle = false, faceCoupling = false;
        index rowPadding = 32;
        /// AoS physical row capacity rounded in elements; logical width stays n*n.
        index aosRowPadding = 1;
        /// Base alignment is independent of row pitch; zero uses provider default.
        index allocationAlignmentBytes = 0;
        DNDS_DECLARE_CONFIG(ReconstructionStorageSettings)
        {
            DNDS_FIELD(representation, "Sweep representation: precomputed AInvB or raw B gathered before local inverse application");
            DNDS_FIELD(matrixLayout, "AInvB or raw B physical payload: aos, soa, aosoa32");
            DNDS_FIELD(inverseLayout, "Raw-sweep local inverse physical payload: aos, soa, aosoa32");
            DNDS_FIELD(vectorLayout, "Mean-jump map physical payload: aos, soa, aosoa32");
            DNDS_FIELD(boundaryLayout, "Boundary-injection map physical payload: aos, soa, aosoa32");
            DNDS_FIELD(inverseTriangle, "Raw only: validated symmetric local inverse in row-packed lower triangle");
            DNDS_FIELD(faceCoupling, "Raw only: one full B per local mesh face in canonical parent orientation; reverse incidence transposes it");
            DNDS_FIELD(rowPadding, "Positive row-padding multiple for SoA payload allocation; tile width stays 32");
            DNDS_FIELD(aosRowPadding, "Positive AoS physical-row pitch multiple in scalar elements, independent of logical width");
            DNDS_FIELD(allocationAlignmentBytes, "Optional power-of-two payload base alignment in bytes; zero uses provider default");
        }
        bool Raw() const { return representation == "raw"; }
        void Validate() const
        {
            DNDS_check_throw_info(representation == "precomputed" || Raw(), "unknown VR sweep representation");
            DNDS_check_throw_info(Raw() || (!inverseTriangle && !faceCoupling), "triangle/face VR storage requires raw sweeps");
            DNDS_check_throw_info(rowPadding > 0 && aosRowPadding > 0, "invalid VR physical row padding");
            DNDS_check_throw_info(allocationAlignmentBytes >= 0 && (!allocationAlignmentBytes || !(allocationAlignmentBytes & (allocationAlignmentBytes - 1))),
                                  "VR allocation alignment must be zero or a power of two");
            for (const auto *layout : {&matrixLayout, &inverseLayout, &vectorLayout, &boundaryLayout})
                DNDS_check_throw_info(*layout == "aos" || *layout == "soa" || *layout == "aosoa32", "unknown VR payload layout");
        }
        Runtime::AllocationOptions Allocation() const { return {size_t(allocationAlignmentBytes)}; }
        Runtime::RowLayout Layout(index rows, index width, const std::string &layout) const
        {
            DNDS_check_throw_info(rows >= 0 && width >= 0, "negative VR payload shape");
            if (layout == "aos")
            {
                const index groups = width / aosRowPadding + (width % aosRowPadding != 0);
                const index stride = CheckedSize::Multiply(groups, aosRowPadding);
                return Runtime::RowLayout::Dense(size_t(rows), size_t(width), size_t(stride));
            }
            if (layout == "soa")
                return Runtime::RowLayout::ColMajor(size_t(rows), size_t(width), {rowPadding, 0});
            DNDS_check_throw_info(layout == "aosoa32", "unknown VR payload layout");
            return Runtime::RowLayout::TiledColMajor(size_t(rows), size_t(width), 32);
        }
    };
}
