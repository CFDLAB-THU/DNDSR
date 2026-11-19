#include "DNDS/Algebra/AmgXDiagnostics.hpp"
#include "nlohmann/json.hpp"

namespace DNDS::Algebra
{
    std::string AmgXMemoryReport(const AmgXMemorySnapshot &m)
    {
        using J = nlohmann::ordered_json;
        J levels = J::array();
        for (const auto &l : m.levels)
            levels.push_back({{"level", l.level}, {"block_rows", l.blockRows}, {"block_nonzeros", l.blockNonzeros}, {"block_size", l.blockSize}, {"numeric_owned", l.numericOwned}, {"numeric_borrowed", l.numericBorrowed}, {"pattern", l.pattern}, {"inverse", l.inverse}, {"transfer", l.transfer}, {"workspace", l.workspace}});
        J first;
        if (const auto &measured = m.copies.firstCorrection)
            first = J{{"rhs_max", measured->rhsMax}, {"correction_max", measured->correctionMax}, {"true_defect_max", measured->defectMax}};
        return J{{"first_correction", first}, {"native_storage", m.nativeStorage}, {"levels", levels}, {"copies", {{"pattern_h2d", m.copies.patternHostToDevice}, {"matrix_d2d", m.copies.matrixDeviceToDevice}, {"rhs_d2d", m.copies.rhsDeviceToDevice}, {"result_d2d", m.copies.resultDeviceToDevice}, {"setups", m.copies.setups}, {"applications", m.copies.applications}}}, {"numeric_refreshes", m.copies.numericRefreshes}, {"hierarchy_setup_seconds", m.copies.hierarchySetupSeconds}, {"numeric_refresh_seconds", m.copies.numericRefreshSeconds}, {"allocation_tracking", m.allocationTracking}, {"cuda_pool_available", m.cudaPoolAvailable}, {"requested_live", m.requestedLive}, {"requested_peak", m.requestedPeak}, {"live_allocations", m.liveAllocations}, {"device_pool_used", m.devicePoolUsed}, {"device_pool_reserved", m.devicePoolReserved}, {"pinned_pool_used", m.pinnedPoolUsed}, {"pinned_pool_reserved", m.pinnedPoolReserved}, {"cuda_default_pool_used", m.cudaPoolUsed}, {"cuda_default_pool_reserved", m.cudaPoolReserved}, {"cuda_default_pool_used_peak", m.cudaPoolUsedPeak}, {"cuda_default_pool_reserved_peak", m.cudaPoolReservedPeak}}.dump();
    }
}
