#pragma once

#include "DNDS/Runtime/Communication/RowTransfer.hpp"

namespace DNDS::Runtime::Detail
{
    /// Receive one logical row width per incoming edge in the immutable route's wire order.
    std::vector<uint64_t> ExchangeLengths(const RowTransferPlan &route, const RowLayout &source,
                                          bool preflightDone = false);
    /// Infer the destination storage from the negotiated widths and caller's physical layout policy.
    RowLayout InferredLayout(const RowTransferPlan &route, const RowLayout &source,
                             const std::vector<uint64_t> &received, TransferLayout policy);
}
