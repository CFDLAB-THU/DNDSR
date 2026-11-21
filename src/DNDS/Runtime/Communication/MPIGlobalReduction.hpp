#pragma once

#include "DNDS/Runtime/Communication/GlobalReduction.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"

namespace DNDS::Runtime
{
    namespace Detail
    {
        std::shared_ptr<const GlobalReductionBinding> BindMPIAllreduceBuffers(
            std::shared_ptr<CommunicationContext>, Buffer, Buffer, const RowLayout &, const RowLayout &, ElementScalar, Reduction);
    }

    /** Bind compact host tables to an elementwise MPI sum, minimum or maximum.
     * Setup is collective. The caller owns collective ordering on the supplied
     * communicator. Submit all its queued collectives on one CPU communication
     * stream per rank, in matching order. Drain before issuing unrelated host
     * collectives on that communicator. Separate MPI contexts may run independently.
     * Requests progress on the MPI controller during event observation and engine
     * progress. A CPU compute worker never waits inside MPI.
     */
    template <class T>
    std::shared_ptr<const GlobalReductionBinding> BindMPIAllreduce(
        std::shared_ptr<CommunicationContext> context, const Array<T> &source, Array<T> &destination, Reduction operation)
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                      std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>);
        DNDS_check_throw_info(context, "MPI allreduce requires a communication context");
        RowLayout in, out;
        Buffer src, dst;
        try
        {
            const auto a = source.SharedStorage(), b = destination.SharedStorage();
            DNDS_check_throw_info(a && b, "MPI allreduce requires published Arrays");
            in = a->Layout();
            out = b->Layout();
            src = a->Values();
            dst = b->Values();
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("MPI allreduce Array setup failed");
            src = {}; // Include a local rejection in mandatory binding agreement.
        }
        return Detail::BindMPIAllreduceBuffers(std::move(context), src, dst, in, out, ElementSchema::Arithmetic<T>().Fields().front().scalar, operation);
    }
}
