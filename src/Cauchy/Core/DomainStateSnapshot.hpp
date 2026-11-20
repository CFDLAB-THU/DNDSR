#pragma once

#include "Cauchy/Core/StateEpoch.hpp"
#include "Cauchy/Config/SimulationConfig.hpp"
#include "Cauchy/State/EquationDofs.hpp"
#include "DNDS/Core/OptionalRef.hpp"

namespace DNDS::Cauchy
{
    class Domain;

    /// Managed immutable view of one accepted domain generation.
    ///
    /// Construction acquires owned read leases for every retained payload, so
    /// means, optional reconstruction, limited-VR history and BDF history
    /// cannot change independently while an output/checkpoint consumer runs.
    /// The snapshot is zero-copy and copyable by shared ownership. It contains
    /// no engine, writer or mutable solver facade; release it before committing
    /// a new accepted state.
    class DomainStateSnapshot
    {
        struct Data;
        std::shared_ptr<const Data> _data;
        DomainStateSnapshot(const Domain &domain, bool reconstruction);
        explicit DomainStateSnapshot(std::shared_ptr<const Data> data) : _data(std::move(data)) {}
        friend class Domain;
        DomainStateSnapshot DetachedTo(Runtime::Device destination, Runtime::ExecutionContext copyContext) const;

    public:
        /// Deep immutable host snapshot for output workers. Copies owned means,
        /// requested reconstruction/history and their matching control metadata.
        /// The source snapshot pins its epoch throughout the synchronous copy;
        /// no MPI or file operation is performed. CPU input is also copied.
        /// Device input uses accounted page-locked host payloads and requires
        /// its ordered copy context. After this returns,
        /// releasing the source snapshot permits solver mutation independently
        /// of the detached result's lifetime. Ghost values are not preserved.
        DomainStateSnapshot DetachedHost(Runtime::ExecutionContext context) const;
        const std::string &Name() const;
        const EquationDofs &Means() const;
        OptionalRef<const EquationDofs> Reconstruction() const;
        OptionalRef<const Runtime::Array<real>> LimitedVRHistory() const;
        OptionalRef<const EquationDofs> PreviousState() const;
        real Time() const;
        real LastAcceptedDt() const;
        real PreviousDt() const;
        const StateEpoch &Epoch() const;
        const std::string &TimeMethod() const;
        const Json &NonlinearState() const;
        const Json &UpdatePolicy() const;
        std::optional<real> PressureGauge() const;
        bool SourceStrang() const;
    };
}
