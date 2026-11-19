#include "Cauchy/Implicit/LocalAMGPreconditioner.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
#include "DNDS/Algebra/AmgXDiagnostics.hpp"
#include "nlohmann/json.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <climits>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    namespace
    {
        void RequireCompact(const R::Array<real> &array, index rows, index width, R::Device device)
        {
            DNDS_check_throw_info(array.GetDevice() == device && array.Layout().Kind() == R::RowLayoutKind::Table &&
                                      array.RowCount() == rows && array.Layout().Width() == width && array.Layout().Stride() == width,
                                  "AmgX copy-free input requires compact AoS; repacking is not implemented");
        }
        template <class T>
        R::ArrayView<T> ScalarView(R::ArrayView<T> view)
        {
            view.rows *= view.width;
            view.width = view.stride = 1;
            return view;
        }
    }
    struct LocalAMGPreconditioner::State
    {
        std::shared_ptr<const Algebra::DofSpace::Partition> partition;
        std::shared_ptr<Algebra::AmgXPreconditioner> provider;
        R::ArrayStorage::Shared diagonalIdentity, offDiagonalIdentity;
        index components = 0;
        uint64_t diagonalEpoch = 0, offDiagonalEpoch = 0;
        bool current = false;
        /// Opt-in completed snapshots; never executed by ordinary production.
        bool diagnostics = std::getenv("DNDS_AMGX_FLOW_DIAGNOSTICS") != nullptr;
        void Report(const char *phase, double seconds, uint64_t nativeMatrixBytes) const
        {
            if (!diagnostics)
                return;
            std::cout << "AMGX_FLOW_MEMORY " << nlohmann::ordered_json{{"phase", phase}, {"seconds", seconds}, {"rows", partition->owned}, {"components", components}, {"native_matrix_bytes", nativeMatrixBytes}, {"memory", nlohmann::ordered_json::parse(Algebra::AmgXMemoryReport(provider->MemorySnapshot()))}}.dump() << std::endl;
        }
    };
    LocalAMGPreconditioner::LocalAMGPreconditioner(const LinearProblemContext &problem, Algebra::AmgXSettings settings)
        : _state(std::make_shared<State>())
    {
        problem.Validate();
        DNDS_check_throw_info(problem.Device().space == R::MemorySpace::CUDA && problem.HasAssembledOffDiagonal() &&
                                  problem.DiagonalMode() == DiagonalStorage::Block && problem.RowMajorBlocks() && problem.OffDiagonal().RowMajorBlocks() && problem.Components() >= 4 && problem.Components() <= 7,
                              "local AmgX requires native row-major assembled CUDA full 4/5/6/7 blocks");
        DNDS_check_throw_info(!problem.Space().Distributed() || MPIInfo(problem.Space().Communicator()).size == 1,
                              "copy-free local AmgX currently requires one rank; native principal-matrix partitions are future work");
        const index rows = index(problem.Space().OwnedRows()), n = problem.Components();
        DNDS_check_throw_info(rows <= INT_MAX / n, "AmgX vector extent exceeds int32");
        const auto &pattern = problem.HostPattern();
        DNDS_check_throw_info(pattern.UsesInt32(), "AmgX requires a primary native int32 linear pattern; index conversion is not implemented");
        RequireCompact(problem.Diagonal(), rows, n * n, problem.Device());
        RequireCompact(problem.OffDiagonal().Values(), problem.OffDiagonal().Values().RowCount(), n * n, problem.Device());
        auto &data = *_state;
        data.partition = problem.Space().Pin();
        data.components = n;
        data.diagonalIdentity = problem.Diagonal().SharedStorage();
        data.offDiagonalIdentity = problem.OffDiagonal().Values().SharedStorage();
        // AmgX's public ROW_MAJOR + separate-diagonal upload consumes the exact
        // native DNDS blocks. It copies into its private provider storage; DNDS
        // neither assembles another matrix nor transposes, filters or converts.
        settings.blockFormat = Algebra::AmgXBlockFormat::RowMajor;
        const auto &weights = problem.GeometricWeights();
        settings.geometricAggregation = true;
        settings.measureFirstCorrection = data.diagnostics;
        data.provider = std::make_shared<Algebra::AmgXPreconditioner>(problem.Device(), settings);
        data.provider->SetPattern(pattern.Offsets32(), pattern.Columns32(), int(n), true);
        data.provider->SetGeometricWeights(weights);
        data.Report("provider_created", 0, uint64_t(rows + problem.OffDiagonal().Values().RowCount()) * n * n * sizeof(real));
    }
    void LocalAMGPreconditioner::Update(R::EagerEngine &engine, const R::ExecutionStream &stream, const LinearProblemContext &problem)
    {
        auto &data = *_state;
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == data.provider->GetDevice() &&
                                  problem.Space().Pin() == data.partition && problem.HasAssembledOffDiagonal() && problem.RowMajorBlocks() && problem.OffDiagonal().RowMajorBlocks() && problem.Components() == data.components &&
                                  problem.Diagonal().SharedStorage() == data.diagonalIdentity && problem.OffDiagonal().Values().SharedStorage() == data.offDiagonalIdentity,
                              "local AmgX matrix/lane identity changed");
        const uint64_t diagonalEpoch = problem.Diagonal().MutationEpoch(), offEpoch = problem.OffDiagonal().Values().MutationEpoch();
        if (data.current && data.diagonalEpoch == diagonalEpoch && data.offDiagonalEpoch == offEpoch)
            return;
        const bool refresh = data.current;
        data.current = false;
        const auto started = data.diagnostics ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        // AmgX is an explicit host-visible boundary. Drain host-progressed MPI
        // before its device-wide join; this provider has no stream API.
        engine.Synchronize();
        const auto diagonal = problem.Diagonal().Read(stream.Context());
        const auto blocks = problem.OffDiagonal().Values().Read(stream.Context());
        if (problem.HasNativeAMGXValues())
            data.provider->AttachValues(stream.Context(), problem.NativeAMGXValues());
        else
            data.provider->UpdateValuesBorrowed(stream.Context(), ScalarView(blocks.View()), ScalarView(diagonal.View()));
        data.diagonalEpoch = diagonalEpoch;
        data.offDiagonalEpoch = offEpoch;
        data.current = true;
        // Diagnostic snapshots join the provider and are excluded from normal
        // timings. Use the before/after epoch/copy counts to identify refresh.
        if (data.diagnostics)
            data.Report(refresh ? "value_refresh" : "initial_setup",
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(),
                        uint64_t(problem.Diagonal().RowCount() + problem.OffDiagonal().Values().RowCount()) * data.components * data.components * sizeof(real));
    }
    void LocalAMGPreconditioner::Apply(R::EagerEngine &engine, const R::ExecutionStream &stream,
                                       const R::Array<real> &rhs, R::Array<real> &output)
    {
        auto &data = *_state;
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == data.provider->GetDevice() && data.current &&
                                  !rhs.Storage().Overlaps(output.Storage()),
                              "local AmgX input/lane aliases or matrix is stale");
        RequireCompact(rhs, index(data.partition->owned), data.components, data.provider->GetDevice());
        RequireCompact(output, index(data.partition->owned), data.components, data.provider->GetDevice());
        engine.Synchronize();
        const auto read = rhs.Read(stream.Context());
        auto write = output.Write(stream.Context());
        const auto started = data.diagnostics ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        data.provider->ApplyBorrowed(stream.Context(), ScalarView(read.View()), ScalarView(write.View()));
        if (data.diagnostics)
            data.Report("apply", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(), 0);
    }
    Algebra::AmgXMemorySnapshot LocalAMGPreconditioner::MemorySnapshot() const
    {
        return _state->provider->MemorySnapshot();
    }

}
