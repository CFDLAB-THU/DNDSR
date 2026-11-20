#pragma once

#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Cauchy/Implicit/ResidualConvergence.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <chrono>
#include <tuple>
#include <type_traits>

namespace DNDS::Cauchy::NonlinearDetail
{
    /// Accumulate numerical work without treating coarse corrections as finest
    /// nonlinear iterations or overwriting the finest convergence observation.
    template <class Report>
    void AccumulateWork(Report &total, const Report &step)
    {
        total.linearIterations += step.linearIterations;
        total.rhsLimited += step.rhsLimited;
        total.incrementLimited += step.incrementLimited;
        total.minimumRHSAlpha = std::min(total.minimumRHSAlpha, step.minimumRHSAlpha);
        total.minimumIncrementAlpha = std::min(total.minimumIncrementAlpha, step.minimumIncrementAlpha);
        total.lastLinear = step.lastLinear;
        total.multigridCycles += step.multigridCycles;
        if (total.levelCorrections.size() < step.levelCorrections.size())
            total.levelCorrections.resize(step.levelCorrections.size());
        for (index level = 0; level < index(step.levelCorrections.size()); ++level)
            total.levelCorrections[level] += step.levelCorrections[level];
    }
    /// Resolve storage before allocating the canonical matrix. Selecting a
    /// AmgX borrows primary storage; CPU AMG extracts its local principal matrix.
    inline LinearMatrixOptions ResolveMatrixStorage(const ImplicitLinearSettings &settings)
    {
        const auto table = [](const ImplicitMatrixTableSettings &input)
        {
            DNDS_check_throw_info(input.layout == "aos" || input.layout == "soa" || input.layout == "aosoa32", "invalid matrix table layout");
            return Algebra::MatrixTableOptions{input.layout == "aos" ? Algebra::MatrixTableLayout::AoS : input.layout == "soa" ? Algebra::MatrixTableLayout::SoA
                                                                                                                               : Algebra::MatrixTableLayout::AoSoA32,
                                               input.rowPaddingMultiple, input.rowStridePaddingMultiple, input.allocationAlignmentBytes};
        };
        DNDS_check_throw_info(settings.matrixStorage.indices == "auto" || settings.matrixStorage.indices == "int32" ||
                                  settings.matrixStorage.indices == "int64",
                              "invalid matrix index storage");
        const bool amg = settings.method == "amgx" || (settings.method == "gmres" && settings.preconditioner == "amgx");
        LinearMatrixOptions result;
        result.diagonal = table(settings.matrixStorage.diagonal);
        result.inverse = table(settings.matrixStorage.inverse);
        result.offDiagonal = table(settings.matrixStorage.offDiagonal);
        result.indices = settings.matrixStorage.indices == "int32" || (settings.matrixStorage.indices == "auto" && amg)
                             ? LinearIndexStorage::Int32
                             : LinearIndexStorage::Int64;
        result.rowMajorBlocks = amg;
        result.cpuAMG = settings.method == "amg" || (settings.method == "gmres" && settings.preconditioner == "amg");
        DNDS_check_throw_info(settings.matrixStorage.amgxStorage == "upload" ||
                                  (amg && settings.matrixStorage.amgxStorage == "native"),
                              "invalid AmgX numeric storage mode");
        result.amgxNativeStorage = settings.matrixStorage.amgxStorage == "native";
        return result;
    }
    inline LinearSolveSettings ResolveLinear(const ImplicitSettings &settings, int dimension, bool acm = false)
    {
        LinearSolveSettings result;
        result.method = settings.linear.method;
        result.preconditioner = settings.linear.preconditioner;
        result.stopping = settings.linear.stopping;
        result.maxIterations = settings.linear.iterations;
        result.residualCheckInterval = settings.linear.residualCheckInterval;
        result.subspace = settings.linear.subspace;
        result.relativeTolerance = settings.linear.relativeTolerance;
        result.absoluteTolerance = settings.linear.absoluteTolerance;
        result.sgsSweeps = settings.sgs.sweeps;
        result.robustACMInverse = acm;
        result.amgxAggregateSize = settings.linear.amgxAggregateSize ? settings.linear.amgxAggregateSize : (dimension == 3 ? 8 : 4);
        result.jacobiLocalSweeps = settings.linear.jacobiLocalSweeps;
        result.diagonalInverseRefinement = settings.linear.diagonalInverseRefinement;
        result.cudaDiagonalInverse = settings.linear.cudaDiagonalInverse;
        result.cudaJacobiMapping = settings.linear.cudaJacobiMapping;
        result.exchangeBetweenSweeps = settings.sgs.exchangeBetweenSweeps;
        result.iluFill = settings.ilu.fill;
        result.iluPreconditionerCorrections = settings.ilu.preconditionerCorrections;
        result.iluCoreFirst = settings.ilu.coreFirst;
        result.iluOrdering = settings.ilu.ordering;
        return result;
    }
    /// Exact controller agreement, including component lists larger than a
    /// machine-word mask. Callers coordinate serialization before entering.
    inline void AgreeControls(const MPIInfo &mpi, const std::string &serialized)
    {
        int length = mpi.rank == 0 ? int(serialized.size()) : 0;
        MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
        std::string reference;
        MPI::LocalStage(mpi, [&]
                        { reference = mpi.rank == 0 ? serialized : std::string(size_t(length), '\0'); });
        MPI_Bcast(reference.data(), length, MPI_CHAR, 0, mpi.comm);
        MPI::LocalStage(mpi, [&]
                        { DNDS_check_throw_info(reference == serialized, "implicit controls differ between domain ranks"); });
    }
    /// Optional local timing only. Scope unwinding preserves partial evidence.
    class PhaseTimer
    {
        using Clock = std::chrono::steady_clock;
        std::shared_ptr<NonlinearProfile> _profile;
        NonlinearProfile::Phase _phase;
        Clock::time_point _start{};

    public:
        PhaseTimer(std::shared_ptr<NonlinearProfile> profile, NonlinearProfile::Phase phase)
            : _profile(std::move(profile)), _phase(phase)
        {
            if (_profile)
                _start = Clock::now();
        }
        PhaseTimer(const PhaseTimer &) = delete;
        PhaseTimer &operator=(const PhaseTimer &) = delete;
        ~PhaseTimer()
        {
            if (_profile)
                _profile->seconds[_phase] += std::chrono::duration<real>(Clock::now() - _start).count();
        }
    };

    /// One iteration policy for field and logical-equation adapters. Residual
    /// and correction callbacks own their numerical access/MPI contracts;
    /// measure returns globally reduced norms without modifying their order.
    /// No physical acceptance, rollback or extra collective is introduced here.
    enum class ResidualObservation
    {
        BeforeUpdate,
        AfterUpdate
    };
}
