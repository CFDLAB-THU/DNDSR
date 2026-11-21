#include "Cauchy/Core/Simulation.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"

// Deliberately fatal subprocess fixtures. The CTest wrapper requires both the
// expected diagnostic and a nonzero, non-timeout exit; abort is not test success
// by itself. Peers deliberately remain in MPI when rank zero fails.
int main(int argc, char **argv)
{
    DNDS::MPI::Init_thread(&argc, &argv);
    const DNDS::MPIInfo mpi(MPI_COMM_WORLD);
    try
    {
        const std::string mode = argc == 2 ? argv[1] : "";
        DNDS_check_throw_info(mpi.size >= 2, "failure fixture needs at least two ranks");
        if (mode == "factory")
        {
            auto config = DNDS::Cauchy::SimulationConfig::ResolveCollective(DNDS::Cauchy::Json::object(), mpi.comm);
            DNDS::Cauchy::Simulation simulation(config, mpi.comm, [&](auto, const auto &, const auto &) -> DNDS::ssp<DNDS::Geom::UnstructuredMesh>
                                                {
                if (mpi.rank == 0) throw std::runtime_error("injected domain factory failure");
                MPI_Barrier(mpi.comm);
                throw std::runtime_error("factory peer escaped unexpectedly"); });
        }
        else if (mode == "transfer")
        {
            namespace R = DNDS::Runtime;
            R::Array<double> source(R::RowLayout::Dense(1, 1)), destination(R::RowLayout::Dense(1, 1));
            {
                auto write = source.Write();
                write.View()(0, 0) = mpi.rank;
            }
            auto context = std::make_shared<R::CommunicationContext>(mpi.comm);
            R::RowTransferPlan route(context, 1, 1, {{(mpi.rank + 1) % mpi.size, 0, 0}});
            auto binding = R::PayloadPlan(route, source.Layout(), destination.Layout(), R::ElementSchema::Arithmetic<double>())
                               .Bind(source, destination);
            auto operation = binding.Begin();
            if (mpi.rank == 0)
                throw std::runtime_error("injected error with active transfer");
            operation.Wait();
            MPI_Barrier(mpi.comm);
        }
        else if (mode == "reducer-completion")
        {
            namespace R = DNDS::Runtime;
            R::Array<int64_t> source(R::RowLayout::Dense(2, 1)), destination(R::RowLayout::Dense(1, 1));
            source.Write().View()(0, 0) = mpi.rank;
            source.Write().View()(1, 0) = mpi.rank + 1;
            R::RowTransferPlan route(mpi.comm, 2, 1, {{mpi.rank, 0, 0}, {(mpi.rank + 1) % mpi.size, 1, 0}});
            auto reducer = R::Reducer::Custom<int64_t>(R::ElementSchema::Arithmetic<int64_t>(), "fatal-reducer", [rank = mpi.rank](auto &a, const auto &b)
                                                       {
                if (rank == 0)
                    throw std::runtime_error("injected reduction failure");
                a += b; }, int64_t(0));
            R::PayloadPlan(route, source.Layout(), destination.Layout(), R::ElementSchema::Arithmetic<int64_t>())
                .Bind(source, destination, reducer)
                .Begin()
                .Wait();
            MPI_Barrier(mpi.comm);
        }
        else if (mode == "solver-checked-preflight" || mode == "solver-unchecked-preflight")
        {
            namespace R = DNDS::Runtime;
            R::Array<double> source(R::RowLayout::Dense(1, 1)), destination(source.Layout());
            // Every peer needs a send from rank zero. In a ring with four or
            // more ranks a non-neighbour can finish and begin MPI teardown.
            R::RowTransferPlan route(mpi.comm, 1, 1, {{mpi.rank == 0 ? 1 : 0, 0, 0}});
            R::TransferOptions options;
            const bool unchecked = mode == "solver-unchecked-preflight";
            options.preflight = unchecked ? R::TransferPreflight::AbortOnFailure : R::TransferPreflight::Collective;
            auto binding = R::PayloadPlan(route, source.Layout(), destination.Layout(), R::ElementSchema::Arithmetic<double>())
                               .Bind(source, destination, R::TransferTransport::Direct, options);
            const DNDS::MPI::SolverFailureScope solverFailures(mpi.comm);
            std::optional<R::Array<double>::ReadLease> held;
            if (mpi.rank == 0)
                held.emplace(destination.Read());
            try
            {
                // In local-preflight mode let the healthy ranks post their
                // receives before rank zero encounters its conflicting lease.
                // Collective preflight requires all ranks to enter Begin together.
                if (unchecked && mpi.rank == 0)
                    MPI_Barrier(mpi.comm);
                auto operation = binding.Begin();
                if (unchecked && mpi.rank != 0)
                    MPI_Barrier(mpi.comm);
                operation.Wait();
            }
            catch (...)
            {
                // Reaching an outer fatal handler would not prove that the
                // production preflight terminates at the point of rejection.
                DNDS::MPI::Abort(mpi.comm, "unexpectedly caught solver preflight exception");
            }
        }
        else if (mode == "solver-worker")
        {
            namespace R = DNDS::Runtime;
            const DNDS::MPI::SolverFailureScope solverFailures(mpi.comm);
            R::EagerEngine engine(1);
            const auto stream = engine.CreateCPUStream();
            if (mpi.rank == 0)
            {
                engine.Submit(stream, [](auto)
                              { throw std::runtime_error("injected solver worker failure"); });
                engine.Synchronize();
            }
            MPI_Barrier(mpi.comm);
        }
        else
            throw std::runtime_error("unknown failure fixture mode");
    }
    catch (const std::exception &error)
    {
        DNDS::MPI::Abort(mpi.comm, error.what());
    }
    DNDS::MPI::Finalize();
    return 0;
}
