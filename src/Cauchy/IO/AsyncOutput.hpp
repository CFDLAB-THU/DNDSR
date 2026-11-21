#pragma once

#include "Cauchy/IO/OutputPlan.hpp"
#include <memory>

namespace DNDS::Cauchy
{
    class Simulation;
    struct NonlinearSolveState;
    /** @brief Optional accepted-step and nonlinear parallel-HDF5 writer service.
     * @details Construction collectively validates MPI_THREAD_MULTIPLE and
     * duplicates each participating domain/control communicator once for an
     * independent I/O ordering namespace. Subsequent requests share them.
     * The controller stages each batch on its compute stream into pinned host
     * slots and completes D2H before returning. Both submission methods permit
     * immediate source reuse. File writes run concurrently with subsequent
     * solver updates. The next capture waits for the preceding write and
     * manifest publication before reusing the slots. HDF5 is
     * process-serialized by SerializerH5/Geom. Drain before synchronous
     * external HDF5/CGNS use, mesh mutation or shutdown. Internal files use only their domain I/O communicator.
     * The service may not outlive MPI. Worker exceptions abort immediately.
     */
    class AsyncOutput
    {
        struct Data;
        std::unique_ptr<Data> _data;

    public:
        explicit AsyncOutput(Simulation &);
        ~AsyncOutput();
        AsyncOutput(const AsyncOutput &) = delete;
        AsyncOutput &operator=(const AsyncOutput &) = delete;
        void Submit(Simulation &, const SimulationConfig &, const OutputPlan &, const std::string &token,
                    bool checkpoint, bool visualization, Json outputController, Json snapshotManifest);
        void SubmitInternal(Simulation &, const SimulationConfig &, const OutputPlan &, index domain, const std::string &token,
                            const NonlinearSolveState &, bool checkpoint, bool visualization, Json outputController, Json snapshotManifest);
        void Drain();
    };
}
