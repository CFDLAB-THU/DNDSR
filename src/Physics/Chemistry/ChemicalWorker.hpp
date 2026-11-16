#pragma once

#include "Physics/Chemistry/ChemicalSource.hpp"
#include "DNDS/Core/Defines.hpp"
#include <utility>

namespace DNDS::Physics::Chemistry
{
    /// Species scratch allocated at worker creation, then reused per point.
    /// rhoY has full-species capacity even though only Ns-1 entries are inputs;
    /// its pointer remains valid for a single-species mechanism's empty span.
    struct ChemicalWorkBuffers
    {
        std::vector<real> rhoY, fractions, diffusivity, enthalpy, production;
        explicit ChemicalWorkBuffers(int species = 0)
            : rhoY(species), fractions(species), diffusivity(species), enthalpy(species), production(species) {}
    };

    /// Explicit CPU-worker ownership. Copies share capacity. Acquisition is
    /// outside a point loop and fails immediately if exhausted; it does not
    /// identify workers through OpenMP IDs or mutable thread-local globals.
    /// Each lease owns one deep-cloned Cantera context and reusable scratch.
    class ChemicalWorkerPool
    {
        struct State;
        std::shared_ptr<State> _state;

    public:
        class Lease
        {
            friend class ChemicalWorkerPool;
            std::shared_ptr<State> _state;
            index _slot = -1;
            Lease(std::shared_ptr<State> state, index slot) : _state(std::move(state)), _slot(slot) {}
            void Release() noexcept;

        public:
            Lease() = default;
            ~Lease();
            Lease(const Lease &) = delete;
            Lease &operator=(const Lease &) = delete;
            Lease(Lease &&) noexcept;
            Lease &operator=(Lease &&) noexcept;
            explicit operator bool() const { return bool(_state); }
            index Slot() const { return _slot; }
            ChemicalSource &Source() const;
            ChemicalWorkBuffers &Buffers() const;
        };
        ChemicalWorkerPool(const ChemicalSource &prototype, index workers);
        index WorkerNumber() const;
        int SpeciesNumber() const;
        /// Retains allocation/backend lifetime after every pool handle is gone.
        /// Caller serializes use of a lease and never lets a raw context or
        /// scratch reference escape its lifetime. Move the lease into async CPU
        /// work if that work outlives submission; no global synchronization.
        Lease Acquire() const;
        /** @brief Reserve an available team atomically before a CPU row loop.
         * @param maximum Positive upper bound on the number of private contexts.
         * @return Between one and maximum exclusive leases, bounded by the
         * currently available capacity. Exhaustion fails immediately.
         * Allocation precedes locking; no partial team can hold workers while
         * waiting for another context. Kernels consume the returned team size,
         * never acquire contexts or clone Cantera states in their point loop.
         */
        std::vector<Lease> AcquireUpTo(index maximum) const;
    };
}
