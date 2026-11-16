#pragma once
#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "Cauchy/Physics/PerfectGasCharacteristics.hpp"

namespace DNDS::Cauchy::CUDA
{
    using CharacteristicIndices = Runtime::ArrayView<const index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>>;
    using CharacteristicOutput = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::TableRows>;
    /// Immutable shared stencil descriptor, uploaded once during placement.
    /// Parent zero is canonical even when it is a ghost on this rank.
    struct CharacteristicGeometry
    {
        Runtime::ArrayView<const FiniteVolumeStencilView, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> stencil;
        void ValidateDevice(Runtime::Device device) const
        {
            stencil.ValidateDevice(device);
        }
    };
    /** @brief Enqueue the perfect-gas flow-five characteristic producer.
     * @details Operands remain retained by the enclosing operator. Outputs are
     * full canonical-face tables of 25 column-major values; unused faces are
     * identity. No RANS scalar is transformed. Status/fallback are per face;
     * the controller checks/reduces them at an existing segment boundary.
     * Invalid physical states write identity plus failure, not a successful
     * fallback. The near-singular componentwise fallback preserves the CPU
     * policy. Optional CFV inverse diagnostics are separate and off by default.
     * Normally enqueue producer then WBAP on the same stream, without a drain.
     */
    void PrepareFaceCharacteristicMatrices(Runtime::ExecutionContext, CharacteristicGeometry,
                                           Runtime::PairReadView<real, Runtime::AccessBoundsChecks, Runtime::TableRows> mean,
                                           PerfectGas gas, CharacteristicOutput forward, CharacteristicOutput inverse,
                                           Runtime::ArrayView<NumericalStatus, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> status,
                                           Runtime::ArrayView<int, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> fallback);
    /// Deterministic integer summary; partial rows = min(256, ceil(faces/128)),
    /// with at least one row for an empty mesh. Result has one row. No host wait.
    void ReduceCharacteristicFallbacks(Runtime::ExecutionContext,
                                       Runtime::ArrayView<const int, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> flags,
                                       Runtime::ArrayView<index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> partial,
                                       Runtime::ArrayView<index, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> result);
    /// Local sum of (next-current)^2 and next^2. Scratch has the same row count
    /// rule as fallback reduction and width two; result is one row of width two.
    /// The nonlinear controller performs MPI and stopping after observation.
    void ReconstructionUpdateNorm(Runtime::ExecutionContext,
                                  Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> current,
                                  Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> next,
                                  Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> partial,
                                  Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<2>> result);
}
