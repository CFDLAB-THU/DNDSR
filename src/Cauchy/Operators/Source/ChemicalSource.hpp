#pragma once

#include "Cauchy/State/DerivedFields.hpp"
#include "DNDS/Core/OptionalRef.hpp"
#include "Physics/Chemistry/ReactorStepSettings.hpp"

namespace DNDS::Cauchy
{
    /// Synchronous owned-only CPU point-source operator. Replaces values with
    /// positive dU/dt per code volume; no volume weighting, transport, MPI or
    /// integration policy. Flow mass/momentum/full thermochemical energy sources
    /// and any nonchemical blocks (including RANS) are zero; chemistry fills
    /// the named independent species block. This replaces, rather than adds to,
    /// the complete output. Sum separate physical source contributions outside.
    ///
    /// Optional Jacobian rows have N*N logical columns, column-major within
    /// each row, storing positive dS/dU. An implicit residual subtracts this
    /// derivative. No spectrum filter is applied (the active Euler policy).
    /// Separate/coupled physical inputs and outputs may have different layouts;
    /// they must use the same schema/row space and distinct storage. Every output
    /// is reserved before writing. On evaluation failure scratch may be partial;
    /// discard it rather than treating it as a valid operator result.
    ///
    /// A compatible published temperature bundle may be stale: it seeds the UV
    /// solve only. State/ghost values are not changed and ghosts are not read.
    /// Retain handles and serialize this call; move the worker lease with CPU
    /// work if launched asynchronously. No implicit device/host fallback.
    void EvaluateChemicalSource(const EquationSystem &equation, const EquationDofs &state,
                                EquationDofs &values, Physics::Chemistry::ChemicalWorkerPool::Lease worker,
                                OptionalRef<Runtime::Array<real>> jacobian = {},
                                real sourceScale = 1,
                                OptionalRef<const DerivedFields> temperatureGuess = {});
    /// Existing Euler constant-volume CVODE step on owned logical rows, with
    /// distinct input/output allocations and any uniform physical layouts.
    /// Replaces every output component: flow5 and nonchemical blocks are copied
    /// exactly; independent species follow the shared fixed-energy reactor.
    /// dt is code time. Zero duration/scale is an exact copy. No halo exchange,
    /// reconstruction mutation, time history or global reduction is performed.
    /// Caller joins work and coordinates failures before consuming output;
    /// discard partially written output after a point/backend failure.
    void AdvanceChemicalSource(const EquationSystem &equation, const EquationDofs &state, EquationDofs &values,
                               Physics::Chemistry::ChemicalWorkerPool::Lease worker, real dt,
                               const Physics::Chemistry::ReactorStepSettings &controls = {}, real sourceScale = 1);
    /// Pool-backed form reserves a bounded private team before numerical work.
    /// Identical point mathematics and output contract to the single-lease form.
    void AdvanceChemicalSource(const EquationSystem &, const EquationDofs &, EquationDofs &,
                               const Physics::Chemistry::ChemicalWorkerPool &, real dt,
                               const Physics::Chemistry::ReactorStepSettings &controls = {}, real sourceScale = 1);
}
