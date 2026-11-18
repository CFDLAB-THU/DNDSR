#pragma once

#include "Cauchy/Physics/PerfectGas.hpp"
#include "Cauchy/Physics/FlowCharacteristics.hpp"
#include "Cauchy/State/DerivedFields.hpp"
#include "DNDS/Core/OptionalRef.hpp"
#include <optional>

namespace DNDS::Cauchy
{
    using RHSFieldView = Runtime::PairReadView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using RHSFlowRows = Runtime::FixedWidthRows<Runtime::UniformTableRows, 5>;
    using RHSFlowFieldView = Runtime::PairReadView<real, Runtime::AccessBoundsChecks, RHSFlowRows>;
    using RHSScalarView = Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using RHSOutputView = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>;
    using RHSFlowOutputView = Runtime::ArrayView<real, Runtime::AccessBoundsChecks, RHSFlowRows>;
    /// One kernel region's plain, movable diagnostics. The cell loop updates
    /// only this local value and returns it for one final merge. A future OMP
    /// implementation can reduce the same value per thread; no atomic belongs
    /// in a face/quadrature hot loop.
    class RHSKernelDiagnostics
    {
        index _faceFallbacks = 0;
        std::vector<Vector3> _forces;
        friend class RHSLocalDiagnostics;

    public:
        explicit RHSKernelDiagnostics(index forceSlots = 0) : _forces(size_t(forceSlots), Vector3{}) {}
        void AddFaceFallback() { ++_faceFallbacks; }
        void AddBoundaryForce(index slot, int component, real value)
        {
            DNDS_assert_info(slot >= 0 && size_t(slot) < _forces.size() && component >= 0 && component < 3,
                             "invalid RHS boundary-force slot");
            _forces[size_t(slot)][component] += value;
        }
        index FaceFallbacks() const { return _faceFallbacks; }
        /// Merge private worker counters after all independent rows finish.
        /// No atomic or shared counter update belongs in a quadrature loop.
        void Merge(const RHSKernelDiagnostics &other) noexcept
        {
            DNDS_assert(_forces.size() == other._forces.size());
            _faceFallbacks += other._faceFallbacks;
            for (index slot = 0; slot < index(_forces.size()); ++slot)
                for (int d = 0; d < 3; ++d)
                    _forces[slot][d] += other._forces[slot][d];
        }
    };
    /// Workspace-owned, rank-local aggregate for one complete RHS evaluation.
    /// Boundary zones are resolved to dense slots during plan setup; kernels
    /// neither search a map nor mutate this aggregate from their inner loop.
    class RHSLocalDiagnostics
    {
        std::vector<int> _zones;
        std::vector<Vector3> _forces;
        index _faceFallbacks = 0;

    public:
        RHSLocalDiagnostics() = default;
        explicit RHSLocalDiagnostics(std::vector<int> zones);
        void Reset();
        void Merge(const RHSKernelDiagnostics &input);
        index FaceFallbacks() const { return _faceFallbacks; }
        Vector3 BoundaryForce(const std::vector<int> &zones) const;
    };
    /// Borrowed [p,a,diffusivity] projection, optionally followed by [H,beta]
    /// for the cached CPU mixture characteristic producer. These are cell-mean
    /// values; face averages are local temporaries, not another field cache.
    /// Preserve parent layout and retain the producer's read scopes.
    struct MeanThermodynamicView
    {
        RHSFieldView values;
        std::array<int, 5> columns{0, 1, 2, 3, 4};
        index width = 3;
        MeanThermodynamicView() = default;
        MeanThermodynamicView(RHSScalarView combined) : values{combined, {}}, width(combined.Width()) {}
        MeanThermodynamicView(RHSFieldView paired, std::array<int, 5> selected, index count)
            : values(paired), columns(selected), width(count) {}
        static MeanThermodynamicView Derived(
            const DerivedFields::ReadScope &owned,
            OptionalRef<const DerivedFields::ReadScope> ghost = {})
        {
            std::array<int, 5> selected{owned.Column(DerivedQuantity::Pressure), owned.Column(DerivedQuantity::SoundSpeed), owned.Column(DerivedQuantity::DiffusionBound), -1, -1};
            const bool characteristics = owned.Has(DerivedQuantity::TotalEnthalpy) && owned.Has(DerivedQuantity::PressureInternalEnergyDerivative);
            if (characteristics)
            {
                selected[3] = owned.Column(DerivedQuantity::TotalEnthalpy);
                selected[4] = owned.Column(DerivedQuantity::PressureInternalEnergyDerivative);
            }
            if (ghost)
            {
                DNDS_check_throw_info(characteristics == (ghost->Has(DerivedQuantity::TotalEnthalpy) && ghost->Has(DerivedQuantity::PressureInternalEnergyDerivative)),
                                      "derived owned/ghost characteristic selections differ");
                std::array<int, 5> ghostColumns{ghost->Column(DerivedQuantity::Pressure), ghost->Column(DerivedQuantity::SoundSpeed), ghost->Column(DerivedQuantity::DiffusionBound), -1, -1};
                if (characteristics)
                {
                    ghostColumns[3] = ghost->Column(DerivedQuantity::TotalEnthalpy);
                    ghostColumns[4] = ghost->Column(DerivedQuantity::PressureInternalEnergyDerivative);
                }
                DNDS_check_throw_info(selected == ghostColumns, "derived owned/ghost property columns differ");
            }
            return {{owned.View(), ghost ? ghost->View() : RHSScalarView{}}, selected, characteristics ? 5 : 3};
        }
        index RowCount() const { return values.RowCount(); }
        index Width() const { return width; }
        real operator()(index iRow, int column) const { return values(iRow, columns[column]); }
        bool HasCharacteristics() const { return width >= 5; }
        /// Shape/capability is checked once by the operator before the loop.
        CharacteristicThermodynamics Characteristics(index iRow) const
        {
            return {(*this)(iRow, 0), (*this)(iRow, 1), (*this)(iRow, 3), (*this)(iRow, 4)};
        }
    };
    /// Packed read-only auxiliary inputs, independent of their producer. Current
    /// fields are mean [p,a,diffusivity], wave jumps and pressure minima. Extend
    /// this bundle for cached temperature/masks rather than adding positional
    /// arguments to every kernel. Optional absence is explicit even on empty ranks.
    struct RHSAuxiliaryViews
    {
        std::optional<MeanThermodynamicView> meanThermodynamics;
        std::optional<RHSFieldView> cellWaveJump, cellPressureMinimum;
    };
    struct RHSReadViews
    {
        RHSFlowFieldView mean;
        RHSFieldView coefficients, primitiveGradient;
        RHSAuxiliaryViews auxiliary;
        /// Euler's physical-dt trial ignores viscosity while retaining the
        /// configured boundary/convective flux policy. Gradients may be absent
        /// when false; normal production residuals keep the default true.
        bool includeViscosity = true;
        /// Optional distinct polynomial for viscosity. Absence uses coefficients.
        /// Retain its read leases through execution, exactly as for coefficients.
        std::optional<RHSFieldView> viscousCoefficients;
        std::optional<RHSScalarView> boundary;
        /// Direct FV physical conservative gradients, component*dim+direction.
        /// Required for direct P1; P0 face flux never reads this input.
        std::optional<RHSFieldView> conservativeGradient, viscousConservativeGradient;
        /// Explicit physical-dt/mean trial; no reconstructed point increment.
        bool meanOnly = false;
        /// Host-wrapper validation of borrowed inputs. Ownership and producer
        /// readiness remain the caller's responsibility; no per-point checks.
        void ValidateDevice(Runtime::Device expected) const
        {
            Runtime::ValidateViewDevices(expected, mean, coefficients, primitiveGradient);
            if (auxiliary.meanThermodynamics)
                auxiliary.meanThermodynamics->values.ValidateDevice(expected);
            if (auxiliary.cellWaveJump)
                auxiliary.cellWaveJump->ValidateDevice(expected);
            if (auxiliary.cellPressureMinimum)
                auxiliary.cellPressureMinimum->ValidateDevice(expected);
            if (viscousCoefficients)
                viscousCoefficients->ValidateDevice(expected);
            if (boundary)
                boundary->ValidateDevice(expected);
            if (conservativeGradient)
                conservativeGradient->ValidateDevice(expected);
            if (viscousConservativeGradient)
                viscousConservativeGradient->ValidateDevice(expected);
        }
    };
    /// Explicit numerical outputs. Diagnostics are returned by the RHS kernel
    /// as a value and merged after each core/dependent operation completes.
    struct RHSWriteViews
    {
        RHSFlowOutputView residual;
        std::optional<RHSFlowOutputView> lowOrder;
        /// Optional integrated negative outward flux in AuxiliaryFaces() order,
        /// in the prepared anchor cell's axes. Selected faces are overwritten;
        /// core/dependent sets can overlap. No source term is included.
        std::optional<RHSFlowOutputView> faceFlux;
        bool boundaryFacesOnly = false;
        /// Outward momentum flux times quadrature weight. Optional observation output.
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::FixedRows<3>> boundaryMomentum;
        void ValidateDevice(Runtime::Device expected) const
        {
            residual.ValidateDevice(expected);
            if (lowOrder)
                lowOrder->ValidateDevice(expected);
            if (faceFlux)
                faceFlux->ValidateDevice(expected);
            if (boundaryMomentum.RowCount())
                boundaryMomentum.ValidateDevice(expected);
        }
    };
}

namespace DNDS::Cauchy::EquationFVDetail
{
    /// Controller-side descriptor construction under the supplied read scope.
    /// Numeric kernels receive the resulting metadata span, never DOF owners.
    Runtime::MetadataBuffer<EquationBlockReadView> BlockViews(const EquationDofs &, const EquationDofs::ReadScope &);
}
