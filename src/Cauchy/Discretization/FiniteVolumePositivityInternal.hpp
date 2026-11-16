#pragma once
#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/Physics/ThermalAdmissibility.hpp"
#include "Cauchy/Discretization/RANSReconstruction.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct FiniteVolume::PositivityWorkspace::State
    {
        std::shared_ptr<const Data> owner;
        EquationDofs prototype;
        ThermalPolicy policy;
        RANSReconstructionView rans;
        index ransFirst = -1;
        std::vector<R::Array<real>> higherValues, baseValues;
        std::optional<EquationDofs> reference;
        R::Array<index> degree;
        R::Array<real> beta, recoveryScale;
        R::Array<uint8_t> recover;
        R::Array<NumericalStatus> emptyStatus{R::RowLayout::Dense(0, 1)};
        struct DeviceStorage
        {
            R::Array<NumericalStatus> status, statusPartial, statusResult;
            R::Array<index> partialCount, count;
            R::Array<real> partialMinimum, minimum;
        };
        std::optional<DeviceStorage> device;
        bool decision = false;
        index limited = 0;
        real minimum = 1;
        State(std::shared_ptr<const Data> data, const EquationDofs &input, const EquationDofs &coefficients,
              std::optional<Physics::Chemistry::ChemicalWorkerPool> workers)
            : owner(std::move(data)), prototype(input.NumericAlias()),
              policy(owner->equation, owner->settings.numerics.densityFloor, owner->settings.numerics.temperatureFloor,
                     owner->settings.numerics.sensibleEnergyFloor, owner->settings.numerics.pressureFloor,
                     workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}),
              degree(R::RowLayout::Dense(owner->metrics.NumCell(), 1), owner->metrics.Device()),
              beta(degree.Layout(), owner->metrics.Device()), recoveryScale(degree.Layout(), owner->metrics.Device()), recover(degree.Layout(), owner->metrics.Device())
        {
            DNDS_check_throw_info(owner->quadrature.positivity, "equation PP sampling geometry is unavailable");
            if (owner->equation.HasRANS())
            {
                rans = ResolveRANSReconstructionPolicy(owner->equation);
                index first = 0;
                for (const auto &block : owner->equation.Schema().blocks)
                {
                    if (block.name == "rans")
                        ransFirst = first;
                    first += index(block.components.size());
                }
                DNDS_assert(ransFirst >= 0);
            }
            for (index iField = 0; iField < input.PhysicalFieldNumber(); ++iField)
            {
                higherValues.emplace_back(FiniteVolumeDetail::FieldLayout(input.PhysicalField(iField).Owned().Layout(), owner->quadrature.positivity->Size(), index(input.PhysicalField(iField).Width())), owner->metrics.Device());
                higherValues.back().SetName("Cauchy/positivity/sampledValues/" + std::to_string(iField), DNDS_ALLOCATION_SITE());
            }
            degree.SetName("Cauchy/positivity/degree", DNDS_ALLOCATION_SITE());
            beta.SetName("Cauchy/positivity/beta", DNDS_ALLOCATION_SITE());
            recoveryScale.SetName("Cauchy/positivity/recoveryScale", DNDS_ALLOCATION_SITE());
            recover.SetName("Cauchy/positivity/recover", DNDS_ALLOCATION_SITE());
            if (!owner->settings.numerics.positivityCompressToMean)
            {
                reference.emplace(EquationDofOperations::NewVector(coefficients));
                reference->Fields().SetName("Cauchy/positivity/referenceCoefficients", DNDS_ALLOCATION_SITE());
                for (const auto &array : higherValues)
                {
                    baseValues.emplace_back(array.Layout(), owner->metrics.Device());
                    baseValues.back().SetName("Cauchy/positivity/baseValues/" + std::to_string(baseValues.size() - 1), DNDS_ALLOCATION_SITE());
                }
            }
        }
        void Sample(const EquationDofs &input, const EquationDofs &coefficients, std::vector<R::Array<real>> &values) const
        {
            for (index iField = 0; iField < input.PhysicalFieldNumber(); ++iField)
                owner->quadrature.positivity->EvaluateValues(input.PhysicalField(iField), coefficients.PhysicalField(iField), values[iField]);
        }
        void Sample(R::EagerEngine &engine, const R::ExecutionStream &stream,
                    const EquationDofs &input, const EquationDofs &coefficients, std::vector<R::Array<real>> &values) const
        {
            for (index iField = 0; iField < input.PhysicalFieldNumber(); ++iField)
                owner->quadrature.positivity->Evaluate(engine, stream, input.PhysicalField(iField), coefficients.PhysicalField(iField), values[iField]);
        }
    };
}
