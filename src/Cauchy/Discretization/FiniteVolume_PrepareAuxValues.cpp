#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using FiniteVolumeDetail::Load;
    using FiniteVolumeDetail::RotateState;
    using FieldView = R::PairReadView<real, R::AccessBoundsChecks, R::UniformTableRows>;
    namespace
    {
        template <class Values>
        auto AuxiliaryArrays(Values &values)
        {
            using Array = std::conditional_t<std::is_const_v<Values>, const R::Array<real>, R::Array<real>>;
            return std::array<Array *, 7>{values.meanThermodynamics ? &*values.meanThermodynamics : nullptr,
                                          values.faceSpectralRadius ? &*values.faceSpectralRadius : nullptr,
                                          values.faceWaveJump ? &*values.faceWaveJump : nullptr,
                                          values.facePressureMinimum ? &*values.facePressureMinimum : nullptr,
                                          values.cellSpectralSum ? &values.cellSpectralSum->Owned() : nullptr,
                                          values.cellWaveJump ? &values.cellWaveJump->Owned() : nullptr,
                                          values.cellPressureMinimum ? &values.cellPressureMinimum->Owned() : nullptr};
        }
    }
    AuxiliaryValues FiniteVolume::BuildAuxiliaryValues(AuxiliaryRequest request) const
    {
        // A caller-provided auxiliary bundle must cover the same mean thermal
        // inputs as the workspace-owned bundle when mixture WBAP is active.
        // This adds only the required cache, not unrelated face outputs.
        if (_data->equation.HasMixture() && _data->settings.numerics.characteristicLimiter &&
            _data->settings.numerics.limiter == "WBAP")
            request.meanThermodynamics = true;
        AuxiliaryValues values;
        const index faces = _data->stencil.stencilFaces.RowCount();
        DNDS_check_throw_info(Device() == R::Device::Host() || request.derived.ComponentNumber() == 0,
                              "queued CUDA derived-field publication is not connected yet");
        if (request.derived.ComponentNumber())
        {
            values._derivedThermodynamics = request.meanThermodynamics;
            if (request.meanThermodynamics)
            {
                for (auto quantity : {DerivedQuantity::Pressure, DerivedQuantity::SoundSpeed, DerivedQuantity::DiffusionBound})
                    request.derived.enabled[int(quantity)] = true;
                if (_data->equation.HasMixture())
                    for (auto quantity : {DerivedQuantity::TotalEnthalpy, DerivedQuantity::PressureInternalEnergyDerivative})
                        request.derived.enabled[int(quantity)] = true;
            }
            values.derivedOwned.emplace(Metrics().Space(), request.derived);
            values.derivedGhost.emplace(Metrics().Space(), request.derived, true);
        }
        if (request.meanThermodynamics && !values._derivedThermodynamics)
            values.meanThermodynamics.emplace(R::RowLayout::Dense(Metrics().NumCellProc(), _data->equation.HasMixture() ? 5 : 3), Device());
        if (request.faceSpectralRadius)
            values.faceSpectralRadius.emplace(R::RowLayout::Dense(faces, 2), Device());
        if (request.faceWaveJump)
            values.faceWaveJump.emplace(R::RowLayout::Dense(faces, 1), Device());
        if (request.facePressureMinimum)
            values.facePressureMinimum.emplace(R::RowLayout::Dense(faces, 1), Device());
        if (request.cellSpectralSum)
            values.cellSpectralSum.emplace(Metrics().BuildField(1));
        if (request.cellWaveJump)
            values.cellWaveJump.emplace(Metrics().BuildField(1));
        if (request.cellPressureMinimum)
            values.cellPressureMinimum.emplace(Metrics().BuildField(1));
        if ((request.cellSpectralSum && !request.faceSpectralRadius) || (request.cellWaveJump && !request.faceWaveJump) ||
            (request.cellPressureMinimum && !request.facePressureMinimum))
            values._faceScratch.emplace(R::RowLayout::Dense(faces, 3), Device());
        const auto label = [](const auto &array, const char *name)
        {
            if (array)
                array->SetName(std::string("Cauchy/auxiliary/") + name, DNDS_ALLOCATION_SITE());
        };
        label(values.meanThermodynamics, "meanThermodynamics");
        label(values.faceSpectralRadius, "faceSpectralRadius");
        label(values.faceWaveJump, "faceWaveJump");
        label(values.facePressureMinimum, "facePressureMinimum");
        label(values._faceScratch, "faceScratch");
        const auto labelCells = [](const auto &field, const char *name)
        {
            if (!field)
                return;
            const auto path = std::string("Cauchy/auxiliary/") + name;
            field->Owned().SetName(path + "/owned", DNDS_ALLOCATION_SITE());
            field->Ghost().SetName(path + "/ghost", DNDS_ALLOCATION_SITE());
        };
        labelCells(values.cellSpectralSum, "cellSpectralSum");
        labelCells(values.cellWaveJump, "cellWaveJump");
        labelCells(values.cellPressureMinimum, "cellPressureMinimum");
        return values;
    }
    AuxiliaryRequest FiniteVolume::CompleteRHSAuxiliaryRequest(AuxiliaryRequest request) const
    {
        if (_data->equation.HasMixture() && _data->settings.numerics.characteristicLimiter &&
            _data->settings.numerics.limiter == "WBAP")
            request.meanThermodynamics = true;
        if (_data->settings.numerics.riemann.useStencil)
        {
            request.cellWaveJump = request.cellWaveJump || _data->fluxPolicy.transverseSignal;
            request.cellPressureMinimum = request.cellPressureMinimum || _data->fluxPolicy.pressureStencil;
        }
        return request;
    }
    std::vector<index> FiniteVolume::AuxiliaryFaces() const
    {
        const auto read = _data->stencil.stencilFaces.Read();
        const auto view = read.View<R::FixedRows<1>>();
        std::vector<index> faces(view.RowCount());
        for (index iFace = 0; iFace < view.RowCount(); ++iFace)
            faces[iFace] = view(iFace, 0);
        return faces;
    }
}
