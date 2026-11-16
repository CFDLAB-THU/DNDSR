#include "Cauchy/Discretization/FiniteVolumePositivityInternal.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"

namespace DNDS::Cauchy
{
    FiniteVolume::PositivityWorkspace::PositivityWorkspace(std::shared_ptr<const Data> owner,
                                                           const EquationDofs &prototype,
                                                           std::optional<Physics::Chemistry::ChemicalWorkerPool> workers)
        : state(std::make_shared<State>(std::move(owner), prototype, std::move(workers))) {}

    std::pair<index, real> FiniteVolume::PositivityWorkspace::LocalStatistics() const
    {
        auto &data = *state;
        DNDS_check_throw_info(data.beta.Storage().IsIdle(), "modal PP statistics require the enclosing completed schedule");
        if (data.device && data.pending)
        {
            DNDS_check_throw_info(data.device->status.Storage().IsIdle() && data.device->count.Storage().IsIdle() &&
                                      data.device->minimum.Storage().IsIdle(),
                                  "modal PP summary is still queued");
            const auto status = data.device->status.To(R::Device::Host());
            RequireNumericalSuccess(status.Read().View()(0, 0), "queued modal PP final stored-state acceptance");
            const auto count = data.device->count.To(R::Device::Host());
            const auto minimum = data.device->minimum.To(R::Device::Host());
            data.limited = count.Read().View()(0, 0);
            data.minimum = minimum.Read().View()(0, 0);
            data.pending = false;
        }
        return {data.limited, data.minimum};
    }

    void FiniteVolume::PositivityWorkspace::Apply(const EquationDofs &input, const EquationDofs &unlimited,
                                                  EquationDofs &flux, bool refresh)
    {
        auto &data = *state;
        DNDS_check_throw_info(data.owner->metrics.Device() == R::Device::Host(), "synchronous modal PP requires host fields");
        input.RequireCompatible(data.prototype);
        unlimited.RequireCompatible(flux);
        EquationDofOperations::RequireDistinct(input, flux);
        const auto meanScope = input.ReadOwned();
        const auto means = meanScope.ComponentLayout();
        bool same = true;
        for (index iField = 0; iField < flux.PhysicalFieldNumber(); ++iField)
            same = same && flux.PhysicalField(iField).Owned().SharedStorage() == unlimited.PhysicalField(iField).Owned().SharedStorage();
        std::optional<EquationDofs::ReadScope> initialScope;
        std::optional<Algebra::DofComponentLayout<const real>> initial;
        if (refresh && !same)
        {
            EquationDofOperations::RequireDistinct(unlimited, flux);
            initialScope.emplace(unlimited.ReadOwned());
            initial.emplace(initialScope->ComponentLayout());
        }
        auto outputScope = flux.WriteOwned();
        const auto coefficients = outputScope.ComponentLayout();
        auto beta = data.beta.Write();
        auto degree = data.degree.Write();
        CPU::ModalPositivityRequest request{data.geometry, data.policy, means.View(),
                                            initial ? initial->View() : Algebra::DofComponentView<const real>{}, coefficients.View(),
                                            beta.View<R::FixedRows<1>>(), degree.View<R::FixedRows<1>>(), bool(initial)};
        const auto result = RequireNumericalSuccess(CPU::LimitPositiveModalCoefficients(request,
                                                                                        {data.cpuWorkers.data(), index(data.cpuWorkers.size())}),
                                                    "cell-local modal PP");
        data.limited = result.limited;
        data.minimum = result.minimum;
    }
}
