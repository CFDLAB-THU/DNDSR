#include "Cauchy/Operators/Source/ChemicalSource.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "Cauchy/Operators/Source/CPU/ChemicalSourceKernels.hpp"
#include "Cauchy/Physics/PointProperties.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/Core/OMP.hpp"
#include <limits>

namespace DNDS::Cauchy
{
    namespace
    {
        void AdvanceChemicalTeam(const EquationSystem &equation, const EquationDofs &state, EquationDofs &values,
                                 std::vector<Physics::Chemistry::ChemicalWorkerPool::Lease> team, real dt,
                                 const Physics::Chemistry::ReactorStepSettings &controls, real sourceScale)
        {
            state.RequireSchema(equation.Schema());
            state.RequireCompatible(values);
            EquationDofOperations::RequireDistinct(state, values);
            controls.Validate();
            DNDS_check_throw_info(equation.HasMixture() && state.Device() == Runtime::Device::Host() &&
                                      std::isfinite(dt) && dt >= 0 && std::isfinite(sourceScale) && sourceScale >= 0,
                                  "unavailable or invalid constant-volume chemical step");
            std::vector<Physics::Thermodynamics::IdealGasMixture::PointEvaluator> evaluators;
            evaluators.reserve(team.size());
            for (auto &worker : team)
                evaluators.push_back(equation.Mixture().Borrow(std::move(worker)));
            const auto input = state.ReadOwned();
            std::vector<EquationBlockArrayView<const real>> from;
            std::vector<EquationBlockArrayView<real>> to;
            for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                from.push_back(input.Block(iBlock).owned);
            std::optional<EquationDofs::WriteScope> output;
            try
            {
                output.emplace(values.WriteOwned());
                for (index iBlock = 0; iBlock < values.BlockNumber(); ++iBlock)
                    to.push_back(output->Block(iBlock));
            }
            catch (...)
            {
                MPI::AbortIfSolverFailure("Cauchy chemical source operation failed");
                if (output)
                    output->CancelUnmodified();
                throw;
            }
            const auto flow = input.Block("flow").owned;
            const int independent = equation.Mixture().SpeciesNumber() - 1;
            const auto species = independent ? input.Block("species").owned : EquationBlockArrayView<const real>{};
            const auto target = independent ? output->Block("species") : EquationBlockArrayView<real>{};
            const Runtime::MetadataBuffer<EquationBlockArrayView<const real>> sourceMetadata(from);
            const Runtime::MetadataBuffer<EquationBlockArrayView<real>> targetMetadata(to);
            RequireNumericalSuccess(CPU::AdvanceChemicalRows(sourceMetadata.HostView(), targetMetadata.HostView(), flow, species, target,
                                                             {evaluators.data(), index(evaluators.size())}, independent, dt, controls, sourceScale),
                                    "AdvanceChemicalRows");
        }
    }
    void AdvanceChemicalSource(const EquationSystem &equation, const EquationDofs &state, EquationDofs &values,
                               Physics::Chemistry::ChemicalWorkerPool::Lease worker, real dt,
                               const Physics::Chemistry::ReactorStepSettings &controls, real sourceScale)
    {
        std::vector<Physics::Chemistry::ChemicalWorkerPool::Lease> team;
        team.push_back(std::move(worker));
        AdvanceChemicalTeam(equation, state, values, std::move(team), dt, controls, sourceScale);
    }
    void AdvanceChemicalSource(const EquationSystem &equation, const EquationDofs &state, EquationDofs &values,
                               const Physics::Chemistry::ChemicalWorkerPool &pool, real dt,
                               const Physics::Chemistry::ReactorStepSettings &controls, real sourceScale)
    {
        AdvanceChemicalTeam(equation, state, values, pool.AcquireUpTo(OMP::DistributedThreads()), dt, controls, sourceScale);
    }
    void EvaluateChemicalSource(const EquationSystem &equation, const EquationDofs &state,
                                EquationDofs &values, Physics::Chemistry::ChemicalWorkerPool::Lease worker,
                                OptionalRef<Runtime::Array<real>> jacobian,
                                real sourceScale,
                                OptionalRef<const DerivedFields> temperatureGuess)
    {
        state.RequireSchema(equation.Schema());
        state.RequireCompatible(values);
        DNDS_check_throw_info(equation.HasMixture() && state.Device() == Runtime::Device::Host(), "chemical source has no kernel for the requested profile/backend");
        DNDS_check_throw_info(std::isfinite(sourceScale) && sourceScale >= 0, "chemical source scale must be finite and nonnegative");
        EquationDofOperations::RequireDistinct(state, values);
        const index components = state.ComponentNumber();
        // Resolve logical component locations explicitly. Physical field names,
        // grouping and segment offsets do not determine equation ordering.
        std::vector<index> columns;
        index first = 0, flowFirst = -1, speciesFirst = -1;
        for (const auto &block : state.Schema().blocks)
        {
            if (block.name == "flow")
                flowFirst = first;
            if (block.name == "species")
                speciesFirst = first;
            first += index(block.components.size());
        }
        for (index j = 0; j < 5; ++j)
            columns.push_back(flowFirst + j);
        const int species = equation.Mixture().SpeciesNumber() - 1;
        for (int k = 0; k < species; ++k)
            columns.push_back(speciesFirst + k);
        if (jacobian)
        {
            const auto &matrix = *jacobian;
            DNDS_check_throw_info(components > 0 && components <= std::numeric_limits<index>::max() / components, "source Jacobian row width overflows");
            DNDS_check_throw_info(matrix.GetDevice() == state.Device() && matrix.RowCount() == index(state.Space().OwnedRows()) &&
                                      !matrix.Layout().IsVariable() && matrix.Layout().Width() == components * components,
                                  "chemical source Jacobian shape/backend differs");
            for (const auto *bound : std::array<const EquationDofs *, 2>{&state, &values})
                for (index iField = 0; iField < bound->PhysicalFieldNumber(); ++iField)
                    for (const auto *array : {&bound->PhysicalField(iField).Owned(), &bound->PhysicalField(iField).Ghost()})
                        DNDS_check_throw_info(!Algebra::Detail::Overlaps(matrix.Storage(), array->Storage()), "source Jacobian aliases equation storage");
        }
        auto evaluator = equation.MakePointEvaluator(std::move(worker), {false, false, false, true, bool(jacobian)});
        std::optional<DerivedFields::ReadScope> guess;
        if (temperatureGuess)
        {
            DNDS_check_throw_info(!temperatureGuess->GhostPart(), "source temperature guess must be owned");
            guess.emplace(temperatureGuess->ReadTemperatureGuess(state, equation));
        }
        auto input = state.ReadOwned();
        const auto flow = input.Block("flow").owned;
        const auto rhoY = species ? input.Block("species").owned : EquationBlockArrayView<const real>{};
        std::optional<Runtime::Array<real>::WriteLease> jacobianWrite;
        std::optional<EquationDofs::WriteScope> output;
        try
        {
            if (jacobian)
                jacobianWrite.emplace(jacobian->Write());
            output.emplace(values.WriteOwned());
        }
        catch (...)
        {
            MPI::AbortIfSolverFailure("Cauchy chemical source operation failed");
            if (jacobianWrite)
                jacobianWrite->CancelUnmodified();
            throw;
        }
        const auto matrix = jacobianWrite ? jacobianWrite->View<Runtime::UniformTableRows>() : Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows>{};
        // This operator replaces the full source, not just its chemical block.
        // Resolve nonchemical outputs once; future RANS/other blocks must not
        // retain values from a previous use of this scratch allocation.
        std::vector<EquationBlockArrayView<real>> nonchemical;
        for (index iBlock = 0; iBlock < values.BlockNumber(); ++iBlock)
            if (values.Schema().blocks[iBlock].name != "species")
                nonchemical.push_back(output->Block(iBlock));
        const auto sourceSpecies = species ? output->Block("species") : EquationBlockArrayView<real>{};
        const auto oldT = guess ? guess->Field(DerivedQuantity::Temperature) : EquationBlockArrayView<const real>{};
        const Runtime::MetadataBuffer<EquationBlockArrayView<real>> outputs(nonchemical);
        const Runtime::MetadataBuffer<index> columnMetadata(columns);
        RequireNumericalSuccess(CPU::EvaluateChemicalRows(flow, rhoY, sourceSpecies, outputs.HostView(), columnMetadata.HostView(),
                                                          components, speciesFirst, species, sourceScale, jacobianWrite ? std::optional(matrix) : std::nullopt,
                                                          guess ? std::optional(oldT) : std::nullopt, evaluator),
                                "EvaluateChemicalRows");
    }
}
