#pragma once

#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/Core/NumericalStatus.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct MeanGradients::Data
    {
        std::weak_ptr<const void> owner;
        // A standalone producer owns a halo bound to one state. The FV
        // workspace borrows the enclosing schedule's state halo and can serve
        // either coupled stage. The issued arrays still pin the runtime input.
        std::optional<EquationDofOperations::Binding> stateBinding;
        std::optional<EquationDofOperations::Binding> origin;
        std::vector<CFV::FVField> fields;
        std::vector<R::Array<real>> boundary;
        std::vector<std::vector<index>> vectors;
        std::vector<CFV::GradientWorkspace> execution;
        struct DeviceBoundaryStatus
        {
            R::Array<NumericalStatus> rows, partial, result, empty;
        };
        std::optional<DeviceBoundaryStatus> deviceBoundaryStatus;
        std::vector<R::TransferBinding> meanHalo, gradientHalo;
        std::vector<uint64_t> sourceGhost, ownedVersion, ghostVersion, ownedShape, ghostShape;
        real time = 0;
        bool ghostsReady = false;
        /// Issued content identities for an ordered preparation segment. This
        /// is provenance only, never permission to read pending arrays on CPU.
        /// Execution permission comes from Runtime operands and stream edges.
        struct IssuedArray
        {
            R::ArrayStorage::Shared storage;
            uint64_t shape = 0, epoch = 0;
            explicit IssuedArray(const R::Array<real> &array)
                : storage(array.SharedStorage()), shape(array.ShapeVersion()), epoch(array.MutationEpoch()) {}
            bool Matches(const R::Array<real> &array) const
            {
                return storage == array.SharedStorage() && shape == array.ShapeVersion() &&
                       epoch == array.MutationEpoch() && array.IsValid();
            }
        };
        std::optional<EquationDofs> queuedSource;
        std::vector<std::array<IssuedArray, 3>> queuedOwned;
        std::vector<IssuedArray> queuedGhost;
        real queuedTime = 0;
        explicit Data(std::weak_ptr<const void> plan) : owner(std::move(plan)) {}
        void CheckQueued(const EquationDofs &state, const std::vector<CFV::FVField> &values,
                         const std::shared_ptr<const void> &plan, real atTime, bool ghosts) const
        {
            if (!queuedSource)
            {
                Check(state, values, plan, atTime, ghosts);
                return;
            }
            if (stateBinding)
                stateBinding->Check(state);
            DNDS_check_throw_info(owner.lock() == plan && queuedTime == atTime && values.size() == fields.size() &&
                                      queuedOwned.size() == fields.size() && (!ghosts || queuedGhost.size() == fields.size()),
                                  "queued mean gradient provenance or halo intent is missing");
            for (index iField = 0; iField < index(values.size()); ++iField)
            {
                const auto &value = values[iField], &prepared = fields[iField];
                DNDS_check_throw_info(value.Space().Same(prepared.Space()) &&
                                          value.Owned().SharedStorage() == prepared.Owned().SharedStorage() &&
                                          value.Ghost().SharedStorage() == prepared.Ghost().SharedStorage() &&
                                          queuedOwned[iField][0].Matches(state.PhysicalField(iField).Owned()) &&
                                          queuedOwned[iField][1].Matches(state.PhysicalField(iField).Ghost()) &&
                                          queuedOwned[iField][2].Matches(value.Owned()) &&
                                          (!ghosts || queuedGhost[iField].Matches(value.Ghost())),
                                      "queued mean gradient source or output changed after submission");
            }
        }
        void RecordQueued(const EquationDofs &state, real atTime, bool ghosts)
        {
            if (stateBinding)
                stateBinding->Check(state);
            if (!ghosts)
            {
                DNDS_check_throw_info(!queuedSource, "mean gradient segment already pending");
                origin.reset();
                ghostsReady = false;
                queuedSource.emplace(state.NumericAlias());
                queuedTime = atTime;
                queuedOwned.clear();
                queuedGhost.clear();
                for (index iField = 0; iField < index(fields.size()); ++iField)
                    queuedOwned.push_back({IssuedArray(state.PhysicalField(iField).Owned()),
                                           IssuedArray(state.PhysicalField(iField).Ghost()), IssuedArray(fields[iField].Owned())});
            }
            else
            {
                DNDS_check_throw_info(queuedSource && queuedGhost.empty(), "mean gradient halo intent already recorded or no owned segment");
                CheckQueued(state, fields, owner.lock(), atTime, false);
                for (const auto &field : fields)
                    queuedGhost.emplace_back(field.Ghost());
            }
        }
        void PublishQueued(const EquationDofs &state)
        {
            DNDS_check_throw_info(queuedSource, "mean gradient segment is not pending");
            CheckQueued(state, fields, owner.lock(), queuedTime, true);
            for (index iField = 0; iField < index(fields.size()); ++iField)
            {
                const auto &field = fields[iField];
                for (const auto *array : {&state.PhysicalField(iField).Owned(), &state.PhysicalField(iField).Ghost(),
                                          &field.Owned(), &field.Ghost()})
                    DNDS_check_throw_info(array->Storage().IsIdle(), "mean gradient publication requires a completed segment");
            }
            origin.emplace(state.NumericAlias());
            time = queuedTime;
            sourceGhost.clear();
            ownedVersion.clear();
            ghostVersion.clear();
            ownedShape.clear();
            ghostShape.clear();
            for (index iField = 0; iField < index(fields.size()); ++iField)
            {
                sourceGhost.push_back(state.PhysicalField(iField).Ghost().Version());
                ownedVersion.push_back(fields[iField].Owned().Version());
                ghostVersion.push_back(fields[iField].Ghost().Version());
                ownedShape.push_back(fields[iField].Owned().ShapeVersion());
                ghostShape.push_back(fields[iField].Ghost().ShapeVersion());
            }
            ghostsReady = true;
            queuedSource.reset();
            queuedOwned.clear();
            queuedGhost.clear();
        }
        void Check(const EquationDofs &state, const std::vector<CFV::FVField> &values,
                   const std::shared_ptr<const void> &plan, real atTime, bool ghosts) const
        {
            DNDS_check_throw_info(owner.lock() == plan && origin && time == atTime && values.size() == fields.size(), "mean gradient publication is missing or belongs to another plan/time");
            origin->Check(state, true);
            DNDS_check_throw_info(!ghosts || ghostsReady, "mean gradient halo is not published");
            for (index iField = 0; iField < index(values.size()); ++iField)
            {
                const auto &a = values[iField], &b = fields[iField];
                DNDS_check_throw_info(a.Space().Same(b.Space()) && a.Owned().SharedStorage() == b.Owned().SharedStorage() &&
                                          a.Ghost().SharedStorage() == b.Ghost().SharedStorage() && a.Owned().Version() == ownedVersion[iField] &&
                                          a.Owned().ShapeVersion() == ownedShape[iField] && a.Ghost().ShapeVersion() == ghostShape[iField] &&
                                          state.PhysicalField(iField).Ghost().Version() == sourceGhost[iField] &&
                                          (!ghosts || a.Ghost().Version() == ghostVersion[iField]),
                                      "mean gradient source or output changed");
            }
        }
    };
    namespace AuxiliaryDetail
    {
        template <class Values>
        auto EquationAuxiliaryArrays(Values &values)
        {
            return std::array<const R::Array<real> *, 9>{values.meanThermodynamics ? &*values.meanThermodynamics : nullptr,
                                                         values.faceSpectralRadius ? &*values.faceSpectralRadius : nullptr,
                                                         values.faceWaveJump ? &*values.faceWaveJump : nullptr,
                                                         values.facePressureMinimum ? &*values.facePressureMinimum : nullptr,
                                                         values.cellSpectralSum ? &values.cellSpectralSum->Owned() : nullptr,
                                                         values.cellWaveJump ? &values.cellWaveJump->Owned() : nullptr,
                                                         values.cellPressureMinimum ? &values.cellPressureMinimum->Owned() : nullptr,
                                                         values.derivedOwned ? &values.derivedOwned->Values() : nullptr,
                                                         values.derivedGhost ? &values.derivedGhost->Values() : nullptr};
        }
    }
    struct AuxiliaryValues::EquationOrigin
    {
        struct Stamp
        {
            R::ArrayStorage::Shared storage;
            uint64_t version = 0, shape = 0;
        };
        std::weak_ptr<const void> owner;
        EquationDofOperations::Binding source;
        std::vector<uint64_t> ghostVersions;
        std::array<Stamp, 9> outputs;
        std::array<R::ArrayStorage::Shared, 3> ghosts;
        std::vector<R::Array<real>> gradientInputs;
        std::vector<std::array<uint64_t, 2>> gradientVersions;
        std::optional<BoundaryInputs> boundary;
        real time;
        EquationOrigin(std::weak_ptr<const void> plan, const EquationDofs &state, const AuxiliaryValues &values, real atTime,
                       std::optional<BoundaryInputs> trace)
            : owner(std::move(plan)), source(state.NumericAlias()), boundary(std::move(trace)), time(atTime)
        {
            for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                ghostVersions.push_back(state.PhysicalField(iField).Ghost().Version());
            const auto arrays = AuxiliaryDetail::EquationAuxiliaryArrays(values);
            for (int k = 0; k < 9; ++k)
                if (arrays[k])
                    outputs[k] = {arrays[k]->SharedStorage(), arrays[k]->Version(), arrays[k]->ShapeVersion()};
            int k = 0;
            for (const auto *field : {&values.cellSpectralSum, &values.cellWaveJump, &values.cellPressureMinimum})
                ghosts[k++] = *field ? (*field)->Ghost().SharedStorage() : nullptr;
        }
    };
}
