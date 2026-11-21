#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"

namespace DNDS::Cauchy
{
    using Workspace = FiniteVolume::Workspace;

    Workspace::ArrayPublication::ArrayPublication(const R::Array<real> &a)
        : storage(a.SharedStorage()), shape(a.ShapeVersion()), version(a.Version()) {}

    bool Workspace::ArrayPublication::Current(const R::Array<real> &a) const
    {
        return storage.lock() == a.SharedStorage() && shape == a.ShapeVersion() && version == a.Version();
    }

    Workspace::PacketPublication::PacketPublication(const BoundaryInputs &packet)
        : publication(packet.Published())
    {
        arrays.reserve(packet.FaceCenterValues().PhysicalFieldNumber() + packet.QuadratureValues().PhysicalFieldNumber());
        for (const auto *section : {&packet.FaceCenterValues(), &packet.QuadratureValues()})
            for (index f = 0; f < section->PhysicalFieldNumber(); ++f)
                arrays.emplace_back(section->PhysicalField(f).Owned());
    }

    bool Workspace::PacketPublication::Current(const BoundaryInputs &packet) const
    {
        const auto current = packet.Published();
        if (!current || current != publication.lock() ||
            arrays.size() != size_t(packet.FaceCenterValues().PhysicalFieldNumber() + packet.QuadratureValues().PhysicalFieldNumber()))
            return false;
        size_t i = 0;
        for (const auto *section : {&packet.FaceCenterValues(), &packet.QuadratureValues()})
            for (index f = 0; f < section->PhysicalFieldNumber(); ++f)
                if (!arrays[i++].Current(section->PhysicalField(f).Owned()))
                    return false;
        return true;
    }

    Workspace::DofPublication::DofPublication(const EquationDofs &values)
    {
        fields.reserve(values.PhysicalFieldNumber());
        for (index f = 0; f < values.PhysicalFieldNumber(); ++f)
        {
            const auto &field = values.PhysicalField(f);
            fields.push_back({ArrayPublication(field.Owned()), ArrayPublication(field.Ghost())});
        }
        blocks.reserve(values.BlockNumber());
        for (index b = 0; b < values.BlockNumber(); ++b)
            blocks.emplace_back(values.LocateBlock(b));
    }

    bool Workspace::DofPublication::Current(const EquationDofs &values) const
    {
        if (size_t(values.PhysicalFieldNumber()) != fields.size() || size_t(values.BlockNumber()) != blocks.size())
            return false;
        for (index f = 0; f < values.PhysicalFieldNumber(); ++f)
        {
            const auto &field = values.PhysicalField(f);
            if (!fields[f][0].Current(field.Owned()) || !fields[f][1].Current(field.Ghost()))
                return false;
        }
        for (index b = 0; b < values.BlockNumber(); ++b)
        {
            const auto live = values.LocateBlock(b);
            if (live.physicalField != blocks[b].physicalField || live.firstComponent != blocks[b].firstComponent ||
                live.componentNumber != blocks[b].componentNumber)
                return false;
        }
        return true;
    }

    Workspace::SpatialInputs::SpatialInputs(const EquationDofs &state, const std::optional<BoundaryInputs> &packet, real t)
        : mean(state), time(t)
    {
        if (packet)
            boundary.emplace(*packet);
    }

    bool Workspace::SpatialInputs::Current(const EquationDofs &state, const std::optional<BoundaryInputs> &packet, real t) const
    {
        return t == time && mean.Current(state) && bool(packet) == bool(boundary) &&
               (!packet || boundary->Current(*packet));
    }

    void Workspace::CheckPreparedResidual(const EquationDofs &state, const ReconstructionState &reconstruction, const std::optional<FinalizedReconstruction> &finalized, real time) const
    {
        DNDS_check_throw_info(rhsState && rhsState->Current(state, incoming, time),
                              "prepared residual means, halo, interface packet or stage time changed");
        DNDS_check_throw_info(bool(rhsCoefficients) == bool(finalized) && (!finalized || rhsCoefficients->Current(finalized->Values())),
                              "prepared residual reconstruction changed");
        DNDS_check_throw_info(!rhsUnlimitedViscousCoefficients || rhsUnlimitedViscousCoefficients->Current(reconstruction.WarmValues()),
                              "prepared residual unlimited viscous reconstruction changed");
    }

    void Workspace::DirectGradientPublication::Publish(const Data &plan, const R::Array<real> &meanValues, const R::Array<real> &inputValues,
                                                       const R::Array<real> &outputValues, const std::optional<BoundaryInputs> &packet, real stageTime)
    {
        const auto meanRead = meanValues.Read({meanValues.GetDevice(), nullptr});
        const auto inputRead = inputValues.Read({inputValues.GetDevice(), nullptr});
        const auto outputRead = outputValues.Read({outputValues.GetDevice(), nullptr});
        mean.emplace(meanValues);
        input.emplace(inputValues);
        output.emplace(outputValues);
        geometry = plan.boundaryLayout->stamp;
        boundary.reset();
        if (plan.hasInterfaces)
        {
            DNDS_check_throw_info(packet && packet->Time() == stageTime, "direct gradient publication needs a current interface packet");
            const auto centers = packet->FaceCenterValues().ReadOwned({meanValues.GetDevice(), nullptr});
            const auto quadrature = packet->QuadratureValues().ReadOwned({meanValues.GetDevice(), nullptr});
            boundary.emplace(*packet);
        }
        time = stageTime;
        reconstruction = plan.settings.numerics.reconstruction;
        gradient = plan.settings.numerics.gradient;
        limiter = plan.settings.numerics.limiter;
    }

    bool Workspace::DirectGradientPublication::Current(const Data &plan, const R::Array<real> &meanValues, const R::Array<real> &inputValues,
                                                       const R::Array<real> &outputValues, const std::optional<BoundaryInputs> &packet, real stageTime) const
    {
        const auto meanRead = meanValues.Read({meanValues.GetDevice(), nullptr});
        const auto inputRead = inputValues.Read({inputValues.GetDevice(), nullptr});
        const auto outputRead = outputValues.Read({outputValues.GetDevice(), nullptr});
        if (time != stageTime || !geometry.Valid() || reconstruction != plan.settings.numerics.reconstruction ||
            gradient != plan.settings.numerics.gradient || limiter != plan.settings.numerics.limiter ||
            !mean || !mean->Current(meanValues) || !input->Current(inputValues) || !output->Current(outputValues))
            return false;
        if (!plan.hasInterfaces)
            return !boundary;
        if (!packet || !boundary || packet->Time() != stageTime)
            return false;
        const auto centers = packet->FaceCenterValues().ReadOwned({meanValues.GetDevice(), nullptr});
        const auto quadrature = packet->QuadratureValues().ReadOwned({meanValues.GetDevice(), nullptr});
        return boundary->Current(*packet);
    }
}
