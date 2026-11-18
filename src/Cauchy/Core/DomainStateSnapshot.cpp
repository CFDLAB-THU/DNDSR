#include "Cauchy/Core/DomainStateSnapshot.hpp"
#include "Cauchy/Core/Domain.hpp"

namespace DNDS::Cauchy
{
    struct DomainStateSnapshot::Data
    {
        std::string name;
        EquationDofs means;
        std::optional<EquationDofs> reconstruction, previous;
        std::optional<Runtime::Array<real>> limitedVR;
        std::optional<EquationDofs::ReadScope> meanRead, reconstructionRead,
            previousRead;
        std::optional<Runtime::Array<real>::ReadLease> limitedVRRead;
        real time = 0, lastAcceptedDt = 0, previousDt = 0;
        StateEpoch epoch;
        std::string method;
        Json nonlinear, updatePolicy;
        std::optional<real> pressureGauge;
        bool sourceStrang = false;

        Data(std::string domainName, EquationDofs accepted,
             std::optional<EquationDofs> acceptedReconstruction,
             std::optional<Runtime::Array<real>> acceptedLimitedVR,
             std::optional<EquationDofs> previousState, real acceptedTime,
             real acceptedDt, real historyDt, StateEpoch acceptedEpoch,
             std::string timeMethod, Json nonlinearState, Json policy,
             bool strang, std::optional<real> gauge, Runtime::ExecutionContext context)
            : name(std::move(domainName)), means(std::move(accepted)),
              reconstruction(std::move(acceptedReconstruction)),
              previous(std::move(previousState)),
              limitedVR(std::move(acceptedLimitedVR)), time(acceptedTime),
              lastAcceptedDt(acceptedDt), previousDt(historyDt),
              epoch(acceptedEpoch), method(std::move(timeMethod)),
              nonlinear(std::move(nonlinearState)),
              updatePolicy(std::move(policy)), pressureGauge(gauge), sourceStrang(strang)
        {
            meanRead.emplace(means.ReadOwned(context));
            if (reconstruction)
                reconstructionRead.emplace(reconstruction->ReadOwned(context));
            if (limitedVR)
                limitedVRRead.emplace(limitedVR->Read(context));
            if (previous)
                previousRead.emplace(previous->ReadOwned(context));
            // These scopes pin the quiescent snapshot; they do not submit
            // numerical work. Consumers acquire their own operation reads.
            // Mark the already-observed prefix so eventual snapshot destruction
            // never needs a stream that its creating domain may have released.
            const auto complete = Runtime::Completion::FromExternalCompletion(context);
            meanRead->Complete(complete);
            if (reconstructionRead)
                reconstructionRead->Complete(complete);
            if (limitedVRRead)
                limitedVRRead->Complete(complete);
            if (previousRead)
                previousRead->Complete(complete);
        }
    };

    DomainStateSnapshot::DomainStateSnapshot(const Domain &domain,
                                             bool includeReconstruction)
    {
        domain._execution.Synchronize();
        std::optional<EquationDofs> reconstruction, previous;
        std::optional<Runtime::Array<real>> limitedVR;
        if (includeReconstruction &&
            domain._problem.finiteVolume.PolynomialReconstruction())
        {
            const auto &accepted = domain._state.restoreReconstruction
                                       ? *domain._state.reconstructionBackup
                                       : domain._problem.finiteVolume
                                             .ReconstructionGuess(
                                                 *domain._state.reconstruction);
            reconstruction.emplace(accepted.NumericAlias());
            if (domain._state.restoreReconstruction)
            {
                if (domain._state.limitedVRBackup)
                    limitedVR.emplace(*domain._state.limitedVRBackup);
            }
            else if (const auto history = domain._problem.finiteVolume
                                              .LimitedVRHistory(
                                                  *domain._state.reconstruction))
                limitedVR.emplace(*history);
        }
        const bool hasHistory = domain._state.timeStepper &&
                                domain._state.timeStepper->Method() == "bdf2" &&
                                domain._state.timeStepper->HasHistory();
        if (hasHistory)
            previous.emplace(
                domain._state.timeStepper->PreviousState().NumericAlias());
        const std::string method = domain._state.timeStepper
                                       ? domain._state.timeStepper->Method()
                                       : "ssprk3";
        const Json nonlinear = domain._state.timeStepper && method == "ditr"
                                   ? domain._state.coupledNonlinear->ToJson()
                                   : (domain._execution.implicit
                                          ? domain._execution.implicit->StateJson()
                                          : NonlinearSolveState{}.ToJson());
        _data = std::make_shared<Data>(
            domain._problem.name, domain._state.accepted.NumericAlias(),
            std::move(reconstruction), std::move(limitedVR),
            std::move(previous), domain._state.time,
            domain._state.lastAcceptedDt,
            hasHistory ? domain._state.timeStepper->PreviousDt() : 0,
            domain._state.epoch, method, nonlinear,
            domain.UpdatePolicyMetadata(), domain._sourceStrang,
            domain._state.pressureGauge ? std::optional<real>(*domain._state.pressureGauge) : std::nullopt,
            domain._execution.ComputeStream().Context());
    }

    DomainStateSnapshot DomainStateSnapshot::DetachedHost(Runtime::ExecutionContext context) const
    {
        const auto &source = *_data;
        const EquationSnapshotOptions allocation{source.means.Device().space == Runtime::MemorySpace::CUDA};
        std::optional<EquationDofs> reconstruction, previous;
        std::optional<Runtime::Array<real>> limitedVR;
        auto means = EquationDofOperations::CopyOwnedTo(source.means, Runtime::Device::Host(), context, allocation);
        means.Fields().SetName(source.name + "/outputSnapshot/means", DNDS_ALLOCATION_SITE());
        if (source.reconstruction)
        {
            reconstruction.emplace(EquationDofOperations::CopyOwnedTo(*source.reconstruction, Runtime::Device::Host(), context, allocation));
            reconstruction->Fields().SetName(source.name + "/outputSnapshot/reconstruction", DNDS_ALLOCATION_SITE());
        }
        if (source.previous)
        {
            previous.emplace(EquationDofOperations::CopyOwnedTo(*source.previous, Runtime::Device::Host(), context, allocation));
            previous->Fields().SetName(source.name + "/outputSnapshot/history", DNDS_ALLOCATION_SITE());
        }
        if (source.limitedVR)
        {
            const auto layout = source.limitedVR->Layout();
            limitedVR.emplace(allocation.pinnedHost
                                  ? Runtime::Array<real>::FromStorage(layout, Runtime::Buffer::AllocatePinnedHost(
                                                                                  CheckedSize::Multiply(layout.Elements(), sizeof(real))))
                                  : Runtime::Array<real>(layout));
            limitedVR->CopyFrom(*source.limitedVR, context).Wait();
            limitedVR->SetName(source.name + "/outputSnapshot/limitedVR", DNDS_ALLOCATION_SITE());
        }
        return DomainStateSnapshot(std::make_shared<Data>(source.name, std::move(means),
                                                          std::move(reconstruction), std::move(limitedVR), std::move(previous), source.time,
                                                          source.lastAcceptedDt, source.previousDt, source.epoch, source.method, source.nonlinear,
                                                          source.updatePolicy, source.sourceStrang, source.pressureGauge, Runtime::ExecutionContext::Host()));
    }

    const std::string &DomainStateSnapshot::Name() const { return _data->name; }
    const EquationDofs &DomainStateSnapshot::Means() const { return _data->means; }
    std::optional<real> DomainStateSnapshot::PressureGauge() const { return _data->pressureGauge; }
    OptionalRef<const EquationDofs> DomainStateSnapshot::Reconstruction() const
    {
        return _data->reconstruction
                   ? OptionalRef<const EquationDofs>(*_data->reconstruction)
                   : OptionalRef<const EquationDofs>{};
    }
    OptionalRef<const Runtime::Array<real>>
    DomainStateSnapshot::LimitedVRHistory() const
    {
        return _data->limitedVR
                   ? OptionalRef<const Runtime::Array<real>>(*_data->limitedVR)
                   : OptionalRef<const Runtime::Array<real>>{};
    }
    OptionalRef<const EquationDofs> DomainStateSnapshot::PreviousState() const
    {
        return _data->previous
                   ? OptionalRef<const EquationDofs>(*_data->previous)
                   : OptionalRef<const EquationDofs>{};
    }
    real DomainStateSnapshot::Time() const { return _data->time; }
    real DomainStateSnapshot::LastAcceptedDt() const
    {
        return _data->lastAcceptedDt;
    }
    real DomainStateSnapshot::PreviousDt() const { return _data->previousDt; }
    const StateEpoch &DomainStateSnapshot::Epoch() const { return _data->epoch; }
    const std::string &DomainStateSnapshot::TimeMethod() const
    {
        return _data->method;
    }
    const Json &DomainStateSnapshot::NonlinearState() const
    {
        return _data->nonlinear;
    }
    const Json &DomainStateSnapshot::UpdatePolicy() const
    {
        return _data->updatePolicy;
    }
    bool DomainStateSnapshot::SourceStrang() const
    {
        return _data->sourceStrang;
    }
}
