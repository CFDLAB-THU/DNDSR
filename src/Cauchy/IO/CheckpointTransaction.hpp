#pragma once

#include "Cauchy/Core/DomainComponents.hpp"

namespace DNDS::Cauchy::IO
{
    /// Detached accepted-checkpoint publication transaction. Distributed file
    /// input and ID remapping may populate the trial allocation, but accepted
    /// state/reconstruction/history change only after every domain reserves.
    /// An abandoned destination writer is explicitly cancelled.
    struct DomainCheckpointRead
    {
        real time = 0, lastDt = 0;
        std::optional<real> pressureGauge;
        StateEpoch epoch;
        std::unique_ptr<EquationTimeStepper> stepper;
        std::optional<Runtime::Array<real>> guess, limitedVRHistory;
        std::optional<EquationDofs> equationGuess;
        std::optional<EquationDofs::ReadScope> source;
        std::optional<EquationDofs::WriteScope> destination;
        std::optional<FiniteVolume::ReconstructionRestore> reconstruction;
        bool committed = false;
        ~DomainCheckpointRead()
        {
            if (!committed && destination)
                destination->CancelUnmodified();
        }
    };

    /// Detached exact nonlinear/stage cursor. It is committed only after the
    /// corresponding accepted checkpoint transaction has published.
    struct DomainInternalCheckpointRead
    {
        EquationTimeStepper::ActiveCheckpoint time;
        NonlinearSolveState nonlinear;
        StateEpoch trialEpoch;
        real trialDt = 0, progressTime = 0;
        std::optional<EquationDofs> endpointWarm, endpointFinal;
        std::optional<Runtime::Array<real>> endpointHistory;
        std::optional<EquationDofs> middleWarm, middleFinal;
        std::optional<Runtime::Array<real>> middleHistory;
        std::array<std::optional<Runtime::Array<real>>, 2> coupledSourceDerivative;
        bool ditr = false;
    };
}
