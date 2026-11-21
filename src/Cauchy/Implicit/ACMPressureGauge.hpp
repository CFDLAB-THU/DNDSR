#pragma once
#include "Cauchy/State/EquationDofs.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"

namespace DNDS::Cauchy::NonlinearOperatorDetail
{
    /** Add a uniform pressure offset (code units) to the owned ACM flow rows.
     * The caller determines the offset from the component's datum. Cell means
     * and receiver packet sections may have different row spaces. The operation
     * makes no collective call. Order subsequent reads and halo refreshes on
     * the supplied stream before publishing the changed values.
     */
    void SubmitPressureOffset(Runtime::EagerEngine &, const Runtime::ExecutionStream &, EquationDofs &, real offset);
}
