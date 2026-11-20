#include "DNDS/Runtime/Communication/Exchange.hpp"

namespace DNDS::Runtime
{
    namespace
    {
        std::vector<RowRequest> Requests(const std::vector<RemoteRow> &leaves)
        {
            std::vector<RowRequest> requests;
            for (size_t i = 0; i < leaves.size(); ++i)
                requests.push_back({leaves[i].rank, leaves[i].row, index(i)});
            return requests;
        }

    }
    ExchangePlan::ExchangePlan(MPI_Comm communicator, size_t localRoots, const std::vector<RemoteRow> &leaves,
                               std::vector<VersionStamp> dependencies)
        : ExchangePlan(std::make_shared<CommunicationContext>(communicator), localRoots, leaves, std::move(dependencies)) {}
    ExchangePlan::ExchangePlan(std::shared_ptr<CommunicationContext> context, size_t localRoots, const std::vector<RemoteRow> &leaves,
                               std::vector<VersionStamp> dependencies)
        : _route(std::move(context), localRoots, leaves.size(), Requests(leaves), std::move(dependencies)) {}

    ExchangePlan::Operation ExchangePlan::BeginCopy(const Buffer &roots, Buffer &leaves, size_t rowBytes) const
    {
        _route.Context()->Preflight(rowBytes && roots.GetDevice() == Device::Host() && leaves.GetDevice() == Device::Host(),
                                    "prototype Exchange facade requires host arrays and positive row width");
        return PayloadPlan(_route, RowLayout::Dense(_route.SourceRows(), rowBytes),
                           RowLayout::Dense(_route.DestinationRows(), rowBytes), ElementSchema::Arithmetic<uint8_t>())
            .Bind(roots, leaves)
            .Begin();
    }

    ExchangePlan::Operation ExchangePlan::BeginAdd(const Buffer &leaves, Buffer &roots, size_t components) const
    {
        _route.Context()->Preflight(components && roots.GetDevice() == Device::Host() && leaves.GetDevice() == Device::Host(),
                                    "prototype Exchange facade requires host arrays and positive row width");
        auto reverse = _route.Reverse();
        return PayloadPlan(reverse, RowLayout::Dense(reverse.SourceRows(), components),
                           RowLayout::Dense(reverse.DestinationRows(), components), ElementSchema::Arithmetic<double>())
            .Bind(leaves, roots, Reducer::Builtin<double>(Reduction::Sum))
            .Begin();
    }
}
