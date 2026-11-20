#include "Cauchy/IO/BoundaryOutputPlan.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    R::Array<real> BoundaryOutputGeometry::InFileOrder(R::Array<real> source, std::shared_ptr<R::CommunicationContext> io) const
    {
        if (!route)
            return source;
        const MPIInfo mpi(io->Communicator());
        std::optional<R::RowTransferPlan> writerRoute;
        if (route->Context() != io)
        {
            int relation = MPI_UNEQUAL;
            MPI_Comm_compare(route->Communicator(), mpi.comm, &relation);
            DNDS_check_throw_info(relation == MPI_IDENT || relation == MPI_CONGRUENT, "boundary output requires the mesh rank order");
            writerRoute.emplace(R::RowTransferPlan::Known(io, route->SourceRows(), route->DestinationRows(), route->SendRows(), route->ReceiveRows()));
        }
        const auto &active = writerRoute ? *writerRoute : *route;
        R::Array<real> destination;
        MPI::LocalStage(mpi, [&]
                        { destination = R::Array<real>(R::RowLayout::Dense(route->DestinationRows(), source.Layout().Width())); });
        R::PayloadPlan payload(active, source.Layout(), destination.Layout(), R::ElementSchema::For<real>());
        R::TransferOptions options;
        options.mode = R::TransferMode::Transient;
        options.context = io;
        payload.Bind(source, destination, R::TransferTransport::Direct, options).Begin().Wait();
        return destination;
    }
}
