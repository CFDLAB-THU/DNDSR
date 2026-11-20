#pragma once

#include "Cauchy/IO/BoundaryOutputGeometry.hpp"
#include "DNDS/Runtime/Communication/RowTransfer.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"

namespace DNDS::Cauchy
{
    struct BoundaryOutputGeometry
    {
        std::shared_ptr<Geom::UnstructuredMesh> mesh;
        // Local fast path stores rows in file order. When ownership differs,
        // store the compact FV rows and route their payloads to file order.
        Runtime::Array<BoundaryOutputRow> rows;
        std::optional<Runtime::RowTransferPlan> route;
        Runtime::Array<real> normals;
        std::shared_ptr<const std::vector<int>> zones;
        std::shared_ptr<const std::vector<std::pair<int, std::string>>> names;

        /// Redistribute a complete payload from parent-face ownership to file ownership.
        /// A writer on a duplicate communicator reuses the discovered peer row ordering.
        Runtime::Array<real> InFileOrder(Runtime::Array<real>, std::shared_ptr<Runtime::CommunicationContext>) const;
    };
}
