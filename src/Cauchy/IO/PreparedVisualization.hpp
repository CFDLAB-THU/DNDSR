#pragma once

#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "DNDS/Runtime/Communication/CommunicationContext.hpp"

namespace DNDS::Cauchy
{
    struct BoundaryOutputGeometry;
    class OutputFields;
    /** @brief Detached numerical output with a retained immutable mesh read.
     * @details Owns detached resident cell fields and optional parent-face flux rows. It
     * retains no solution/reconstruction/workspace or physics worker. Geometry
     * remains shared and read-pinned until writing ends; mesh mutation requires
     * the output service to drain first. Both files use reference-frame coordinates. Write stages device fields, gathers boundary values, and performs
     * file redistribution and parallel HDF5 on the supplied communicator.
     */
    class PreparedVisualization
    {
        friend class FiniteVolume;
        friend class AsyncOutput;
        void BindWriterCopy(Runtime::ExecutionContext);
        struct Data;
        std::shared_ptr<Data> _data;
        PreparedVisualization(std::shared_ptr<Geom::UnstructuredMesh>, OutputFields,
                              std::vector<std::string>, std::shared_ptr<const BoundaryOutputGeometry>,
                              std::optional<Runtime::Array<real>>, std::vector<std::string>, real time, int deflate);

    public:
        void Write(const std::string &prefix, std::shared_ptr<Runtime::CommunicationContext>) const;
    };
}
