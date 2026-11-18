#pragma once

#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "DNDS/Runtime/Communication/CommunicationContext.hpp"

namespace DNDS::Cauchy
{
    /** @brief Detached numerical output with a retained immutable mesh read.
     * @details Owns prepared host cell fields and optional boundary flux rows. It
     * retains no solution/reconstruction/workspace or physics worker. Geometry
     * remains shared and read-pinned until writing ends; mesh mutation requires
     * the output service to drain first. Both files use reference-frame coordinates. Write performs collective
     * file I/O only, on the supplied retained communicator.
     */
    class PreparedVisualization
    {
        friend class FiniteVolume;
        struct Boundary
        {
            std::shared_ptr<Geom::UnstructuredMesh> mesh;
            Runtime::Array<real> fields, normals;
            std::shared_ptr<const std::vector<int>> zones;
            std::shared_ptr<const std::vector<std::pair<int, std::string>>> zoneNames;
            std::optional<Runtime::Array<real>> flux;
            std::vector<std::string> fluxNames;
        };
        struct Data;
        std::shared_ptr<const Data> _data;
        PreparedVisualization(std::shared_ptr<Geom::UnstructuredMesh>, Runtime::Array<real>,
                              std::vector<std::string>, Boundary, real time, int deflate);

    public:
        void Write(const std::string &prefix, std::shared_ptr<Runtime::CommunicationContext>) const;
    };
}
