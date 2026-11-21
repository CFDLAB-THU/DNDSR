#pragma once

#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "DNDS/Runtime/Communication/CommunicationContext.hpp"

namespace DNDS::Cauchy
{
    struct BoundaryOutputGeometry;
    /** @brief Detached numerical output with a retained immutable mesh read.
     * @details The controller completes device-to-host staging on the producing
     * stream. The writer owns host fields and retains immutable geometry. It
     * gathers boundary rows in file order and writes both parallel HDF5 files.
     */
    class PreparedVisualization
    {
        friend class FiniteVolume;

        struct Data;
        std::shared_ptr<Data> _data;
        PreparedVisualization(std::shared_ptr<Geom::UnstructuredMesh>, Runtime::Array<real>,
                              std::vector<std::string>, std::shared_ptr<const BoundaryOutputGeometry>,
                              std::optional<Runtime::Array<real>>, std::vector<std::string>, real time, int deflate);

    public:
        void Write(const std::string &prefix, std::shared_ptr<Runtime::CommunicationContext>) const;
    };
}
