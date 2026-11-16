#include "Geom/Access/CellFaceIncidence.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "Geom/Unstructured/UnstructuredMeshSnapshot.hpp"

namespace DNDS::Geom
{
    namespace
    {
        template <class T>
        struct FrozenPair
        {
            Runtime::ArrayPair<T> values;
            typename Runtime::ArrayPair<T>::ReadLease read;
            FrozenPair(Runtime::ArrayPair<T> source, Runtime::Device device)
                : values(std::move(source)), read(values.ReadCombined({device, nullptr})) {}
            Runtime::PairReadView<T> View() const { return read.View(); }
            Runtime::ArrayPair<T> Placed(Runtime::Device device, Runtime::ExecutionContext context, const char *name) const
            {
                Runtime::ArrayPair<T> result{values.owned.To(device, context), values.ghost.To(device, context)};
                result.owned.SetName(std::string("Geom/cellFaceIncidences/") + name + "/owned", DNDS_ALLOCATION_SITE());
                result.ghost.SetName(std::string("Geom/cellFaceIncidences/") + name + "/ghost", DNDS_ALLOCATION_SITE());
                return result;
            }
            void Visit(const std::function<void(uint64_t, size_t, Runtime::Device)> &visitor) const
            {
                values.owned.SharedStorage()->VisitAllocations(visitor);
                values.ghost.SharedStorage()->VisitAllocations(visitor);
            }
        };
    }

    struct CellFaceIncidenceRead::Data
    {
        std::shared_ptr<UnstructuredMesh> source;
        Runtime::VersionStamp stamp;
        Runtime::Device device;
        FrozenPair<index> cellFaces, faceCells;
        FrozenPair<ElemInfo> faceInfo;
        std::optional<FrozenPair<NodePeriodicBits>> bits;
        CellFaceIncidenceView view;

        Data(std::shared_ptr<UnstructuredMesh> mesh, Runtime::VersionStamp generation, Runtime::Device where,
             Runtime::ArrayPair<index> cells, Runtime::ArrayPair<index> faces, Runtime::ArrayPair<ElemInfo> info,
             std::optional<Runtime::ArrayPair<NodePeriodicBits>> images)
            : source(std::move(mesh)), stamp(std::move(generation)), device(where),
              cellFaces(std::move(cells), where), faceCells(std::move(faces), where), faceInfo(std::move(info), where)
        {
            if (images)
                bits.emplace(std::move(*images), where);
            view = {cellFaces.View(), faceCells.View(), bits ? bits->View() : Runtime::PairReadView<NodePeriodicBits>{},
                    faceInfo.View(), bool(bits)};
            view.ValidateDevice(device);
        }
    };

    CellFaceIncidenceStorage CellFaceIncidenceStorage::Capture(std::shared_ptr<UnstructuredMesh> source)
    {
        DNDS_check_throw_info(source, "cell-face incidence capture requires a live mesh");
        // Normal semantic validation establishes ordered local incidences and
        // periodic side bits. This setup guard also excludes publication while
        // selecting shared handles from the authoritative host snapshot.
        const auto guard = source->AcquireRead(MeshReadRequirements::CellFlux());
        const auto native = source->CurrentNativeState();
        const auto &snapshot = native->Snapshot(Runtime::Device::Host());
        const auto &state = snapshot.Live();
        const auto &cells = state.relations[size_t(MeshRelation::CellFaces)];
        const auto &faces = state.relations[size_t(MeshRelation::FaceCells)];
        DNDS_check_throw_info(cells && faces && state.faceInfo &&
                                  (!state.semantics.isPeriodic || state.cellFacesPbi),
                              "canonical cell-face incidence publication is incomplete");
        auto data = std::make_shared<CellFaceIncidenceRead::Data>(source,
                                                                  source->versions.Stamp(Runtime::Change::Topology | Runtime::Change::Distribution | Runtime::Change::Layout | Runtime::Change::Interface),
                                                                  Runtime::Device::Host(), *cells, *faces, *state.faceInfo,
                                                                  state.semantics.isPeriodic ? state.cellFacesPbi : std::nullopt);
        return CellFaceIncidenceStorage(std::move(data));
    }

    CellFaceIncidenceStorage CellFaceIncidenceStorage::Placed(Runtime::Device device, Runtime::ExecutionContext context) const
    {
        DNDS_check_throw_info(_data && context.device == (device == Runtime::Device::Host() ? _data->device : device),
                              "cell-face incidence placement context differs");
        const auto guard = _data->source->versions.Read();
        _data->stamp.Validate();
        if (device == _data->device)
            return *this;
        std::optional<Runtime::ArrayPair<NodePeriodicBits>> images;
        if (_data->bits)
            images = _data->bits->Placed(device, context, "cellFaceBits");
        auto data = std::make_shared<CellFaceIncidenceRead::Data>(_data->source, _data->stamp, device,
                                                                  _data->cellFaces.Placed(device, context, "cellFaces"), _data->faceCells.Placed(device, context, "faceCells"),
                                                                  _data->faceInfo.Placed(device, context, "faceInfo"), std::move(images));
        return CellFaceIncidenceStorage(std::move(data));
    }

    CellFaceIncidenceRead::CellFaceIncidenceRead(std::shared_ptr<const Data> data)
        : _data(std::move(data)), _generation(_data->source->versions.Read())
    {
        _data->stamp.Validate();
    }
    CellFaceIncidenceView CellFaceIncidenceRead::View() const { return _data->view; }
    CellFaceIncidenceRead CellFaceIncidenceStorage::AcquireRead() const
    {
        DNDS_check_throw_info(_data, "cell-face incidence publication is empty");
        return CellFaceIncidenceRead(_data);
    }
    Runtime::Device CellFaceIncidenceStorage::GetDevice() const
    {
        DNDS_check_throw_info(_data, "cell-face incidence publication is empty");
        return _data->device;
    }
    void CellFaceIncidenceStorage::VisitAllocations(const std::function<void(uint64_t, size_t, Runtime::Device)> &visitor) const
    {
        DNDS_check_throw_info(_data, "cell-face incidence publication is empty");
        _data->cellFaces.Visit(visitor);
        _data->faceCells.Visit(visitor);
        _data->faceInfo.Visit(visitor);
        if (_data->bits)
            _data->bits->Visit(visitor);
    }
}
