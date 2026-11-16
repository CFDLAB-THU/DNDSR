#pragma once

#include "Cauchy/Discretization/FiniteVolumeStencil.hpp"
#include "DNDS/Runtime/Execution/KernelArguments.hpp"
#include <algorithm>

namespace CauchyTest
{
    namespace CompactR = DNDS::Runtime;
    using CompactIndex = DNDS::index;

    /** Numerical-leaf fixture with the production compact topology/metric
     * representation. It does not pretend to validate mesh construction.
     * Inputs and expected arithmetic are supplied independently by each test.
     * It retains every array/lease and a single resident POD descriptor; no
     * persistent expanded sample/rotation table enters the tested leaves.
     */
    class CompactStencilFixture
    {
    public:
        struct Input
        {
            std::vector<std::vector<CompactIndex>> cellFaces;
            std::vector<std::vector<uint8_t>> incidenceBits;
            std::vector<std::array<CompactIndex, 2>> faceCells;
            std::vector<int> zones;
            std::vector<DNDS::real> cells, faceMetrics, quadrature, distances;
            std::vector<CompactIndex> faceOffsets;
            std::vector<uint8_t> codes;
            DNDS::CFV::FacePeriodicParameters parameters;
            std::vector<DNDS::Cauchy::FaceRouting> routing;
            std::vector<CompactIndex> coreCells, dependentCells, coreFaces, dependentFaces;
            bool quadratureBoundaryValues = true;
        };

    private:
        CompactR::Array<CompactIndex> _cellFaces, _faceCells, _faceOffsets, _incidenceOffsets,
            _columns, _active, _auxiliaryCellFaces, _coreCells, _dependentCells, _coreFaces, _dependentFaces;
        CompactR::Array<DNDS::Geom::NodePeriodicBits> _bits;
        CompactR::Array<DNDS::Geom::ElemInfo> _faceInfo;
        CompactR::Array<DNDS::real> _cells, _faceMetrics, _quadrature, _distances, _neighbors, _radial;
        CompactR::Array<uint8_t> _codes;
        CompactR::Array<DNDS::CFV::FacePeriodicParameters> _parameters;
        CompactR::Array<DNDS::Cauchy::FaceRouting> _routing;
        CompactR::Array<CompactIndex> _empty{CompactR::RowLayout::Dense(0, 1)};
        std::vector<std::shared_ptr<void>> _readOwners;
        DNDS::Cauchy::FiniteVolumeStencilView _view;
        CompactR::MetadataBuffer<DNDS::Cauchy::FiniteVolumeStencilView> _descriptor;
        CompactR::ExecutionContext _context;

        template <class T, class Policy>
        auto Capture(const CompactR::Array<T> &array)
        {
            auto read = std::make_shared<typename CompactR::Array<T>::ReadLease>(array.Read(_context));
            const auto view = read->template View<Policy>();
            _readOwners.push_back(std::move(read));
            return view;
        }
        template <class T>
        static CompactR::Array<T> Dense(CompactIndex width, const std::vector<T> &values)
        {
            DNDS_assert(width > 0 && values.size() % width == 0);
            CompactR::Array<T> array(CompactR::RowLayout::Dense(CompactIndex(values.size()) / width, width));
            auto write = array.Write();
            const auto out = write.template View<CompactR::TableRows>();
            for (CompactIndex iRow = 0; iRow < out.RowCount(); ++iRow)
                for (CompactIndex j = 0; j < width; ++j)
                    out(iRow, j) = values[iRow * width + j];
            return array;
        }

    public:
        explicit CompactStencilFixture(Input input, CompactR::ExecutionContext context = CompactR::ExecutionContext::Host())
            : _context(context)
        {
            const CompactIndex owned = CompactIndex(input.cellFaces.size()), faces = CompactIndex(input.faceCells.size());
            DNDS_assert(input.cells.size() % 8 == 0 && input.faceMetrics.size() == faces * 7 &&
                        input.faceOffsets.size() == faces + 1 && input.quadrature.size() == input.faceOffsets.back() * 7);
            std::vector<size_t> lengths;
            std::vector<CompactIndex> offsets{0}, columns(faces * 2, DNDS::UnInitIndex), parents, active;
            std::vector<DNDS::Geom::ElemInfo> info;
            for (const auto &row : input.cellFaces)
            {
                lengths.push_back(row.size());
                offsets.push_back(offsets.back() + CompactIndex(row.size()));
            }
            _cellFaces = CompactR::Array<CompactIndex>(CompactR::RowLayout::Ragged(lengths));
            _bits = CompactR::Array<DNDS::Geom::NodePeriodicBits>(CompactR::RowLayout::Ragged(lengths));
            {
                auto write = _cellFaces.Write();
                auto bits = _bits.Write();
                for (CompactIndex iCell = 0; iCell < owned; ++iCell)
                {
                    std::vector<CompactIndex> neighbors;
                    for (CompactIndex slot = 0; slot < CompactIndex(input.cellFaces[iCell].size()); ++slot)
                    {
                        const auto iFace = input.cellFaces[iCell][slot];
                        const auto pair = input.faceCells[iFace];
                        const uint8_t bit = input.incidenceBits.empty() ? 0 : input.incidenceBits[iCell][slot];
                        const int side = pair[0] == pair[1] ? (bit ? 1 : 0) : (pair[0] == iCell ? 0 : 1);
                        const auto other = pair[1 - side];
                        if (other != DNDS::UnInitIndex && other != iCell)
                            neighbors.push_back(other);
                        write.View()(iCell, slot) = iFace;
                        bits.View()(iCell, slot) = DNDS::Geom::NodePeriodicBits{bit};
                    }
                    std::sort(neighbors.begin(), neighbors.end());
                    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
                    for (CompactIndex slot = 0; slot < CompactIndex(input.cellFaces[iCell].size()); ++slot)
                    {
                        const auto iFace = input.cellFaces[iCell][slot];
                        const auto pair = input.faceCells[iFace];
                        const uint8_t bit = input.incidenceBits.empty() ? 0 : input.incidenceBits[iCell][slot];
                        const int side = pair[0] == pair[1] ? (bit ? 1 : 0) : (pair[0] == iCell ? 0 : 1);
                        const auto other = pair[1 - side];
                        if (other != DNDS::UnInitIndex && other != iCell)
                            columns[2 * iFace + side] = std::lower_bound(neighbors.begin(), neighbors.end(), other) - neighbors.begin();
                    }
                }
            }
            for (CompactIndex iFace = 0; iFace < faces; ++iFace)
            {
                parents.insert(parents.end(), input.faceCells[iFace].begin(), input.faceCells[iFace].end());
                active.push_back(iFace);
                info.push_back({DNDS::Geom::Elem::Line2, input.zones.empty() ? DNDS::Geom::INTERNAL_ZONE : input.zones[iFace]});
            }
            if (input.routing.empty())
                for (CompactIndex iFace = 0; iFace < faces; ++iFace)
                    input.routing.push_back({iFace, 0, DNDS::UnInitIndex, DNDS::UnInitIndex, DNDS::UnInitIndex});
            if (input.codes.empty())
                input.codes.assign(faces, 0);
            if (input.distances.empty())
                input.distances.assign(input.faceOffsets.back(), 1);
            std::vector<CompactIndex> auxiliaryCellFaces;
            for (const auto &row : input.cellFaces)
            {
                std::vector<CompactIndex> auxiliary;
                for (const auto iFace : row)
                    auxiliary.push_back(input.routing[iFace].auxiliaryFace);
                std::sort(auxiliary.begin(), auxiliary.end());
                auxiliaryCellFaces.insert(auxiliaryCellFaces.end(), auxiliary.begin(), auxiliary.end());
            }
            _auxiliaryCellFaces = Dense<CompactIndex>(1, auxiliaryCellFaces);
            _faceCells = Dense<CompactIndex>(2, parents);
            _faceInfo = Dense<DNDS::Geom::ElemInfo>(1, info);
            _faceOffsets = Dense<CompactIndex>(1, input.faceOffsets);
            _incidenceOffsets = Dense<CompactIndex>(1, offsets);
            _columns = Dense<CompactIndex>(2, columns);
            _active = Dense<CompactIndex>(1, active);
            _cells = Dense<DNDS::real>(8, input.cells);
            _faceMetrics = Dense<DNDS::real>(7, input.faceMetrics);
            _quadrature = Dense<DNDS::real>(7, input.quadrature);
            _distances = Dense<DNDS::real>(1, input.distances);
            _codes = Dense<uint8_t>(1, input.codes);
            _parameters = Dense<DNDS::CFV::FacePeriodicParameters>(1, {input.parameters});
            _routing = Dense<DNDS::Cauchy::FaceRouting>(1, input.routing);
            _coreCells = Dense<CompactIndex>(1, input.coreCells);
            _dependentCells = Dense<CompactIndex>(1, input.dependentCells);
            _coreFaces = Dense<CompactIndex>(1, input.coreFaces);
            _dependentFaces = Dense<CompactIndex>(1, input.dependentFaces);
            _radial = CompactR::Array<DNDS::real>(CompactR::RowLayout::Dense(0, 3));
            _neighbors = CompactR::Array<DNDS::real>(CompactR::RowLayout::Dense(faces, 6));
            {
                const auto codes = _codes.Read();
                const auto parameters = _parameters.Read();
                const DNDS::CFV::FacePeriodicView periodic{codes.View<CompactR::FixedRows<1>>(), parameters.View<CompactR::FixedRows<1>>()};
                auto write = _neighbors.Write();
                for (CompactIndex iFace = 0; iFace < faces; ++iFace)
                    for (int side = 0; side < 2; ++side)
                    {
                        const auto pair = input.faceCells[iFace];
                        const auto other = pair[1 - side] == DNDS::UnInitIndex ? pair[side] : pair[1 - side];
                        const auto point = periodic.OtherPoint(iFace, side,
                                                               DNDS::Geom::tPoint{input.cells[8 * other + 1], input.cells[8 * other + 2], input.cells[8 * other + 3]});
                        for (int d = 0; d < 3; ++d)
                            write.View()(iFace, 3 * side + d) = point[d];
                    }
            }
            const auto place = [&](auto &array)
            {
                if (context.device != CompactR::Device::Host())
                    array = array.To(context.device, context);
            };
            place(_cellFaces);
            place(_faceCells);
            place(_faceInfo);
            place(_bits);
            place(_faceOffsets);
            place(_incidenceOffsets);
            place(_columns);
            place(_active);
            place(_auxiliaryCellFaces);
            place(_cells);
            place(_faceMetrics);
            place(_quadrature);
            place(_distances);
            place(_codes);
            place(_parameters);
            place(_routing);
            place(_coreCells);
            place(_dependentCells);
            place(_coreFaces);
            place(_dependentFaces);
            place(_neighbors);
            place(_radial);
            place(_empty);
            _view.topology = {{Capture<CompactIndex, CompactR::RowLayout>(_cellFaces), {}},
                              {Capture<CompactIndex, CompactR::RowLayout>(_faceCells), {}},
                              {Capture<DNDS::Geom::NodePeriodicBits, CompactR::RowLayout>(_bits), {}},
                              {Capture<DNDS::Geom::ElemInfo, CompactR::RowLayout>(_faceInfo), {}},
                              true};
            _view.metric.cells = Capture<DNDS::real, CompactR::FixedRows<8>>(_cells);
            _view.metric.faceOffsets = Capture<CompactIndex, CompactR::FixedRows<1>>(_faceOffsets);
            _view.metric.quadrature = Capture<DNDS::real, CompactR::FixedRows<7>>(_quadrature);
            _view.metric.radialMoment = Capture<DNDS::real, CompactR::FixedRows<3>>(_radial);
            _view.metric.faceMean = Capture<DNDS::real, CompactR::FixedRows<7>>(_faceMetrics);
            _view.metric.neighborCenters = Capture<DNDS::real, CompactR::FixedRows<6>>(_neighbors);
            _view.metric.periodic = {Capture<uint8_t, CompactR::FixedRows<1>>(_codes), Capture<DNDS::CFV::FacePeriodicParameters, CompactR::FixedRows<1>>(_parameters)};
            _view.routing = Capture<DNDS::Cauchy::FaceRouting, CompactR::FixedRows<1>>(_routing);
            _view.columns = Capture<CompactIndex, CompactR::FixedRows<2>>(_columns);
            _view.viscousDistance = Capture<DNDS::real, CompactR::FixedRows<1>>(_distances);
            _view.jacobianOffsets = Capture<CompactIndex, CompactR::FixedRows<1>>(_incidenceOffsets);
            _view.stencilFaces = Capture<CompactIndex, CompactR::FixedRows<1>>(_active);
            _view.coreCells = Capture<CompactIndex, CompactR::FixedRows<1>>(_coreCells);
            _view.dependentCells = Capture<CompactIndex, CompactR::FixedRows<1>>(_dependentCells);
            _view.rhsCoreFaces = _view.auxiliaryCoreFaces = Capture<CompactIndex, CompactR::FixedRows<1>>(_coreFaces);
            _view.rhsDependentFaces = _view.auxiliaryDependentFaces = Capture<CompactIndex, CompactR::FixedRows<1>>(_dependentFaces);
            const auto empty = Capture<CompactIndex, CompactR::FixedRows<1>>(_empty);
            _view.auxiliaryCellOffsets = _view.jacobianOffsets;
            _view.auxiliaryCellFaces = Capture<CompactIndex, CompactR::FixedRows<1>>(_auxiliaryCellFaces);
            _view.auxiliaryInterfaceIndependentFaces = _view.auxiliaryInterfaceFaces = empty;
            _view.interfaceIndependentCells = _view.interfaceCoreCells = empty;
            _view.quadratureBoundaryValues = input.quadratureBoundaryValues;
            _descriptor = CompactR::MetadataBuffer<DNDS::Cauchy::FiniteVolumeStencilView>({_view});
            if (context.device != CompactR::Device::Host())
                _descriptor = _descriptor.CopyTo(context.device, context);
        }
        const DNDS::Cauchy::FiniteVolumeStencilView &View() const { return _view; }
        auto Descriptor() const
        {
            const auto values = _descriptor.View(_context.device);
            return CompactR::ArrayView<const DNDS::Cauchy::FiniteVolumeStencilView,
                                       CompactR::AccessBoundsChecks, CompactR::FixedRows<1>>{
                values.data(), nullptr, CompactIndex(values.size()), 1, 1, nullptr,
                CompactR::RowLayoutKind::Table, _context.device};
        }
    };
}
