#include "Cauchy/IO/CheckpointMetadata.hpp"
#include "Cauchy/Core/Domain.hpp"
#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "DNDS/MPI/Failure.hpp"

namespace DNDS::Cauchy::CheckpointMetadata
{
    static void RemoveInactiveQCR(Json &model)
    {
        // This optional switch did not exist in early RANS checkpoints.
        // Only the inactive default normalizes away; true remains physical.
        if (model.is_object() && !model.value("useQCR", false))
            model.erase("useQCR");
    }
    nlohmann::json CompatibleEquationMetadata(Json value)
    {
        if (value.is_object() && value.contains("rans"))
            RemoveInactiveQCR(value["rans"]);
        return value;
    }
    static index CanonicalPeriodicZone(index zone)
    {
        if (zone == Geom::BC_ID_PERIODIC_1_DONOR)
            return Geom::BC_ID_PERIODIC_1;
        if (zone == Geom::BC_ID_PERIODIC_2_DONOR)
            return Geom::BC_ID_PERIODIC_2;
        if (zone == Geom::BC_ID_PERIODIC_3_DONOR)
            return Geom::BC_ID_PERIODIC_3;
        return zone;
    }
    Json EquationMetadata(const EquationSystem &equation)
    {
        const auto &schema = equation.Schema();
        Json blocks = Json::array();
        for (const auto &block : schema.blocks)
        {
            Json components = Json::array();
            for (const auto &component : block.components)
                components.push_back({{"name", component.name}, {"units", component.units}, {"conservation_group", component.conservationGroup}});
            blocks.push_back({{"name", block.name}, {"components", components}, {"vector_starts", block.vectorStarts}});
        }
        Json result{{"family", schema.family}, {"energy_convention", schema.energyConvention}, {"physical_accumulation", schema.physicalAccumulation}, {"blocks", blocks}};
        if (equation.IsACM())
        {
            const auto &model = equation.ACMParameters();
            result["eos"] = "none";
            result["acm"] = {{"alpha", model.alpha}, {"beta2", model.beta2}, {"kinematic_viscosity", model.kinematicViscosity}};
        }
        else if (equation.HasMixture())
        {
            const auto &material = equation.Mixture();
            const auto &scales = material.Scales();
            result["eos"] = "idealGasMixture";
            result["species"] = material.SpeciesNames();
            result["reference_scales"] = {scales.rho0, scales.U0, scales.L0, scales.T0};
            result["base_temperature"] = material.BaseTemperature();
            result["base_internal_energies"] = material.BaseSpecificInternalEnergies();
            result["model_description"] = material.ModelDescription();
        }
        else
        {
            result["eos"] = "perfectGas";
            result["gamma"] = equation.Gas().gamma;
            result["gas_constant"] = equation.Gas().gasConstant;
        }
        if (equation.HasRANS())
            result["rans"] = equation.RANS().Settings();
        return result;
    }
    void AgreeLoadMode(const MPIInfo &mpi, CheckpointLoadMode mode)
    {
        std::optional<int> candidate;
        if (mode == CheckpointLoadMode::Resume || mode == CheckpointLoadMode::MeansOnly)
            candidate = int(mode);
        DNDS_check_throw_info(MPI::AgreeOptionalInteger(mpi, candidate, 2),
                              "Cauchy checkpoint load mode is invalid or differs between ranks");
    }
    nlohmann::json CompatibleConfig(Json config, CheckpointLoadMode mode)
    {
        // Display/config JSON preserves insertion order. Semantic restart
        // equality must not: filling old defaults appends keys at another
        // position. Return ordinary object JSON for order-independent keys.
        // Time/output/placement/execution choices and polynomial setup do
        // not change stored means. Keep other physical/numerical choices
        // explicit: e.g. an old manual Roe stencil must not silently become
        // automatic. Raw non-root caller metadata is still compared exactly.
        if (!config.is_object() || !config.contains("domains") || !config["domains"].is_array())
            return config;
        config.erase("time");
        config.erase("output");
        for (auto &domain : config["domains"])
        {
            domain.erase("ranks");
            if (!domain.contains("settings") || !domain["settings"].is_object())
                continue;
            auto &settings = domain["settings"];
            // Gauge agreement is checked against the resolved target stored
            // in the checkpoint. A mean-only import can choose a new target.
            if (settings.contains("equation"))
                settings["equation"].erase("acmPressureMean");
            if (settings.contains("equation") && settings["equation"].value("system", std::string("compressible")) != "incompressibleACM")
            {
                // These new defaults have no meaning in older compressible files.
                settings["equation"].erase("acmAlpha");
                settings["equation"].erase("acmBeta2");
                settings["equation"].erase("acmEntropyFixRatio");
            }
            if (settings.value("initialTemperature", real(0)) == 0)
                settings.erase("initialTemperature");
            if (settings.contains("initialMassFractions") && settings["initialMassFractions"].is_object() && settings["initialMassFractions"].empty())
                settings.erase("initialMassFractions");
            if (settings.contains("equation") && settings["equation"].value("eos", std::string("perfectGas")) == "perfectGas")
                settings["equation"].erase("mixture");
            if (settings.contains("equation") && settings["equation"].contains("rans"))
                RemoveInactiveQCR(settings["equation"]["rans"]);
            settings.erase("execution");
            settings.erase("implicit");
            if (settings.contains("motion") && settings["motion"].value("angularVelocity", Vector3{}) == Vector3{} &&
                settings["motion"].value("translationVelocity", Vector3{}) == Vector3{})
                settings.erase("motion"); // Old fixed-frame archives had no motion section.
            if (settings.contains("boundaries") && settings["boundaries"].is_object())
                for (auto &condition : settings["boundaries"])
                {
                    if (condition.value("wallVelocityFrame", std::string("mesh")) == "mesh")
                        condition.erase("wallVelocityFrame");
                    // Old flow5 archives predate species boundary input.
                    // An inactive default is compatible; prescribed
                    // nonempty composition remains a physical difference.
                    if (condition.contains("massFractions") && condition["massFractions"].is_object() && condition["massFractions"].empty())
                        condition.erase("massFractions");
                }
            if (settings.contains("mesh") && settings["mesh"].is_object())
                settings["mesh"].erase("file");
            if (mode == CheckpointLoadMode::MeansOnly)
                settings.erase("numerics"); // The new solve initializes every numerical workspace from the means.
            if (settings.contains("numerics") && settings["numerics"].is_object())
            {
                auto &numerics = settings["numerics"];
                // Both the old positive default and 0 use the same sample floor.
                if (numerics.contains("wallDistanceFloor") && numerics["wallDistanceFloor"] == 0)
                    numerics["wallDistanceFloor"] = NumericsSettings::wallDistanceSafetyFloor;
                // Earlier archives predate the coarse convection switch.
                // Its false default preserves their operator; true remains
                // a deliberate numerical compatibility difference.
                if (!numerics.value("firstOrderConvection", false))
                    numerics.erase("firstOrderConvection");
                if (!numerics.value("ransMeanFaceValues", false))
                    numerics.erase("ransMeanFaceValues");
                if (!numerics.value("ransCellCenterViscosity", false))
                    numerics.erase("ransCellCenterViscosity");
                if (!numerics.value("ransP1Overlay", false))
                    numerics.erase("ransP1Overlay");
                if (!numerics.value("cellCenterWallDistanceForQuadrature", false))
                    numerics.erase("cellCenterWallDistanceForQuadrature");
                if (!numerics.value("firstLayerCenterWallDistance", false))
                    numerics.erase("firstLayerCenterWallDistance");
                const bool resolvedOmega = numerics.value("resolvedCellOmega", false);
                if (!resolvedOmega)
                    numerics.erase("resolvedCellOmega");
                if ((!numerics.contains("firstLayerCenterWallDistance") && !resolvedOmega) ||
                    numerics.value("firstLayerWallDistanceMethod", std::string{"cellCenter"}) == "cellCenter")
                    numerics.erase("firstLayerWallDistanceMethod");
                // Adding inactive limiter controls must not invalidate
                // older non-LVR checkpoints. Active LVR controls remain a
                // numerical compatibility requirement.
                if (numerics.value("limiter", "none") != "LVR")
                    numerics.erase("limitedVR");
                numerics.erase("cfv");
                // Retuning a reconstruction solve changes neither stored
                // means nor the compatible polynomial coefficient schema.
                numerics.erase("reconstructionIterations");
                numerics.erase("reconstructionTolerance");
                // Older archives predate this opt-in Euler wall mode.
                // An inactive option has no numerical compatibility effect.
                if (!numerics.value("noRsOnWall", false))
                    numerics.erase("noRsOnWall");
                // Missing viscous controls in older archives denote the
                // original primitive-cell-gradient scheme. Selecting the
                // polynomial operator is a visible numerical method change.
                const auto transport = settings.contains("equation") ? settings["equation"].value("transport", "inviscid") : "inviscid";
                if (transport == "NewtonianFourier")
                {
                    if (!numerics.contains("viscousGradient"))
                        numerics["viscousGradient"] = "primitive";
                    if (numerics["viscousGradient"] == "polynomial")
                    {
                        if (!numerics.contains("viscousLimited"))
                            numerics["viscousLimited"] = true;
                        if (!numerics.contains("noGRPOnWall"))
                            numerics["noGRPOnWall"] = false;
                    }
                    else
                    {
                        numerics.erase("viscousLimited");
                        numerics.erase("noGRPOnWall");
                    }
                }
                else
                {
                    // Newtonian ACM always uses polynomial velocity gradients.
                    // Its unlimited viscous input is an active operator choice.
                    numerics.erase("viscousGradient");
                    numerics.erase("noGRPOnWall");
                    if (transport != "Newtonian" || numerics.value("viscousLimited", true))
                        numerics.erase("viscousLimited");
                }
                // Earlier Cauchy archives used pointwise segment limiting.
                // Missing controls must not silently select Euler's newer
                // common-beta policy. Inactive PP choices do not affect a
                // non-PP continuation and need not invalidate old archives.
                if (numerics.value("positivity", false))
                {
                    if (!numerics.contains("positivityReconstruction"))
                        numerics["positivityReconstruction"] = "Segment";
                    // The old explicit positivityVolumePoints option describes
                    // a retired face-only/maximum-volume-grid policy. Do not
                    // invent that option for new archives or normalize away
                    // an active old policy difference: PP now follows RHS/source.
                    if (numerics["positivityReconstruction"] == "Segment")
                    {
                        numerics.erase("positivityCompressToMean");
                        numerics.erase("positivityBetaPower");
                    }
                    else
                    {
                        if (!numerics.contains("positivityCompressToMean"))
                            numerics["positivityCompressToMean"] = true;
                        if (!numerics.contains("positivityBetaPower"))
                            numerics["positivityBetaPower"] = 1;
                    }
                }
                else
                    for (const auto *name : {"positivityReconstruction", "positivityVolumePoints", "positivityCompressToMean", "positivityBetaPower"})
                        numerics.erase(name);
                if (numerics.contains("riemann") && numerics["riemann"].is_object() && !numerics["riemann"].contains("useStencil"))
                    numerics["riemann"]["useStencil"] = false;
            }
        }
        return config;
    }
    Json FrameMetadata(const FiniteVolume &fv, real time)
    {
        const auto frame = fv.Motion().At(time);
        std::array<real, 9> rotation;
        std::array<real, 3> translation;
        std::copy_n(frame.pose.rotation.data(), 9, rotation.data());
        std::copy_n(frame.pose.translation.data(), 3, translation.data());
        return {{"convention", "absolute-velocity-and-energy-in-mesh-axes"}, {"law", fv.Motion().Settings()}, {"rotation_column_major", rotation}, {"translation", translation}};
    }
    bool CompatibleFrame(const Json &metadata, const FiniteVolume &fv, real time)
    {
        if (!metadata.is_object() || !metadata.contains("frame"))
            return !fv.Motion().At(0).moving;
        try
        {
            const auto expected = FrameMetadata(fv, time);
            const auto &stored = metadata["frame"];
            if (stored.at("convention") != expected["convention"] || stored.at("law") != expected["law"])
                return false;
            for (const auto *name : {"rotation_column_major", "translation"})
            {
                const auto &a = stored.at(name), &b = expected.at(name);
                if (!a.is_array() || a.size() != b.size())
                    return false;
                for (index j = 0; j < index(a.size()); ++j)
                {
                    const real x = a[j].get<real>(), y = b[j].get<real>();
                    if (!std::isfinite(x) || std::abs(x - y) > 32 * std::numeric_limits<real>::epsilon() * std::max(real(1), std::abs(y)))
                        return false;
                }
            }
            return true;
        }
        catch (const Json::exception &)
        {
            MPI::AbortIfSolverFailure("Cauchy checkpoint validation failed");
            return false;
        }
    }
    index StoredRows(const Serializer::SerializerBaseSSP &serializer, const std::string &name)
    {
        const auto previous = serializer->GetCurrentPath();
        serializer->GoToPath(name);
        int format = 0, ranks = 0;
        serializer->ReadInt("runtime_array_format", format);
        serializer->ReadInt("writer_ranks", ranks);
        index rows = 0;
        if (format >= 3)
            serializer->ReadIndex("rows_global", rows);
        else
        {
            serializer->GoToPath("rows");
            std::vector<index> counts;
            auto region = Serializer::ArrayGlobalOffset{ranks, 0};
            serializer->ReadIndexVector("size", counts, region);
            for (const auto count : counts)
            {
                DNDS_check_throw_info(count >= 0 && count <= INT64_MAX - rows, "invalid restart row count");
                rows += count;
            }
        }
        serializer->GoToPath(previous);
        DNDS_check_throw_info(rows >= 0, "negative restart row count");
        return rows;
    }
    Runtime::Array<index> CellIds(const Domain &domain)
    {
        const auto &cells = domain.Discretization().Metrics().Cells();
        Runtime::Array<index> ids(Runtime::RowLayout::Dense(cells.NumOwned(), 1));
        auto write = ids.Write();
        for (index iCell = 0; iCell < cells.NumOwned(); ++iCell)
            write.View()(iCell, 0) = domain.Discretization().Mesh().OriginalEntityId(Geom::EntityKind::Cell, iCell);
        return ids;
    }
    // Exact per-owned-cell records, not a collision-prone hash. These are
    // restart metadata; the numerical flow payload is never packed/copied.
    MeshEnvelope DescribeMesh(const Domain &domain)
    {
        const auto &mesh = domain.Discretization().Mesh();
        // The metric read pins the same facade publication and its CellFlux
        // groups. Borrow only while this scope prevents mesh replacement.
        auto read = domain.Discretization().Metrics().AcquireRead({domain.Discretization().Device(), nullptr});
        const auto state = mesh.CurrentNativeState();
        const auto view = state->BorrowHostView();
        std::vector<size_t> topologySizes, coordinateSizes;
        for (index iCell = 0; iCell < view.NumCell(); ++iCell)
        {
            topologySizes.push_back(4 + 2 * view.CellNodes(iCell).size() + 3 * view.CellFaces(iCell).size());
            coordinateSizes.push_back(3 * view.CellNodes(iCell).size());
        }
        MeshEnvelope envelope{Runtime::Array<index>(Runtime::RowLayout::Ragged(topologySizes)),
                              Runtime::Array<real>(Runtime::RowLayout::Ragged(coordinateSizes)),
                              Runtime::Array<real>(Runtime::RowLayout::Dense(1, view.IsPeriodic() ? 45 : 0))};
        if (view.IsPeriodic())
        {
            auto transforms = envelope.periodicTransforms.Write();
            const auto periodic = view.Periodic();
            for (int axis = 1; axis <= 3; ++axis)
                for (int d = 0; d < 3; ++d)
                {
                    const int base = (axis - 1) * 15;
                    transforms.View()(0, base + d) = periodic.translation[axis].map()[d];
                    transforms.View()(0, base + 3 + d) = periodic.rotationCenter[axis].map()[d];
                    for (int k = 0; k < 3; ++k)
                        transforms.View()(0, base + 6 + 3 * d + k) = periodic.rotation[axis].map()(d, k);
                }
        }
        auto topology = envelope.topology.Write();
        auto coordinates = envelope.coordinates.Write();
        std::vector<Geom::NodePeriodicBits> faceBits;
        for (index iCell = 0; iCell < view.NumCell(); ++iCell)
        {
            auto row = topology.View()[iCell];
            const auto nodes = view.CellNodes(iCell), faces = view.CellFaces(iCell);
            auto element = view.GetCellElement(iCell);
            row[0] = index(view.GetCellInfo(iCell).getElemType());
            row[1] = view.GetCellZone(iCell);
            row[2] = nodes.size();
            row[3] = faces.size();
            for (rowsize slot = 0; slot < rowsize(nodes.size()); ++slot)
            {
                row[4 + 2 * slot] = mesh.OriginalEntityId(Geom::EntityKind::Node, nodes[slot]);
                row[5 + 2 * slot] = view.IsPeriodic() ? index(uint8_t(view.CellNodePeriodicBits(iCell)[slot])) : 0;
                const auto point = view.GetCoordNodeOnCell(iCell, slot);
                for (int d = 0; d < 3; ++d)
                    coordinates.View()(iCell, 3 * slot + d) = point[d];
            }
            for (rowsize slot = 0; slot < rowsize(faces.size()); ++slot)
            {
                const auto iFace = faces[slot], other = view.CellFaceOther(iCell, iFace, slot);
                const index offset = 4 + 2 * nodes.size() + 3 * slot;
                // A periodic face's main/donor zone and cell2face bits
                // describe its selected stored image. Native topology
                // selects that image from ordered parents; a new partition
                // may select the complementary image. Archive physical
                // axis identity and the cell's own face-node frame instead.
                row[offset] = CanonicalPeriodicZone(view.GetFaceZone(iFace));
                row[offset + 1] = other == UnInitIndex ? UnInitIndex : mesh.OriginalEntityId(Geom::EntityKind::Cell, other);
                uint8_t frame = 0;
                if (view.IsPeriodic())
                {
                    faceBits.resize(element.ObtainFace(slot).GetNumNodes());
                    element.ExtractFaceNodes(slot, view.CellNodePeriodicBits(iCell), faceBits);
                    frame = 7;
                    for (const auto bit : faceBits)
                        frame &= uint8_t(bit);
                }
                row[offset + 2] = frame;
            }
        }
        return envelope;
    }
    bool SameRows(const Runtime::Array<real> &left, const Runtime::Array<real> &right)
    {
        if (left.RowCount() != right.RowCount())
            return false;
        auto a = left.Read(), b = right.Read();
        for (index iRow = 0; iRow < left.RowCount(); ++iRow)
        {
            if (left.RowSize(iRow) != right.RowSize(iRow))
                return false;
            for (index j = 0; j < left.RowSize(iRow); ++j)
                if (a.View()(iRow, j) != b.View()(iRow, j))
                    return false;
        }
        return true;
    }
    std::string RowDifference(const Runtime::Array<real> &left, const Runtime::Array<real> &right, const Runtime::Array<index> &ids)
    {
        if (left.RowCount() != right.RowCount())
            return fmt::format("row counts {} != {}", left.RowCount(), right.RowCount());
        const auto a = left.Read(), b = right.Read();
        const auto labels = ids.Read();
        for (index iRow = 0; iRow < left.RowCount(); ++iRow)
        {
            if (left.RowSize(iRow) != right.RowSize(iRow))
                return fmt::format("cell {} widths {} != {}", labels.View()(iRow, 0), left.RowSize(iRow), right.RowSize(iRow));
            for (index j = 0; j < left.RowSize(iRow); ++j)
                if (a.View()(iRow, j) != b.View()(iRow, j))
                    return fmt::format("cell {} slot {}: {} != {}", labels.View()(iRow, 0), j, a.View()(iRow, j), b.View()(iRow, j));
        }
        return {};
    }
    std::string TopologyDifference(const Runtime::Array<index> &left, const Runtime::Array<index> &right,
                                   const Runtime::Array<index> &ids, bool legacyFaceImage)
    {
        if (left.RowCount() != right.RowCount())
            return "topology row counts differ";
        const auto a = left.Read(), b = right.Read(), labels = ids.Read();
        for (index iRow = 0; iRow < left.RowCount(); ++iRow)
        {
            if (left.RowSize(iRow) != right.RowSize(iRow))
                return fmt::format("cell {} topology widths differ", labels.View()(iRow, 0));
            const index faceStart = 4 + 2 * b.View()(iRow, 2);
            for (index j = 0; j < left.RowSize(iRow); ++j)
            {
                index stored = a.View()(iRow, j), current = b.View()(iRow, j);
                if (j >= faceStart)
                {
                    const index field = (j - faceStart) % 3;
                    if (field == 0)
                    {
                        stored = CanonicalPeriodicZone(stored);
                        current = CanonicalPeriodicZone(current);
                    }
                    if (field == 2 && legacyFaceImage)
                    {
                        // Old archives stored a partition-dependent face
                        // image gauge, not a physical mesh invariant. Cell
                        // node IDs/bits/coordinates, neighbor IDs, periodic
                        // axis and global transformations remain exact.
                        if (stored < 0 || stored > 7)
                            return "invalid legacy periodic image bits";
                        continue;
                    }
                }
                if (stored != current)
                    return fmt::format("cell {} slot {}: {} != {}", labels.View()(iRow, 0), j, stored, current);
            }
        }
        return {};
    }
}
