#pragma once

#include "DNDS/Core/Defines.hpp"
#include <algorithm>

namespace DNDS::Cauchy
{
    /// Physical-time accumulation selected once by the equation adapter.
    /// Pseudo-time preconditioning is a separate spatial/nonlinear policy.
    enum class PhysicalAccumulationKind
    {
        Identity,
        ConstantDensityVelocityPressure
    };

    struct ComponentSchema
    {
        std::string name, units, conservationGroup;
    };
    struct EquationBlockSchema
    {
        std::string name;
        std::vector<ComponentSchema> components;
        /// Physical polar-vector triples, independent of geometric dimension.
        std::vector<index> vectorStarts;
    };
    /// Fully qualified logical-component identity used at equation boundaries.
    /// The vector metadata is part of compatibility: a component with the same
    /// name but different frame-transform semantics is not interchangeable.
    struct LogicalComponentSchema
    {
        std::string block, name, units, conservationGroup;
        index vectorStart = -1, vectorComponent = -1;

        std::string QualifiedName() const { return block + "." + name; }
        bool operator==(const LogicalComponentSchema &other) const
        {
            return block == other.block && name == other.name &&
                   units == other.units &&
                   conservationGroup == other.conservationGroup &&
                   vectorStart == other.vectorStart &&
                   vectorComponent == other.vectorComponent;
        }
        bool operator!=(const LogicalComponentSchema &other) const
        {
            return !(*this == other);
        }
    };
    /// Logical component meaning and order, independent of physical arrays.
    /// Build before binding state; published bindings keep an immutable copy.
    struct EquationSchema
    {
        std::string family, energyConvention, physicalAccumulation;
        std::vector<EquationBlockSchema> blocks;

        PhysicalAccumulationKind AccumulationKind() const
        {
            if (physicalAccumulation == "conservative-volume" || physicalAccumulation == "identity")
                return PhysicalAccumulationKind::Identity;
            if (physicalAccumulation == "constant-density-velocity-pressure")
                return PhysicalAccumulationKind::ConstantDensityVelocityPressure;
            DNDS_check_throw_info(false, "unknown equation physical accumulation: " + physicalAccumulation);
            return PhysicalAccumulationKind::Identity;
        }

        /// Flatten the schema without referring to physical Array grouping.
        /// This is a setup/publication operation; no numerical loop calls it.
        std::vector<LogicalComponentSchema> LogicalComponents() const
        {
            std::vector<LogicalComponentSchema> result;
            for (const auto &block : blocks)
            {
                for (const index start : block.vectorStarts)
                    DNDS_check_throw_info(
                        start >= 0 &&
                            start + 3 <= index(block.components.size()),
                        "equation vector declaration exceeds block " +
                            block.name);
                for (index j = 0; j < index(block.components.size()); ++j)
                {
                    index vectorStart = -1, vectorComponent = -1;
                    for (const index start : block.vectorStarts)
                        if (j >= start && j < start + 3)
                        {
                            DNDS_check_throw_info(
                                vectorStart < 0,
                                "overlapping equation vector declarations in block " +
                                    block.name);
                            vectorStart = start;
                            vectorComponent = j - start;
                        }
                    const auto &component = block.components[size_t(j)];
                    result.push_back({block.name, component.name,
                                      component.units,
                                      component.conservationGroup,
                                      vectorStart, vectorComponent});
                }
            }
            return result;
        }

        /// Map each source logical ordinal to a target ordinal by qualified
        /// component name, while requiring identical physical semantics. This
        /// permits different storage/block order without silently coupling
        /// same-width equations, materials or vector conventions.
        std::vector<index> MapComponentsTo(const EquationSchema &target) const
        {
            DNDS_check_throw_info(
                family == target.family &&
                    energyConvention == target.energyConvention &&
                    physicalAccumulation == target.physicalAccumulation,
                "equation families or energy/accumulation conventions differ");
            const auto source = LogicalComponents();
            const auto destination = target.LogicalComponents();
            DNDS_check_throw_info(source.size() == destination.size(),
                                  "equation logical component counts differ");
            std::vector<index> result(source.size(), -1);
            std::vector<uint8_t> used(destination.size(), 0);
            for (index iSource = 0; iSource < index(source.size()); ++iSource)
            {
                index match = -1;
                for (index iTarget = 0; iTarget < index(destination.size());
                     ++iTarget)
                    if (source[size_t(iSource)].QualifiedName() ==
                        destination[size_t(iTarget)].QualifiedName())
                    {
                        DNDS_check_throw_info(
                            match < 0,
                            "duplicate equation component name: " +
                                source[size_t(iSource)].QualifiedName());
                        match = iTarget;
                    }
                DNDS_check_throw_info(
                    match >= 0,
                    "target equation omits component: " +
                        source[size_t(iSource)].QualifiedName());
                DNDS_check_throw_info(
                    source[size_t(iSource)] == destination[size_t(match)],
                    "equation component semantics differ: " +
                        source[size_t(iSource)].QualifiedName());
                DNDS_check_throw_info(
                    !used[size_t(match)],
                    "target equation component is mapped more than once: " +
                        destination[size_t(match)].QualifiedName());
                used[size_t(match)] = 1;
                result[size_t(iSource)] = match;
            }
            return result;
        }

        /// Resolve setup/config names to logical indices, independently of
        /// physical field grouping/layout. Accept a unique component name or
        /// block.component; empty selects all. Unknown, ambiguous and duplicate
        /// selections fail before numerical work. No name lookup is done in a
        /// cell loop or at each residual evaluation.
        std::vector<index> ResolveComponents(const std::vector<std::string> &names) const
        {
            std::vector<index> selected;
            selected.reserve(names.size());
            for (const auto &name : names)
            {
                index found = -1, ordinal = 0;
                for (const auto &block : blocks)
                    for (const auto &component : block.components)
                    {
                        if (name == component.name || name == block.name + "." + component.name)
                        {
                            DNDS_check_throw_info(found < 0, "ambiguous equation component: " + name);
                            found = ordinal;
                        }
                        ++ordinal;
                    }
                DNDS_check_throw_info(found >= 0, "unknown equation component: " + name);
                DNDS_check_throw_info(std::find(selected.begin(), selected.end(), found) == selected.end(), "duplicate equation component: " + name);
                selected.push_back(found);
            }
            return selected;
        }
    };
}
