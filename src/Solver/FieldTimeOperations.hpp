#pragma once

#include "DNDS/Algebra/DofAlgebra.hpp"

namespace DNDS::ODE
{
    /// Native single-field adapter for StateTimeStepper/StateDITR. The time
    /// algorithms own no array indexing; alternative state owners supply the
    /// same small operations/binding protocol without duplicating tableaus.
    struct FieldTimeOperations
    {
        using Field = Algebra::DofField<real>;
        static Field NewVector(const Field &prototype)
        {
            prototype.Validate();
            DNDS_check_throw_info(prototype.Device() == Runtime::Device::Host(), "CPU time driver cannot execute device fields");
            return Field(prototype.Space(), Runtime::Array<real>(prototype.Owned().Layout()), Runtime::Array<real>(prototype.Ghost().Layout()));
        }
        static void CheckCompatible(const Field &prototype, const Field &field)
        {
            prototype.Validate();
            field.Validate();
            DNDS_check_throw_info(field.Space().Same(prototype.Space()) && field.Width() == prototype.Width() && field.Device() == prototype.Device(),
                                  "time field space, components or backend differs");
        }
        static void RequireDistinct(const Field &initial, const Field &trial)
        {
            for (const auto *input : {&initial.Owned(), &initial.Ghost()})
                for (const auto *output : {&trial.Owned(), &trial.Ghost()})
                    DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), output->Storage()), "time initial/trial fields alias");
        }
        struct Binding
        {
            Runtime::ArrayStorage::Shared owned, ghost;
            Algebra::DofSpace space;
            size_t width;
            Runtime::Device device;
            uint64_t ownedVersion, ownedShape, ghostShape;
            explicit Binding(const Field &field)
                : owned(field.Owned().SharedStorage()), ghost(field.Ghost().SharedStorage()), space(field.Space()), width(field.Width()), device(field.Device()), ownedVersion(field.Owned().Version()),
                  ownedShape(field.Owned().ShapeVersion()), ghostShape(field.Ghost().ShapeVersion()) {}
            void Check(const Field &field, bool unchangedOwned = false) const
            {
                field.Validate();
                DNDS_check_throw_info(field.Space().Same(space) && field.Width() == width && field.Device() == device &&
                                          field.Owned().SharedStorage() == owned && field.Ghost().SharedStorage() == ghost &&
                                          field.Owned().ShapeVersion() == ownedShape && field.Ghost().ShapeVersion() == ghostShape,
                                      "time callback rebound/reshaped a field");
                DNDS_check_throw_info(!unchangedOwned || field.Owned().Version() == ownedVersion, "time callback modified an owned input/source");
            }
        };
        static void Copy(Field &out, const Field &x) { Algebra::Copy(out, x); }
        static void Fill(Field &out, real value) { Algebra::Fill(out, value); }
        static void Scale(Field &out, real value) { Algebra::Scale(out, value); }
        static void Axpy(Field &out, real value, const Field &x) { Algebra::Axpy(out, value, x); }
        static void LinearCombination(Field &out, real a, const Field &x, real b, const Field &y) { Algebra::LinearCombination(out, a, x, b, y); }
        /// Identity physical accumulation for the generic single-field ODE
        /// adapter. Equation-specific adapters may replace these three actions.
        static void PhysicalCopyScale(Field &out, real a, const Field &x)
        {
            Copy(out, x);
            Scale(out, a);
        }
        static void PhysicalLinearCombination(Field &out, real a, const Field &x, real b, const Field &y)
        {
            LinearCombination(out, a, x, b, y);
        }
        static void PhysicalAxpy(Field &out, real a, const Field &x) { Axpy(out, a, x); }
        static void PhysicalStateAndRate(Field &out, real a, const Field &x, real b, const Field &rate)
        {
            LinearCombination(out, a, x, b, rate);
        }
        static void AssembleTimeResidual(Field &out, const Field &constant, const Field &state,
                                         const Field &rate, real inverseDt, real rhsScale)
        {
            LinearCombination(out, 1, constant, -inverseDt, state);
            Axpy(out, rhsScale, rate);
        }
    };
}
