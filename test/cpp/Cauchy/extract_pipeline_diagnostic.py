"""Add external diagnostic scopes to private copies of the unified FV pipeline.

The generated translation unit preserves every production statement and adds
only clocks/counters. Unique anchors deliberately fail when orchestration is
restructured so the performance diagnostic cannot silently measure a stale
path.
"""
import argparse
import hashlib
import re
from pathlib import Path


def pattern(text):
    return r"\s+".join(re.escape(part) for part in text.strip().split())


def insert_after_once(text, anchor, addition):
    matches = list(re.finditer(pattern(anchor), text))
    if len(matches) != 1:
        raise ValueError(f"pipeline anchor changed: {anchor!r}")
    end = text.find("\n", matches[0].end()) + 1
    return text[:end] + addition + text[end:]


def insert_before_once(text, anchor, addition):
    matches = list(re.finditer(pattern(anchor), text))
    if len(matches) != 1:
        raise ValueError(f"pipeline anchor changed: {anchor!r}")
    start = text.rfind("\n", 0, matches[0].start()) + 1
    return text[:start] + addition + text[start:]


def replace_once(text, old, new):
    matches = list(re.finditer(pattern(old), text))
    if len(matches) != 1:
        raise ValueError(f"pipeline anchor changed: {old!r}")
    match = matches[0]
    return text[:match.start()] + new + text[match.end():]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--extra-source", action="append", default=[], type=Path)
    args = parser.parse_args()
    sources = [args.source, *args.extra_source]
    text = "\n".join(source.read_text() for source in sources)

    correction = (
        "void FiniteVolume::PrepareCorrectionAux(EquationDofs &state, ReconstructionState &reconstruction,\n"
        "Workspace &work, AuxiliaryValues &auxiliary,\n"
        "Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream,\n"
        "const Runtime::ExecutionStream &communication, real time,\n"
        "OptionalRef<const QueuedEvaluationDependency> dependency) const\n    {")
    text = insert_after_once(
        text, correction,
        "        ::DiagnosticTiming::Scope diagnosticAuxiliary(::DiagnosticTiming::Block::Auxiliary);\n"
        "        ::DiagnosticTiming::AuxiliaryTrace correctionTrace(!work.meanGradients, 1, ::DiagnosticTiming::currentLevel);\n")
    text = insert_before_once(
        text, "        for (auto &halo : reconstruction.meanHalo)\n"
              "            transfers.push_back(halo.BeginAfter(reconstruction.meanEpoch, engine, communication, stream.Record(),\n"
              "                                                _data->settings.CheckHaloFreshness()));",
        "        correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::LaunchBegin);\n")
    text = insert_after_once(
        text, "            transfers.push_back(halo.BeginAfter(reconstruction.meanEpoch, engine, communication, stream.Record(),\n"
              "                                                _data->settings.CheckHaloFreshness()));",
        "        correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::LaunchEnd);\n"
        "        correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::BoundaryEnd);\n"
        "        correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::WaitBegin);\n")
    text = insert_before_once(
        text, "        transfers.clear();",
        "                correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::WaitEnd);\n"
        "                correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::ComputeBegin);\n")
    text = insert_after_once(
        text, "            else preparation->Finish(auxiliary);",
        "                correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::ComputeEnd);\n")

    evaluation = (
        "void FiniteVolume::EvaluateImpl(EquationDofs &state,\n"
        "OptionalRef<EquationDofs> residual,\n"
        "ReconstructionState &reconstruction, Workspace &work, Runtime::EagerEngine &engine,\n"
        "const Runtime::ExecutionStream &stream,\n"
        "const Runtime::ExecutionStream &communication, real time,\n"
        "OptionalRef<AuxiliaryValues> auxiliary,\n"
        "OptionalRef<const QueuedEvaluationDependency> dependency, OptionalRef<R::Array<real>> sourceDerivative, bool stopAfterGradient) const\n    {")
    text = insert_after_once(
        text, evaluation,
        "        ::DiagnosticTiming::Level diagnosticLevel(_data->settings.numerics.firstOrderConvection ? 2 : _data->settings.numerics.reconstruction == \"variational\" ? 0 : 1);\n"
        "        ::DiagnosticTiming::Scope diagnosticFV(::DiagnosticTiming::Block::Spatial);\n"
        "        ::DiagnosticTiming::AuxiliaryTrace evaluationTrace(auxiliary && !work.meanGradients, 0, ::DiagnosticTiming::currentLevel);\n")
    text = replace_once(
        text,
        "                      auxiliaryPreparation.emplace(BeginAuxValues(state, *auxiliary, time, work.workers, inputs)); });",
        "                      ::DiagnosticTiming::Scope scope(::DiagnosticTiming::Block::AuxiliaryPrepare);\n"
        "                      auxiliaryPreparation.emplace(BeginAuxValues(state, *auxiliary, time, work.workers, inputs)); });")
    text = insert_before_once(
        text, "        for (auto &halo : reconstruction.meanHalo)\n"
              "            meanOperations.push_back(halo.BeginAfter(\n"
              "                reconstruction.meanEpoch, engine, communication, stream.Record(),\n"
              "                _data->settings.CheckHaloFreshness()));",
        "        evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::LaunchBegin);\n"
        "        ::DiagnosticTiming::Scope auxiliaryExchange(::DiagnosticTiming::Block::AuxiliaryExchange);\n")
    text = insert_after_once(
        text, "            meanOperations.push_back(halo.BeginAfter(\n"
              "                reconstruction.meanEpoch, engine, communication, stream.Record(),\n"
              "                _data->settings.CheckHaloFreshness()));",
        "        evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::LaunchEnd);\n"
        "        evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::BoundaryEnd);\n"
        "        evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::WaitBegin);\n")
    # This private diagnostic explicitly admits CPU-inline execution only.
    # A submitted CPU operator executes before its Submit returns. These clocks
    # must not be interpreted as worker/CUDA numerical execution timings.
    core = pattern("auxiliaryPreparation->Apply(engine, stream, CFV::ReconstructionRegion::Core);")
    text, count = re.subn(
        core,
        "{ ::DiagnosticTiming::Scope scope(::DiagnosticTiming::Block::AuxiliaryCompute); "
        "auxiliaryPreparation->Apply(engine, stream, CFV::ReconstructionRegion::Core); }",
        text)
    if count != 2:
        raise ValueError(f"expected two auxiliary core branches, got {count}")
    text = replace_once(
        text, "auxiliaryPreparation->ApplyInterfaceIndependent(engine, stream);",
        "{ ::DiagnosticTiming::Scope scope(::DiagnosticTiming::Block::AuxiliaryCompute); "
        "auxiliaryPreparation->ApplyInterfaceIndependent(engine, stream); }")
    text = insert_after_once(
        text, "        meanOperations.clear();",
        "        auxiliaryExchange.Finish();\n"
        "        evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::WaitEnd);\n")
    text = replace_once(
        text, "auxiliaryPreparation->Apply(engine, stream, rest);",
        "{ ::DiagnosticTiming::Scope scope(::DiagnosticTiming::Block::AuxiliaryCompute); "
        "evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::ComputeBegin); "
        "auxiliaryPreparation->Apply(engine, stream, rest); "
        "evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::ComputeEnd); }")

    # Matched semantic sites in the active queued path. These remain fail-closed
    # rather than silently timing retired synchronous wrappers from the archive.
    for anchor, block in (
        ("const auto timing = work.BeginQueuedPhase(engine, stream, Workspace::GradientPhase);", "Gradient"),
        ("const auto timing = work.BeginQueuedPhase(engine, stream, Workspace::VRPhase);", "Reconstruction"),
        ("const auto timing = BeginQueuedPhase(engine, stream, RHSPhase);", "FaceRHS"),
        ("const auto timing = BeginQueuedPhase(engine, stream, SourcePhase);", "VolumeSource"),
    ):
        text = insert_before_once(text, anchor,
                                  f"        ::DiagnosticTiming::Scope detailScope(::DiagnosticTiming::Block::{block});\n")
    for anchor in (
        "gradientPositivitySubmission = work.SubmitGradientPositivity(engine, stream, state);",
        "work.positivity->Apply(engine, stream, state, reconstruction.WarmValues(), reconstruction.MutableFinalized(), false);",
    ):
        text = replace_once(text, anchor,
                            "{ ::DiagnosticTiming::Scope detailScope(::DiagnosticTiming::Block::PolynomialPP); " + anchor + " }")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for source in sources:
        digest.update(str(source).encode())
        digest.update(b"\0")
        digest.update(source.read_bytes())
    args.output.write_text(
        "// Generated unified FV diagnostic; original SHA256: " + digest.hexdigest() +
        '\n#include "DiagnosticTiming.hpp"\n' + text)


if __name__ == "__main__":
    main()
