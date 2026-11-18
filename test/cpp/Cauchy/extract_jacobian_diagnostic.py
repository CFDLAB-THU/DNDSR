"""Observe four disjoint assembly batches in a private production source copy.

Only external scope declarations/Finish calls are inserted. The round-trip
check below rejects any accidental change to the production text. Unique
anchors fail explicitly when the implementation structure changes.

After free-kernel extraction the legacy report labels JacobianFreeze and
JacobianFaces bracket InitializeCompressibleFluxJacobian and PrepareCompressibleFluxJacobian,
respectively. Their sum remains assembly work; do not compare either individual
bucket with pre-extraction subphase timings without this qualification.
"""
import argparse
import hashlib
import re
from pathlib import Path


def extract(source):
    text = source
    insertions = []

    def before(anchor, addition):
        nonlocal text
        # Match tokens while accepting indentation/wrapping from the commit
        # formatter. Insert before the complete original line, retaining every
        # original byte for the round-trip assertion below.
        pattern = r"\s+".join(re.escape(part) for part in anchor.strip().split())
        matches = list(re.finditer(pattern, text))
        if len(matches) != 1:
            raise ValueError(f"expected one Jacobian batch anchor: {anchor!r}")
        start = text.rfind("\n", 0, matches[0].start()) + 1
        text = text[:start] + addition + text[start:]
        insertions.append(addition)

    before("        ValidateAuxValues(state, auxiliary, time);",
           "        ::DiagnosticTiming::Scope assemblySetup(::DiagnosticTiming::Block::JacobianSetup);\n")
    before('            RequireNumericalSuccess(CPU::InitializeCompressibleFluxJacobian(request), "initialize equation Jacobian");',
           "            assemblySetup.Finish();\n"
           "            ::DiagnosticTiming::Scope assemblyFreeze(::DiagnosticTiming::Block::JacobianFreeze);\n")
    before("            if ((chemical || rans) && sourceWorkspace)\n",
           "            assemblyFreeze.Finish();\n"
           "            ::DiagnosticTiming::Scope assemblySource(::DiagnosticTiming::Block::JacobianSource);\n")
    before('            RequireNumericalSuccess(CPU::PrepareCompressibleFluxJacobian(request, _data->boundaryPoint,',
           "            assemblySource.Finish();\n"
           "            ::DiagnosticTiming::Scope assemblyFaces(::DiagnosticTiming::Block::JacobianFaces);\n")
    restored = text
    for addition in reversed(insertions):
        if restored.count(addition) != 1:
            raise ValueError("ambiguous diagnostic insertion")
        restored = restored.replace(addition, "", 1)
    if restored != source:
        raise ValueError("Jacobian extraction changed production text")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.read_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("// Generated assembly diagnostic; original SHA256: " +
                           hashlib.sha256(source).hexdigest() +
                           '\n#include "DiagnosticTiming.hpp"\n' + extract(source.decode()))
