"""
C++ code emitter for shape function derivatives and element traits.

Takes symbolic SymPy expressions and produces optimized C++ assignment code
using CSE (Common Subexpression Elimination).

Generated structure per element:
    // <GEN_SHAPE_FUNCS_BEGIN>
    template <> struct ShapeFuncImpl<Line2> {
        template <class TPoint, class TArray>
        DNDS_DEVICE_CALLABLE static inline void Diff0(const TPoint &p, TArray &&v) { ... }
        ...Diff1, Diff2, Diff3...
    };
    // <GEN_SHAPE_FUNCS_END>

    // <GEN_ELEM_TRAITS_BEGIN>
    template <> struct ElementTraits<Line2> { ... };
    // <GEN_ELEM_TRAITS_END>

The primary templates are declared in ElementTraitsBase.hpp and each generated
header provides full specializations.
"""

import os
import sys
from sympy import symbols, diff, cse, numbered_symbols, simplify, Rational, Pow, S
from sympy.printing.c import C99CodePrinter
from .diff_layout import get_diff_layout
from .traits_emitter import emit_elem_traits


class _ShapeFuncPrinter(C99CodePrinter):
    """Custom C printer that avoids pow() calls for integer exponents.

    Integer powers are expanded as multiplications so the generated code
    does not depend on std::pow or any custom pow helper, and is safe for
    both host and CUDA device compilation.
    """

    def _print_Pow(self, expr):
        base = expr.base
        exp = expr.exp
        # For small integer exponents, expand as multiplications
        if exp.is_integer and abs(int(exp)) <= 8:
            n = int(exp)
            b = self._print(base)
            if n == 0:
                return "1.0"
            if n == 1:
                return b
            if n == -1:
                return f"(1.0 / ({b}))"
            if n > 1:
                # Wrap the whole product in parens so it composes safely
                # inside larger Mul expressions.
                return "(" + " * ".join([f"({b})"] * n) + ")"
            if n < -1:
                pos = " * ".join([f"({b})"] * (-n))
                return f"(1.0 / ({pos}))"
        # Fall back to std::pow for non-integer or large exponents
        return super()._print_Pow(expr)

    def _print_Rational(self, expr):
        # Emit as double division to avoid integer arithmetic
        p, q = expr.p, expr.q
        if q == 1:
            return f"{p}.0"
        return f"({float(expr)})"


_printer = _ShapeFuncPrinter()


def _ccode(expr):
    """Generate C code string using our custom printer."""
    return _printer.doprint(expr)


def compute_derivatives(elem_cls, diff_order):
    """Compute all derivative expressions for one element at one diff order.

    Returns a list of (row, col, sympy_expr) triples for non-zero entries.
    """
    Nj = elem_cls.shape_functions()
    pvars = elem_cls.param_vars()
    layout = get_diff_layout(elem_cls.dim, diff_order)

    entries = []
    for row_idx, deriv_tuple in enumerate(layout):
        for col_idx, N in enumerate(Nj):
            expr = N
            for d in range(len(deriv_tuple)):
                for _ in range(deriv_tuple[d]):
                    expr = diff(expr, pvars[d])
            expr = simplify(expr)
            if expr != 0:
                entries.append((row_idx, col_idx, expr))
    return entries


def emit_cpp_block(entries, indent="    "):
    """Emit optimized C++ code for a set of (row, col, expr) entries.

    Uses CSE to factor out common subexpressions.
    Returns a list of C++ lines (strings).
    """
    if not entries:
        return [f"{indent}// all zero"]

    # Collect all expressions
    exprs = [e[2] for e in entries]
    row_cols = [(e[0], e[1]) for e in entries]

    # CSE
    temps = numbered_symbols("_t")
    replacements, reduced = cse(exprs, symbols=temps)

    lines = []
    for sym, sub_expr in replacements:
        code = _ccode(sub_expr)
        lines.append(f"{indent}const t_real {sym} = {code};")

    for (row, col), expr in zip(row_cols, reduced):
        code = _ccode(expr)
        lines.append(f"{indent}v({row}, {col}) = {code};")

    return lines


def _extract_preserved_content(filepath):
    """Extract content after the last generation marker.

    Checks for GEN_ELEM_TRAITS_END first (new style), then falls back to
    GEN_SHAPE_FUNCS_END (legacy style). Returns everything after that marker
    line, or None if no marker found.
    """
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        content = f.read()

    # Try new-style marker first
    for marker in ["// <GEN_ELEM_TRAITS_END>", "// <GEN_SHAPE_FUNCS_END>"]:
        idx = content.find(marker)
        if idx != -1:
            line_end = content.find('\n', idx)
            if line_end == -1:
                return None
            remaining = content[line_end:]
            # For legacy marker, skip the hand-written ElementTraits since
            # we are now generating it. Only preserve content after
            # GEN_ELEM_TRAITS_END if present, otherwise discard the old
            # hand-written traits entirely.
            if marker == "// <GEN_SHAPE_FUNCS_END>":
                # Check if the remaining content has the new marker
                new_marker_idx = remaining.find("// <GEN_ELEM_TRAITS_END>")
                if new_marker_idx != -1:
                    new_line_end = remaining.find('\n', new_marker_idx)
                    if new_line_end != -1:
                        return remaining[new_line_end:]
                # Legacy: the remaining content IS the hand-written traits.
                # We will replace it entirely with generated traits.
                # Return None to indicate nothing to preserve.
                return None
            else:
                return remaining

    return None


def emit_element_file(elem_cls, out_path):
    """Generate a complete C++ header for one element type.

    Produces:
    1. ShapeFuncImpl<ElemType> specialization (Diff0..Diff3) between
       GEN_SHAPE_FUNCS markers.
    2. ElementTraits<ElemType> specialization between GEN_ELEM_TRAITS markers.
    3. Any preserved content after GEN_ELEM_TRAITS_END (if present).

    Writes to out_path (e.g. src/Geom/Elements/Line2.hpp).
    """
    name = elem_cls.name
    # Decide decorators
    device_attr = "" if elem_cls.rational else "DNDS_DEVICE_CALLABLE "

    # Try to extract preserved content (anything after the last gen marker)
    preserved = _extract_preserved_content(out_path)

    lines = []
    lines.append("#pragma once")
    lines.append(f"// Auto-generated by tools/gen_shape_functions -- DO NOT EDIT")
    lines.append(f"// Element: {name}")
    lines.append(f"// Regenerate: /usr/bin/python3 -m tools.gen_shape_functions.generate")
    lines.append("")
    lines.append('#include "DNDS/Defines.hpp"')
    lines.append('#include "Geom/Geometric.hpp"')
    lines.append('#include "Geom/ElemEnum.hpp"')
    lines.append('#include "Geom/ElementTraitsBase.hpp"')
    lines.append("")
    lines.append("namespace DNDS::Geom::Elem")
    lines.append("{")
    lines.append("")
    lines.append(f"    // Forward declaration (primary template is in ElementTraitsBase.hpp)")
    lines.append(f"    template <ElemType> struct ShapeFuncImpl;")
    lines.append("")

    # ---- Shape functions ----
    lines.append(f"    // <GEN_SHAPE_FUNCS_BEGIN>")
    lines.append(f"    template <>")
    lines.append(f"    struct ShapeFuncImpl<{name}>")
    lines.append(f"    {{")

    # For each diffOrder (0..3), emit a static method
    for diff_order in range(4):
        if diff_order > 0:
            lines.append("")
        lines.append(f"        template <class TPoint, class TArray>")
        lines.append(f"        {device_attr}static inline void Diff{diff_order}"
                     f"(const TPoint &p, TArray &&v)")
        lines.append(f"        {{")

        # Extract parametric coordinates
        if elem_cls.dim >= 1:
            lines.append("            t_real xi = p[0];")
        if elem_cls.dim >= 2:
            lines.append("            t_real et = p[1];")
        if elem_cls.dim >= 3:
            lines.append("            t_real zt = p[2];")

        # Singularity guard for rational elements
        if elem_cls.rational and elem_cls.singularity_guard:
            lines.append(f"            // singularity guard")
            for guard_line in elem_cls.singularity_guard.split("\n"):
                lines.append(f"            {guard_line}")

        # Compute and emit
        print(f"  {name} diffOrder={diff_order} ...", end="", flush=True, file=sys.stderr)
        entries = compute_derivatives(elem_cls, diff_order)
        print(f" {len(entries)} non-zero entries", file=sys.stderr)

        cpp_lines = emit_cpp_block(entries, indent="            ")
        lines.extend(cpp_lines)
        lines.append(f"        }}")

    lines.append(f"    }};")
    lines.append(f"    // <GEN_SHAPE_FUNCS_END>")

    # ---- Element traits ----
    lines.append(f"")
    lines.append(f"    // <GEN_ELEM_TRAITS_BEGIN>")
    traits_lines = emit_elem_traits(name)
    lines.extend(traits_lines)
    lines.append(f"    // <GEN_ELEM_TRAITS_END>")

    # Append preserved content (anything after the last gen marker)
    if preserved:
        lines.append(preserved)
    else:
        # Default closing for new/clean files
        lines.append("")
        lines.append(f"}} // namespace DNDS::Geom::Elem")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    return len(lines)
