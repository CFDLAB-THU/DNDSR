"""Pyramid element shape function definitions (rational)."""

from .base import ElementDef, xi, et, zt
from sympy import Rational


class Pyramid5Def(ElementDef):
    name = "Pyramid5"
    dim = 3
    num_nodes = 5
    rational = True
    singularity_guard = (
        "if (std::abs(1 - zt) < 1e-15)\n"
        "    zt -= DNDS::signP(1 - zt) * 1e-15;"
    )

    @classmethod
    def shape_functions(cls):
        # Degenerate hex approach: map xi,et to [-1,1]^2 base, zt from 0 to 1
        # The four base corners collapse to the apex at zt=1.
        # N0 = -(et+zt-1)*(xi+zt-1) / (4*(zt-1))
        # etc.
        d = zt - 1  # denominator (negative at base)
        return [
            -(et + zt - 1) * (xi + zt - 1) / (4 * d),
            (et + zt - 1) * (xi - zt + 1) / (4 * d),
            -(et - zt + 1) * (xi - zt + 1) / (4 * d),
            (xi + zt - 1) * (et - zt + 1) / (4 * d),
            zt,
        ]


class Pyramid14Def(ElementDef):
    name = "Pyramid14"
    dim = 3
    num_nodes = 14
    rational = True
    singularity_guard = (
        "if (std::abs(1 - zt) < 1e-15)\n"
        "    zt -= DNDS::signP(1 - zt) * 1e-15;"
    )

    @classmethod
    def shape_functions(cls):
        # Quadratic pyramid using MATLAB reference construction:
        #   NB   = SF_Quad9(xi/(1-zt), et/(1-zt))
        #   NB_2 = SF_Quad4(xi/(1-zt), et/(1-zt))
        #   Nz   = SF_Line3(2*zt - 1)
        # Nodes: 0-3 = NB[0..3]*Nz[0], 4 = Nz[1], 5-8 = NB[4..7]*Nz[0],
        #         9-12 = NB_2[0..3]*Nz[2], 13 = NB[8]*Nz[0]
        from sympy import Rational

        d = 1 - zt  # denominator for collapsed coords
        u = xi / d   # mapped xi in [-1,1]
        v = et / d   # mapped et in [-1,1]

        # 1D quadratic Lagrange basis
        def Lq(t, ti):
            if ti == -1:
                return t * (t - 1) / 2
            elif ti == 1:
                return t * (t + 1) / 2
            else:  # ti == 0
                return 1 - t**2

        # SF_Quad9(u, v) — 9-node serendipity quad
        xi_n9 = [-1, 1, 1, -1, 0, 1, 0, -1, 0]
        et_n9 = [-1, -1, 1, 1, -1, 0, 1, 0, 0]
        NB = [Lq(u, xi_n9[j]) * Lq(v, et_n9[j]) for j in range(9)]

        # SF_Quad4(u, v) — 4-node bilinear quad
        NB_2 = [
            (1 - u) * (1 - v) / 4,
            (1 + u) * (1 - v) / 4,
            (1 + u) * (1 + v) / 4,
            (1 - u) * (1 + v) / 4,
        ]

        # SF_Line3(2*zt - 1) — quadratic line in zt
        # t = 2*zt - 1 maps zt in [0,1] to t in [-1,1]
        # Nz[0] at t=-1 (zt=0): t*(t-1)/2 = (2z-1)*(z-1)
        # Nz[1] at t=+1 (zt=1): t*(t+1)/2 = z*(2z-1)
        # Nz[2] at t=0 (zt=0.5): 1-t^2 = 4*z*(1-z)
        t = 2 * zt - 1
        Nz = [t * (t - 1) / 2, t * (t + 1) / 2, 1 - t**2]

        return [
            NB[0] * Nz[0],    # 0: base corner (-1,-1)
            NB[1] * Nz[0],    # 1: base corner (+1,-1)
            NB[2] * Nz[0],    # 2: base corner (+1,+1)
            NB[3] * Nz[0],    # 3: base corner (-1,+1)
            Nz[1],             # 4: apex
            NB[4] * Nz[0],    # 5: base edge mid (0,-1)
            NB[5] * Nz[0],    # 6: base edge mid (+1,0)
            NB[6] * Nz[0],    # 7: base edge mid (0,+1)
            NB[7] * Nz[0],    # 8: base edge mid (-1,0)
            NB_2[0] * Nz[2],  # 9: lateral edge mid (corner 0 -> apex)
            NB_2[1] * Nz[2],  # 10: lateral edge mid (corner 1 -> apex)
            NB_2[2] * Nz[2],  # 11: lateral edge mid (corner 2 -> apex)
            NB_2[3] * Nz[2],  # 12: lateral edge mid (corner 3 -> apex)
            NB[8] * Nz[0],    # 13: base face center (0,0)
        ]
