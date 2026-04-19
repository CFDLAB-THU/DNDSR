"""Base class for element shape function definitions."""

from sympy import symbols

# Parametric coordinates
xi, et, zt = symbols('xi et zt', real=True)


class ElementDef:
    """Base class for element definitions.

    Subclasses must set:
        name:       C++ ElemType enum name, e.g. 'Line2'
        dim:        spatial dimension of the parametric space (1, 2, or 3)
        num_nodes:  number of shape function nodes
        rational:   True if shape functions involve division (Pyramid)

    Subclasses must implement:
        shape_functions() -> list of sympy expressions (length num_nodes)
            Expressed in terms of xi, et, zt as needed.
    """

    name: str = ""
    dim: int = 0
    num_nodes: int = 0
    rational: bool = False

    # If rational, the singularity guard code (C++ snippet) to prepend
    singularity_guard: str = ""

    @classmethod
    def shape_functions(cls):
        """Return list of num_nodes sympy expressions for N_j."""
        raise NotImplementedError

    @classmethod
    def param_vars(cls):
        """Return the parametric variable symbols for this element's dimension."""
        if cls.dim == 1:
            return (xi,)
        elif cls.dim == 2:
            return (xi, et)
        elif cls.dim == 3:
            return (xi, et, zt)
        else:
            raise ValueError(f"Invalid dim={cls.dim}")
