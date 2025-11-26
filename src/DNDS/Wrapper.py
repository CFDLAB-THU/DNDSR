import inspect
import textwrap


def is_settable(cls, name):
    """Check if attribute `name` on class `cls` is settable."""
    attr = getattr(cls, name, None)
    # Look for descriptor in class dict
    descriptor = cls.__dict__.get(name, None)

    # Case 1: descriptor with setter (data descriptor)
    # e.g., properties with fset, descriptors implementing __set__
    if hasattr(descriptor, "__set__"):
        return True

    # Case 2: property without setter -> not settable
    if isinstance(descriptor, property):
        return descriptor.fset is not None

    # Case 3: not a descriptor -> instance attribute is OK to set
    return True


# def generate_wrapper(pyclass: type, init_from_obj=False):
#     """Create a Python wrapper class for a pybind11 class."""

#     if not init_from_obj:

#         class Wrapper:
#             def __init__(self, *args, **kwargs):
#                 self._obj = pyclass(*args, **kwargs)

#     else:

#         class Wrapper:
#             def __init__(self, OBJ):
#                 self._obj = OBJ

#     Wrapper.__name__ = pyclass.__name__ + "__Wrapper"

#     for name, attr in pyclass.__dict__.items():
#         if callable(attr) and not (name.startswith("__") and name.endswith("__")):
#             # Generate a wrapper method with correct closure binding
#             def make_wrapper(func_name):
#                 def wrapper(self, *args, **kwargs):
#                     # This wrapper is a real Python function → cProfile visible
#                     return getattr(self._obj, func_name)(*args, **kwargs)

#                 wrapper.__name__ = func_name
#                 wrapper.__qualname__ = f"{Wrapper.__name__}.{func_name}"
#                 return wrapper

#             setattr(Wrapper, name, make_wrapper(name))

#         elif isinstance(attr, property):
#             # Wrap properties
#             def make_prop_getter(prop):
#                 def getter(self):
#                     return getattr(self._obj, prop)

#                 return getter

#             def make_prop_setter(prop):
#                 def setter(self, v):
#                     return setattr(self._obj, prop, v)

#                 if is_settable(pyclass, prop):
#                     return setter
#                 return None

#             setattr(
#                 Wrapper,
#                 name,
#                 property(make_prop_getter(name), make_prop_setter(name)),
#             )

#     return Wrapper


def generate_wrapper(pyclass: type, init_from_obj=False):

    if not init_from_obj:

        class Wrapper:
            def __init__(self, *args, **kwargs):
                self._obj = pyclass(*args, **kwargs)

    else:

        class Wrapper:
            def __init__(self, OBJ):
                self._obj = OBJ

    namespace = {}

    for name, attr in pyclass.__dict__.items():

        # skip all dunders
        if name.startswith("__") and name.endswith("__"):
            continue

        # callable = method
        if callable(attr):
            src = textwrap.dedent(
                f"""
            def {name}(self, *args, **kwargs):
                # auto-generated wrapper for {pyclass.__name__}.{name}
                return self._obj.{name}(*args, **kwargs)
            """
            )
            # compile a new code object for THIS method only
            local_ns = {}
            exec(src, {}, local_ns)
            func = local_ns[name]
            setattr(Wrapper, name, func)
            continue

        # property
        descriptor = pyclass.__dict__.get(name)
        if isinstance(descriptor, property):
            # generate getter
            src_get = textwrap.dedent(
                f"""
            def get_{name}(self):
                return self._obj.{name}
            """
            )
            local_ns = {}
            exec(src_get, {}, local_ns)
            getter = local_ns[f"get_{name}"]

            if descriptor.fset is not None:
                src_set = textwrap.dedent(
                    f"""
                def set_{name}(self, value):
                    self._obj.{name} = value
                """
                )
                local_ns = {}
                exec(src_set, {}, local_ns)
                setter = local_ns[f"set_{name}"]
                prop = property(getter, setter)
            else:
                prop = property(getter)

            setattr(Wrapper, name, prop)
            continue

    return Wrapper


if __name__ == "__main__":

    class Test:
        def __init__(self):
            self.val = 0

        def foo(self):
            return 1

        @property
        def bar(self):
            return self.val

        @bar.setter
        def bar(self, val):
            self.val = val

        @property
        def mon(self):
            return self.val

    Wrapped = generate_wrapper(Test)
    print("A")
    w = Wrapped()
    print(w.foo())
    print(w.bar)
    w.bar = 1
    print(w.mon)
    try:
        w.mon = 0
    except AttributeError:
        pass

    print(Wrapped.__dict__)
