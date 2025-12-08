# CUDA notes

## About Eigen

Eigen can create temporary terms that uses dynamic sized buffer.

For example, with

```
Eigen::Map<Eigen::MatrixXd> a{...}, b{...} ...
```

With

```
a = b * b;
```

`b*b` generates a temporary value, then assigned to a.

Using

```
a.noalias() = b * b;
```

would be fine even for dynamic-sized maps.

## Thread load

Each thread should better have lower computational load.

For 3x5 sized linear reconstruction problem, using 1 thread per cell is OK, but 5 threads per cell is far better.

## Compilation

It seems sometimes CUAD toolchain could be corrupted that dynamic linking / separated linking of cub::reduce would cause runtime error?

Changing from 12.1 to 12.0 installation fixed.

## Silent error

Complicated type systems on device should be very carefully treated.

Sometimes your CTOR/assignment for trivial copy / construction on the call chain (Base class's) or some member functions (see array iterator's getView()) are missing the `__device__` mark, that may cause the compiler to silently **default initialize** the object. Normally it would be an error emitted, but maybe in CRTP and/or relocatable-device-code this could be silently error.

Always ensure full `__device__` coverage on the call chain of device side.

## JIT

NVCC could generate ptx code and JIT it by driver. If the driver JIT compiler is buggy, some errors could happen **SILENTLY**.

Safe thing to do: use:

``` bash
cmake ... -DCMAKE_CUDA_ARCHITECTURES=native
```

##
