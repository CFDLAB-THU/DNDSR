#!/usr/bin/env bash
set -euo pipefail
prefix=${1:?missing cfd_externals prefix}

for header in zlib.h hdf5.h cgnslib.h metis.h parmetis.h cantera/base/Solution.h caliper/cali.h; do
    if [[ ! -s "$prefix/include/$header" ]]; then
        printf 'Missing external header %s\n' "$header" >&2
        exit 1
    fi
done
grep -Eq '^#define H5_HAVE_PARALLEL 1' "$prefix/include/H5pubconf.h"
for mechanism in h2o2 gri30; do
    test -s "$prefix/data/$mechanism.yaml"
done

for lib in z hdf5 cgns metis parmetis cantera_shared caliper; do
    path="$prefix/lib/lib$lib.so"
    if [[ ! -e "$path" ]]; then
        path="$prefix/lib64/lib$lib.so"
    fi
    if [[ ! -r "$path" ]]; then
        printf 'Missing external library lib%s.so\n' "$lib" >&2
        exit 1
    fi
    dependencies="$(ldd "$path")"
    if [[ "$dependencies" == *"not found"* ]]; then
        printf 'Unresolved libraries for %s\n%s\n' "$path" "$dependencies" >&2
        exit 1
    fi
done
