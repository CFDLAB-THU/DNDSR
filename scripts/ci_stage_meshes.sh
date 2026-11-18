#!/usr/bin/env bash
# CI owns data/mesh. Stage required blobs from one immutable upstream revision.
set -euo pipefail
cd "$(dirname "$0")/.."

revision=2eb20c1591a3fcdf37dd1fab91b812b4c6f52632
level=${1:?missing CI test level}
files=(
    IV10_2.cgns IV10_10.cgns IV10_20.cgns IV10U_10.cgns NACA0012_H2.cgns UP3D_32.cgns
    Uniform32_Periodic.cgns Uniform32_3D_Periodic.cgns
    UniformSquare_10.cgns Uniform_3x3.cgns Uniform_3x3_wall.cgns Uniform_5x5_wave.cgns
)
case "$level" in
    0|1) ;;
    2) files+=(Ball2.cgns) ;;
    *) printf 'Invalid CI test level: %s\n' "$level" >&2; exit 2 ;;
esac

scratch="$(mktemp -d)"
trap 'rm -rf -- "$scratch"' EXIT
git -C "$scratch" init -q
git -C "$scratch" fetch --depth=1 https://github.com/harryzhou2000/cfd_meshes.git "$revision"
paths=("${files[@]/#/data\/mesh\/}")
git -C "$scratch" archive FETCH_HEAD -- "${paths[@]}" | tar -x -C "$scratch"
mkdir -p data/mesh
for file in "${files[@]}"; do
    # A required input must be present in the pinned tree. Avoid following a
    # stale symlink when replacing a staged file on a persistent runner.
    test -s "$scratch/data/mesh/$file"
    mv -f "$scratch/data/mesh/$file" "data/mesh/$file"
done
printf 'Staged %s CI meshes from %s\n' "${#files[@]}" "$revision"
