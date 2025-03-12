import os
import sys

dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dirname, "..", "..", "script"))

from utils.GraceExit import GraceExit

handler = GraceExit(max_attempts=5)

config_name = os.path.join(dirname, "config_0012_mg_bench.json")

out_base = "../data/out/NACA0012_H2-MGtest_2_VRes_AOA15.dir"

os.makedirs(out_base, exist_ok=True)

options_list = {}


mg_seqs = [
    (0, 3, 3),
    (1, 3, 3),
    (2, 3, 3),
    (1, 4, 4),
    (2, 4, 4),
]

for mg_set in mg_seqs:
    options = []
    options.append(("/convergenceControl/nTimeStepInternal", 10000))
    options.append(("/linearSolverControl/jacobiCode", 2))
    options.append(("/linearSolverControl/gmresCode", 1))
    options.append(
        ("/linearSolverControl/coarseGridLinearSolverControlList/1/jacobiCode", 2)
    )
    options.append(
        ("/linearSolverControl/coarseGridLinearSolverControlList/2/jacobiCode", 2)
    )
    name = "gmres5x1ilu"
    for i in (1, 2):
        name += f"-{mg_set[i]}ilu" if i <= mg_set[0] else ""
    options.append(
        (
            "/dataIOControl/outPltName",
            f'\\"{os.path.join(out_base, name)}\\"',
        )
    )
    options.extend(
        [
            ("/linearSolverControl/multiGridLP", mg_set[0]),
            (
                "/linearSolverControl/coarseGridLinearSolverControlList/1/multiGridNIter",
                mg_set[1],
            ),
            (
                "/linearSolverControl/coarseGridLinearSolverControlList/2/multiGridNIter",
                mg_set[2],
            ),
        ]
    )
    options_list[name] = options

print(options_list)


try:
    for name, options in options_list.items():
        cmd = f"mpirun -np 16 app/eulerSA.exe {config_name}"
        for opt in options:
            cmd += f" -k {opt[0]}"
            cmd += f" -v {opt[1]}"
        cmd += f" > {os.path.join(out_base, name)}-stdout.txt"

        print(f"Command::: {name} \n\n {cmd}\n")

        os.system(cmd)
except KeyboardInterrupt:
    pass
