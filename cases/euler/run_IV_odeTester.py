import os
import sys
import json
import collections
import subprocess
import pprint

dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dirname, "..", "..", "script"))

from utils.GraceExit import GraceExit

handler = GraceExit(max_attempts=5)


def run():
    no_run = False
    config_name = os.path.join(dirname, "euler_config_HM3Tester.json")

    # out_base = "../data/outUnsteady/IV_HM3Test_B0_JC2"

    out_base = "../data/outUnsteady/IV_HM3Test_B0_JC1_RB_tr"

    ode_select = [
        "U2R2",
        "U2R2OP",
        "U2R2Pre0p75m0",
        "U2R2Pre0p75m2",
        "U2R2Pre0p75m4",
        "U2R2Pre1m2",
        "U2R2Pre1m4",
        "ESDIRK3",
        "ESDIRK4",
        "BDF2",
        # "U2R2Pre1mInf",
    ]

    dt_opts = collections.defaultdict(lambda: [0.4, 0.2, 0.1, 0.1 / 2, 0.1 / 4])
    dt_opts["BDF2"] = [0.4, 0.2, 0.1, 0.1 / 2, 0.1 / 4, 0.1 / 8, 0.1 / 16]

    name_prefix = ""
    # name_prefix = "x1-"

    os.makedirs(out_base, exist_ok=True)
    global_opt = {
        "/linearSolverControl/jacobiCode": 1,
        "/linearSolverControl/sgsIter": 0,
        "/dataIOControl/meshDirectBisect": 0,
        "/convergenceControl/useVolWiseResidual": False,
        "/convergenceControl/mergeMultiResidual": 1,
    }
    ode_opts = get_ode_opts()

    # as tEnd = 2

    options_list = {}

    for ode in ode_select:
        ode_opt = ode_opts[ode]
        dt_opt = dt_opts[ode]
        cur_opt = {}
        cur_opt.update(global_opt)
        cur_opt.update(ode_opt)
        for idt, dt in enumerate(dt_opt):
            case_name = name_prefix + f"out_{ode}_{idt}"
            cOption = []
            for k, v in cur_opt.items():
                cOption.append((k, json.dumps(v)))
            cOption.append(("/timeMarchControl/dtImplicit", f"{dt}"))
            cOption.append(
                ("/dataIOControl/outPltName", json.dumps(os.path.join(out_base, case_name)))
            )

            options_list[case_name] = cOption

    pprint.pprint(options_list)

    try:
        for name, options in options_list.items():
            # cmd = f"mpirun -np 16 app/euler.exe {config_name}"
            # for opt in options:
            #     cmd += f" -k {opt[0]}"
            #     cmd += f" -v {opt[1]}"
            # cmd += f" > {os.path.join(out_base, name)}-stdout.txt"

            # print(f"Command::: {name} \n\n {cmd}\n")
            # os.system(cmd)

            cmd = [
                "mpirun",
                "-np",
                "16",
                "app/euler.exe",
                config_name,
            ]
            for opt in options:
                cmd.extend(["-k", opt[0], "-v", opt[1]])
            print("\n" + "==" * 32)
            print(name)
            print("--" * 32)
            pprint.pprint(cmd)

            stdout_file = os.path.join(out_base, f"{name}-stdout.txt")

            print(f"stdout_file: {stdout_file}")

            if no_run:
                continue

            with open(stdout_file, "w") as stdout_file:
                subprocess.run(
                    cmd,
                    stdout=stdout_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                    shell=False,
                )

    except KeyboardInterrupt:
        pass


def get_ode_opts():
    ode_opts = {}
    ode_opts["U2R2"] = {
        "/timeMarchControl/odeCode": 411,
        "/timeMarchControl/odeSetting1": 0.5,
        "/timeMarchControl/odeSetting2": 0,
        "/timeMarchControl/odeSetting3": 0.75,
        "/timeMarchControl/odeSetting4": -1e10,
        "/timeMarchControl/odeSettingsExtra": {
            "nMG": 0,
            "thetaMMG": 1,
            "coefIncMidMG": 1,
        },
    }

    ode_opts["U2R2Pre1mInf"] = {
        "/timeMarchControl/odeCode": 411,
        "/timeMarchControl/odeSetting1": 0.5,
        "/timeMarchControl/odeSetting2": 0,
        "/timeMarchControl/odeSetting3": 1,
        "/timeMarchControl/odeSetting4": -1e10,
        "/timeMarchControl/odeSettingsExtra": {
            "nMG": 0,
            "thetaMMG": 1,
            "coefIncMidMG": 1,
        },
    }

    ode_opts["U2R2Pre1m4"] = {
        "/timeMarchControl/odeCode": 411,
        "/timeMarchControl/odeSetting1": 0.5,
        "/timeMarchControl/odeSetting2": 0,
        "/timeMarchControl/odeSetting3": 1,
        "/timeMarchControl/odeSetting4": -4,
        "/timeMarchControl/odeSettingsExtra": {
            "nMG": 0,
            "thetaMMG": 1,
            "coefIncMidMG": 1,
        },
    }

    ode_opts["U2R2Pre1m2"] = {
        "/timeMarchControl/odeCode": 411,
        "/timeMarchControl/odeSetting1": 0.5,
        "/timeMarchControl/odeSetting2": 0,
        "/timeMarchControl/odeSetting3": 1,
        "/timeMarchControl/odeSetting4": -2,
        "/timeMarchControl/odeSettingsExtra": {
            "nMG": 0,
            "thetaMMG": 1,
            "coefIncMidMG": 1,
        },
    }

    ode_opts["U2R2Pre0p75m4"] = {
        "/timeMarchControl/odeCode": 411,
        "/timeMarchControl/odeSetting1": 0.5,
        "/timeMarchControl/odeSetting2": 0,
        "/timeMarchControl/odeSetting3": 0.75,
        "/timeMarchControl/odeSetting4": -4,
        "/timeMarchControl/odeSettingsExtra": {
            "nMG": 0,
            "thetaMMG": 1,
            "coefIncMidMG": 1,
        },
    }

    ode_opts["U2R2Pre0p75m2"] = {
        "/timeMarchControl/odeCode": 411,
        "/timeMarchControl/odeSetting1": 0.5,
        "/timeMarchControl/odeSetting2": 0,
        "/timeMarchControl/odeSetting3": 0.75,
        "/timeMarchControl/odeSetting4": -2,
        "/timeMarchControl/odeSettingsExtra": {
            "nMG": 0,
            "thetaMMG": 1,
            "coefIncMidMG": 1,
        },
    }

    ode_opts["U2R2Pre0p75m0"] = {
        "/timeMarchControl/odeCode": 411,
        "/timeMarchControl/odeSetting1": 0.5,
        "/timeMarchControl/odeSetting2": 0,
        "/timeMarchControl/odeSetting3": 0.75,
        "/timeMarchControl/odeSetting4": 0,
        "/timeMarchControl/odeSettingsExtra": {
            "nMG": 0,
            "thetaMMG": 1,
            "coefIncMidMG": 1,
        },
    }

    ode_opts["U2R2OP"] = {
        "/timeMarchControl/odeCode": 411,
        "/timeMarchControl/odeSetting1": 0.5,
        "/timeMarchControl/odeSetting2": 0,
        "/timeMarchControl/odeSetting3": 1,
        "/timeMarchControl/odeSetting4": 0,
        "/timeMarchControl/odeSettingsExtra": {
            "nMG": 0,
            "thetaMMG": 1,
            "coefIncMidMG": 1,
        },
    }

    ode_opts["ESDIRK3"] = {
        "/timeMarchControl/odeCode": 202,
        "/timeMarchControl/odeSettingsExtra": {},
    }

    ode_opts["ESDIRK4"] = {
        "/timeMarchControl/odeCode": 0,
        "/timeMarchControl/odeSettingsExtra": {},
    }

    ode_opts["BDF2"] = {
        "/timeMarchControl/odeCode": 1,
        "/timeMarchControl/odeSettingsExtra": {},
    }

    return ode_opts


if __name__ == "__main__":
    run()
