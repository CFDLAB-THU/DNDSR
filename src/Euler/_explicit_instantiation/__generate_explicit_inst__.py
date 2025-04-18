import sys, os


TU_templates = {
    # EulerEvaluator_EvaluateDt_
    "EulerEvaluator_EvaluateDt_": """
#include "../EulerEvaluator_EvaluateDt.hxx"

DNDS_EulerEvaluator_EvaluateDt_INS_EXTERN({model}, )
""",
    # EulerEvaluator_EvaluateRHS_
    "EulerEvaluator_EvaluateRHS_": """
#include "../EulerEvaluator_EvaluateRHS.hxx"

DNDS_EulerEvaluator_EvaluateRHS_INS_EXTERN({model}, )
""",
    # EulerEvaluator_
    "EulerEvaluator_": """
#include "../EulerEvaluator.hxx"

DNDS_EulerEvaluator_INS_EXTERN({model}, )
""",
    # EulerSolver_Init_
    "EulerSolver_Init_": """
#include "../EulerSolver_Init.hxx"

DNDS_EULERSOLVER_INIT_INS_EXTERN({model}, )
""",
    # EulerSolver_
    "EulerSolver_": """
#include "../EulerSolver.hxx"

DNDS_EULERSOLVER_INS_EXTERN({model}, )
""",
    # EulerSolver_PrintData_
    "EulerSolver_PrintData_": """
#include "../EulerSolver_PrintData.hxx"

DNDS_EULERSOLVER_PRINTDATA_INS_EXTERN({model}, )
""",
}

models = [
    "NS",
    "NS_2D",
    "NS_3D",
    "NS_SA",
    "NS_SA_3D",
    "NS_2EQ",
    "NS_2EQ_3D",
    "NS_EX",
    "NS_EX_3D",
]


base_pos = os.path.abspath(os.path.dirname(__file__))

file_done = set()

for inst_prefix, template in TU_templates.items():
    for model in models:
        fname = inst_prefix + model + ".cpp"
        assert fname not in file_done, f"file name clash: {fname}"
        file_done.add(fname)
        template = template.strip()
        with open(os.path.join(base_pos, fname), "w") as f:
            f.write(template.format(model=model))
        print(f"generated: {fname}")
