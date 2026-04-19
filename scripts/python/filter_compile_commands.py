import json
import re


def filter_compile_commnads(fname: str, fname_out: str, excludes: list[str]):
    with open(fname) as f:
        listv = json.load(f)
    exclude_patterns = [re.compile(exclude) for exclude in excludes]

    listv_new = []
    for item in listv:
        need = True
        for pat in exclude_patterns:
            if pat.match(item["file"]):
                need = False
                print(f'{item["file"]}: excluded by  {pat.pattern}')
                break
        if need:
            listv_new.append(item)

    with open(fname_out, "w") as f:
        json.dump(listv_new, f)


if __name__ == "__main__":
    excludes = [
        r".*app/external.*\.cpp",
        r".*app/Solver.*\.cpp",
        r".*app/DNDS.*\.cpp",
        r".*app/Geom.*\.cpp",
        r".*app/CFV.*\.cpp",
        r".*app/Euler/(?!euler.*).*\.cpp",
        r".*src/\w+/\w+_pybind11\.cpp",
        r".*src/[\w/]+/\w+_bind\w*\.cpp",
    ]
    filter_compile_commnads(
        "compile_commands.json", "compile_commands_out.json", excludes=excludes
    )
