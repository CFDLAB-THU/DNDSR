import sys, os
import argparse


def count_lines(base_dir: str, suffixes: list[str]):
    path_to_count = {}
    suffix_to_count = {}
    for dir, dirs, files in os.walk(base_dir):
        for file in files:
            if not file.endswith(tuple(suffixes)):
                continue
            cur_suf = None
            for suffix in suffixes:
                if file.endswith(suffix):
                    cur_suf = suffix

            fpath = os.path.join(dir, file)
            with open(fpath, "r") as f:
                lc = len(f.readlines())
                path_to_count[fpath] = lc
                suffix_to_count[cur_suf] = suffix_to_count.get(cur_suf, 0) + lc
    return path_to_count, suffix_to_count


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("count_lines")
    argparser.add_argument("base_dir", type=str)
    argparser.add_argument(
        "-s",
        "--suffix",
        type=lambda s: s.split(","),
        help="comma separated list",
        default=[".cpp", ".hpp", ".c", ".h", ".hxx", ".py"],
    )
    args = argparser.parse_args()
    base_dir = os.path.realpath(args.base_dir)
    path_to_count, suffix_to_count = count_lines(base_dir, args.suffix)
    for k, v in path_to_count.items():
        print(f"{v}: {os.path.relpath(k, base_dir)}")
    print(f"\nSummary: {sum(suffix_to_count.values())} lines")
    for k, v in suffix_to_count.items():
        print(f"{k:10} ##LC## {v}")
