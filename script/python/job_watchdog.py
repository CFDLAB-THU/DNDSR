import os
import subprocess
from time import sleep
import datetime


def get_squeue_jobs():
    result = subprocess.run(
        ["squeue", "--noheader", "-o", "%i %j"],
        capture_output=True,
        text=True,
        check=True,
    )

    jobs = {}
    for line in result.stdout.strip().split("\n"):
        if line:
            job_id, job_name = line.split(maxsplit=1)
            jobs[job_id] = job_name

    return jobs


def getset_output_fsize(jobid: str, base_dir: str):
    f_path = os.path.join(base_dir, jobid)
    if not os.path.isfile(f_path):
        return (-1, -1)
    with open(f_path, "r") as f:
        lines = f.readlines()
    prev_lc = -1
    if len(lines) > 1:
        prev_lc_word = lines[1]
        try:
            prev_lc = int(prev_lc_word)
        except ValueError as e:
            prev_lc = -1

    cur_lc = -1
    l_path = lines[0].strip()
    if os.path.isfile(l_path):
        cur_lc = os.path.getsize(l_path)
        lines_o = []
        lines_o.append(lines[0])
        lines_o.append(str(cur_lc))
        with open(f_path, "w") as f:
            f.writelines(lines_o)

    print(f"{jobid}: {l_path}: {prev_lc} -> {cur_lc}")
    return (prev_lc, cur_lc)


class Watcher:

    def __init__(self, job_record: str, kill_after: int = 3):
        self.stall_records = {}
        self.job_record = job_record
        assert os.path.isdir(job_record)
        self.kill_after = kill_after

    def get_ct(self):
        return datetime.datetime.now()

    def watch_once(self):
        sq_jobs = get_squeue_jobs()
        for dirpath, dirnames, filenames in os.walk(self.job_record):
            for f_record in filenames:
                if f_record in sq_jobs:
                    if f_record not in self.stall_records:
                        self.stall_records[f_record] = 0
                    last_size, this_size = getset_output_fsize(
                        f_record, self.job_record
                    )
                    if last_size == this_size and last_size >= 0:
                        self.stall_records[f_record] += 1
                    else:
                        self.stall_records[f_record] = 0

        self.stall_records = {
            key: self.stall_records[key]
            for key in set(sq_jobs.keys()).intersection(self.stall_records.keys())
        }

    def clean_jobs(self):
        for id, n_stall in self.stall_records.items():
            if n_stall >= self.kill_after:
                print(f"{self.get_ct()} !!! cancelling {id}", flush=True)
                os.system(f"scancel {id}")

    def watch(self, interval: float = 300, nMax: int = 2**32):
        iW = 0
        while iW < nMax:
            iW += 1
            self.watch_once()
            print(f"{self.get_ct()} Watch Results:", flush=True)
            for id, n_stall in self.stall_records.items():
                print(
                    f"{self.get_ct()} Watching: {id} has stalled [{n_stall}]times",
                    flush=True,
                )
            self.clean_jobs()
            sleep(interval)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--interval", type=float, default=300)
    ap.add_argument("-r", "--records", type=str, default=".job_record")
    ap.add_argument("-k", "--killafter", type=int, default=3)

    args = ap.parse_args()

    watcher = Watcher(args.records, args.killafter)
    watcher.watch(args.interval)
