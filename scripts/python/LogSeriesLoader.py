import pandas as pd
import os, sys, re
import numpy as np


def GetDNDSLogSeries(fname_first):
    fname_first_file = os.path.basename(fname_first)
    base_dir = os.path.dirname(fname_first)
    match_result = re.match(r"(.+).log", fname_first_file)
    if match_result is None:
        raise ValueError(f"file input not a good pattern: {fname_first_file}")
    match_base = match_result.group(1)

    series = {}
    if os.path.isfile(fname_first):
        series[0] = fname_first_file
    else:
        raise RuntimeWarning(f"first file not found: [{fname_first}]")

    for f in os.listdir(base_dir):
        if f == fname_first_file:
            continue
        if not os.path.isfile(os.path.join(base_dir, f)):
            continue
        match_result = re.match(rf"{match_base}[_]*(\d+).log", f)
        if match_result is not None:
            id = int(match_result.group(1))
            if id not in series:
                series[id] = f
            else:
                raise RuntimeWarning(
                    f"the file ordinals are not consistent: [{f}], [{series[id]}]"
                )
    series_list = sorted(series.items(), key=lambda t: t[0])
    ids = np.array([t[0] for t in series_list], dtype=np.int64)
    max_id = ids.max()
    idsIfFull = np.zeros((max_id + 1), dtype=np.int32)
    idsIfFull[ids] = 1
    if not idsIfFull.all():
        missing = np.arange(max_id + 1, dtype=np.int64)[idsIfFull == 0]
        raise ResourceWarning(f"missing items: [{missing}]")

    mtime_latest = max(
        [os.path.getmtime(os.path.join(base_dir, fname)) for id, fname in series_list]
    )

    return base_dir, series_list, mtime_latest


def LoadDNDSLogSeriesMergePandas(fname_first, use_cache=True, save_cache=True) -> pd.DataFrame:
    base_dir, series_list, mtime_latest = GetDNDSLogSeries(fname_first)

    cache_file_name = fname_first + ".pkl"
    if use_cache:
        if os.path.isfile(cache_file_name):
            if mtime_latest < os.path.getmtime(cache_file_name):
                return pd.read_pickle(cache_file_name)

    data = []
    for iFile, (id, fname) in enumerate(series_list):
        df = pd.read_csv(os.path.join(base_dir, fname), dtype=np.float64)
        data.append(df)

        print(f"File [{iFile}] [{fname}] reading complete")

    df = pd.concat(data, ignore_index=True)
    if save_cache:
        df.to_pickle(cache_file_name)
    return df


if __name__ == "__main__":
    print(
        LoadDNDSLogSeriesMergePandas(
            "data/CylinderB1_L8U2_Rawstart2_PdT05_restart1_TH3_.log"
        )
    )
