import h5py
import re
import numpy as np
from contextlib import contextmanager


@contextmanager
def get_restart_data(fname: str):
    with h5py.File(fname, mode="r") as doc:
        father_sig = doc["u"]["father"]["array"].attrs["array_sig"]
        m = re.match(r".+_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)", str(father_sig))
        sig_sizes = tuple([int(i) for i in m.groups()])
        row_size = sig_sizes[1]
        if not row_size > 0:
            raise ValueError(f"does not support dynamic rowsize data now")
        tot_size = doc["u"]["father"]["array"]["data"].size
        size = tot_size / row_size
        if tot_size % row_size:
            raise ValueError(
                f"insane, tot_size [{tot_size}] not multiply of row_size [{row_size}]"
            )

        if "cell2cellOrig" not in doc:
            raise ValueError(f"cell2cellOrig not in file")
        c2cOSize = doc["cell2cellOrig"]["father"]["data"].size
        if size != c2cOSize:
            raise ValueError(f"cell2cellOrig size [{c2cOSize}] != [{size}]")

        cell2cellOrigArr = doc["cell2cellOrig"]["father"]
        uArr = doc["u"]["father"]["array"]
        print(uArr.keys())
        yield doc, uArr, cell2cellOrigArr, (size, row_size)


def restratFileRankRemap(src: str, tmp: str, dst: str):
    # fmt: off
    with get_restart_data(src) as (doc, uArr,  cell2cellOrigArr,  (size,  row_size )),\
         get_restart_data(tmp) as (docT, uArrT, cell2cellOrigArrT, (sizeT, row_sizeT)):
    # fmt: on
        if sizeT != size or row_sizeT != row_size:
             raise ValueError(f"src,tmp size not matching: size [{size}/{sizeT}], [{row_size}/{row_sizeT}]")
        c2cO = np.array(cell2cellOrigArr["data"])
        c2cOT = np.array(cell2cellOrigArrT["data"])
        uData = np.array(uArr["data"])
        uData = uData.reshape((-1, row_size))
        uData[c2cO, :] = uData
        uData = uData[c2cOT, :]
        with h5py.File(dst, 'w') as fout:
            fout.copy(docT["u"], "u")
            fout.copy(docT["cell2cellOrig"], "cell2cellOrig")
            fout["u"]["father"]["array"]["data"][:] = uData.reshape((-1,))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("restartFileRankRemap")
    parser.add_argument("src", help="input dndsr.h5 file")
    parser.add_argument("tmp", help="reference template dndsr.h5 file")
    parser.add_argument("dst", help="output dndsr.h5 file")
    args = parser.parse_args()
    restratFileRankRemap(
        src=args.src,
        tmp=args.tmp,
        dst=args.dst
    )
