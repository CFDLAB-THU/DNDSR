import h5py
import argparse, os
import numpy as np
import scipy as sp
import scipy.sparse as spsp
import scipy.sparse.linalg as spsplinalg


class DNDS_CSR_Data:

    def __init__(self, array: h5py.Group):
        d = DNDS_CSR_Data.__read_h5_csr_array(array)
        self.sizes = d["sizes"]
        self.size = d["size"]
        self.data = d["data"]
        self.data_rank_offset = d["data_rank_offset"]
        self.row_start = d["row_start"]
        self.row_start_offset = d["row_start_offset"]
        self.row_start_full = d["row_start_full"]

    def __getitem__(self, i: int):
        return self.data[self.row_start_full[i] : self.row_start_full[i + 1],]

    def __len__(self):
        return self.size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def __read_h5_csr_array(cls, array: h5py.Group):
        data = np.array(array["data"])
        data_rank_offset = np.array(array["data::rank_offsets"])
        row_start = np.array(array["pRowStart"])
        row_start_offset = np.array(array["pRowStart::rank_offsets"])
        row_start_full = []
        sizes = np.array(array["size"])
        size_global = sizes.sum()
        nRank = len(data_rank_offset) - 1
        assert sizes.size == nRank
        for iRank, data_offset in enumerate(data_rank_offset):
            if iRank == nRank:
                continue
            row_start_piece = row_start[
                row_start_offset[iRank] : row_start_offset[iRank + 1]
            ]
            assert row_start_piece[0] == 0
            if iRank > 0:
                assert data_offset == row_start_full[iRank - 1][-1]
                row_start_full[iRank - 1] = row_start_full[iRank - 1][
                    0:-1
                ]  # remove last
            row_start_full.append(row_start_piece + data_offset)
        row_start_full = np.concatenate(row_start_full, axis=0)
        assert size_global == row_start_full.shape[0] - 1
        assert row_start_full[-1] == data.shape[0]

        return {
            "sizes": sizes,
            "size": size_global,
            "data": data,
            "data_rank_offset": data_rank_offset,
            "row_start": row_start,
            "row_start_offset": row_start_offset,
            "row_start_full": row_start_full,
        }


def read_matrix(file_name):
    with h5py.File(file_name, "r") as f:
        assert len(f.keys()) == 1
        matrix_root = f[list(f.keys())[0]]
        print(matrix_root["cell2cellFace"]["father"])
        cell2cellFace = DNDS_CSR_Data(matrix_root["cell2cellFace"]["father"])

        matrix_derived_array_sig = matrix_root["matrixAAInvB"]["father"].attrs[
            "DerivedType"
        ]
        print(matrix_derived_array_sig)
        assert "ArrayEigenUniMatrixBatch__-1_-1" == matrix_derived_array_sig.decode(
            "UTF-8"
        )
        mat_Rows = matrix_root["matrixAAInvB"]["father"].attrs["row_dynamic"]
        mat_Cols = matrix_root["matrixAAInvB"]["father"].attrs["col_dynamic"]
        matrixAAInvBData = DNDS_CSR_Data(matrix_root["matrixAAInvB"]["father"]["array"])

        assert cell2cellFace.size == matrixAAInvBData.size

        matrixAAInvB = []
        for iCell, c2cRow in enumerate(cell2cellFace):
            cellMatRow = {}
            mat_data = matrixAAInvBData[iCell]
            rowSize = c2cRow.size
            row_matrices = mat_data.reshape(
                (rowSize + 1, mat_Cols, mat_Rows)
            )  # here is Row-Major
            cellMatRow[iCell] = row_matrices[0, :, :].reshape(mat_Cols, mat_Rows).T
            for ic2c, iCellOther in enumerate(c2cRow):
                cellMatRow[int(iCellOther)] = (
                    row_matrices[ic2c + 1, :, :].reshape(mat_Cols, mat_Rows).T
                )
            matrixAAInvB.append(cellMatRow)

    return matrixAAInvB, (mat_Rows, mat_Cols)


def matrix_get_no_inverse(
    matrixAAInvB: list[dict[int, np.ndarray]],
) -> list[dict[int, np.ndarray]]:
    matrixAB = []
    for iCell, AAInvBRow in enumerate(matrixAAInvB):
        ABRow = {}
        AInv = AAInvBRow[iCell]
        A = np.linalg.pinv(AInv)
        ABRow[iCell] = A
        for iCellOther, AInvB in AAInvBRow.items():
            if iCellOther != iCell:
                B = A @ AInvB
                ABRow[iCellOther] = B
        matrixAB.append(ABRow)

    return matrixAB


def matrix_get_diag_max_cond(A: list[dict[int, np.ndarray]]):
    Aij_cond_max = 0
    for iCell in range(len(A)):
        Aij_cond_max = max(np.linalg.cond(A[iCell][iCell]), Aij_cond_max)
    return Aij_cond_max


def matrix_get_sym_deviation_max(matrixAB: list[dict[int, np.ndarray]]):
    max_sym_deviation = 0
    for iCell, ABRow in enumerate(matrixAB):
        for iCellOther, B in ABRow.items():
            if iCellOther != -9223372036854775808:
                max_sym_deviation = max(
                    max_sym_deviation,
                    np.linalg.norm(B - matrixAB[iCellOther][iCell].T)
                    / np.linalg.norm(B),
                )
    return max_sym_deviation


def matrix_to_jacobian_preconditioned(
    matrixAB: list[dict[int, np.ndarray]],
) -> list[dict[int, np.ndarray]]:
    matrixDIAB = []
    for iCell, ABRow in enumerate(matrixAB):
        DIABRow = {}
        A = ABRow[iCell]
        AInv = np.linalg.pinv(A)
        DIABRow[iCell] = np.eye(*A.shape)
        for iCellOther, B in ABRow.items():
            if iCellOther != iCell:
                DIB = AInv @ B
                DIABRow[iCellOther] = DIB
        matrixDIAB.append(DIABRow)
    return matrixDIAB


def matrix_get_component(
    matrixAB: list[dict[int, np.ndarray]], part: str
) -> list[dict[int, np.ndarray]]:
    matrixPart = []

    def ij_select(i, j) -> bool:
        if part == "L":
            return j < i
        if part == "U":
            return j > i
        if part == "LD":
            return j <= i
        if part == "UD":
            return j >= i
        if part == "D":
            return j == i
        if part == "LU":
            return j != i

    for iCell, ABRow in enumerate(matrixAB):
        PartRow = {}
        for iCellOther, B in ABRow.items():
            if ij_select(iCell, iCellOther):
                PartRow[iCellOther] = B
        matrixPart.append(PartRow)
    return matrixPart


def matrix_to_bsr(matrixAB: list[dict[int, np.ndarray]], blk_r, blk_c):
    nCell = len(matrixAB)
    data_entries = [
        (iCell, iCellOther, m)
        for iCell, row in enumerate(matrixAB)
        for iCellOther, m in row.items()
        if iCellOther != -9223372036854775808
    ]
    # ij = ([t[0] for t in data_entries], [t[1] for t in data_entries])

    data = [t[2] for t in data_entries]
    indices = []
    indptr = [0]
    for iCell, row in enumerate(matrixAB):
        for iCellOther, m in row.items():
            if iCellOther != -9223372036854775808:
                indices.append(iCellOther)
        indptr.append(len(indices))

    return spsp.bsr_array(
        (data, indices, indptr),
        shape=(nCell * blk_r, nCell * blk_c),
        blocksize=(blk_r, blk_c),
        dtype=np.float64,
    )


def matrix_to_bsr_Jacobian_mat(matrixAB: list[dict[int, np.ndarray]], blk_r, blk_c):
    # x = b - D^-1(L + U)x
    D = matrix_get_component(matrixAB, "D")
    LU = matrix_get_component(matrixAB, "LU")
    for iC, r in enumerate(D):
        for iCOther, m in r.items():
            D[iC][iCOther] = np.linalg.pinv(m)
    bsr_DI = matrix_to_bsr(D, blk_r=blk_r, blk_c=blk_c)
    bsr_LU = matrix_to_bsr(LU, blk_r=blk_r, blk_c=blk_c)
    return bsr_DI @ bsr_LU


def matrix_to_lin_SOR_mat(
    matrixAB: list[dict[int, np.ndarray]], blk_r, blk_c, omega=1.0
):
    from scipy.sparse import bsr_matrix, csc_matrix
    from scipy.sparse.linalg import spilu, eigs, LinearOperator

    # (D + omega L)x = omega b - [omega U + (omega-1) D] x
    D = matrix_get_component(matrixAB, "D")
    L = matrix_get_component(matrixAB, "L")
    U = matrix_get_component(matrixAB, "U")

    bsr_D = matrix_to_bsr(D, blk_r=blk_r, blk_c=blk_c)
    bsr_L = matrix_to_bsr(L, blk_r=blk_r, blk_c=blk_c)
    bsr_U = matrix_to_bsr(U, blk_r=blk_r, blk_c=blk_c)
    DoL = bsr_D + omega * bsr_L
    oUo1D = omega * bsr_U + (omega - 1) * bsr_D

    ilu = spilu(csc_matrix(DoL))

    def matvec(x):
        Bx = oUo1D @ x
        return ilu.solve(Bx)

    return LinearOperator(shape=bsr_D.shape, matvec=matvec, dtype=bsr_D.dtype)


def scipy_bsr_get_cond(A: spsp.bsr_array):

    from scipy.sparse.linalg import lobpcg, LinearOperator, spilu

    print(f"{A.shape}, nnz is {A.nnz}")
    ilu = spilu(A, fill_factor=20)
    AInvOp = LinearOperator(matvec=ilu.solve, dtype=A.dtype, shape=A.shape)
    evals, evecs = spsplinalg.eigs(
        A, k=3, which="LM", maxiter=50, tol=1e-5, sigma=0.0, OPinv=AInvOp
    )
    print(evals)
    minEVal = np.abs(evals).min()
    evals, evecs = spsplinalg.eigs(A, k=3, which="LM", maxiter=500, tol=1e-5)
    print(evals)
    maxEVal = np.abs(evals).max()
    condEst = maxEVal / minEVal
    print(f"Cond Est: {condEst : .3e} ||| rho: {maxEVal:.4g}")
    return condEst, maxEVal


def scipy_bsr_get_rho(A: spsp.bsr_array):

    print(f"{A.shape}, nnz is {A.nnz}")
    evals, evecs = spsplinalg.eigs(A, k=3, which="LM", maxiter=500, tol=1e-5)
    print(evals)
    maxEVal = np.abs(evals).max()
    print(f"rho: {maxEVal:.4g}")
    return maxEVal


def scipy_lin_get_rho(A: spsp.bsr_array):

    print(f"{A.shape}")
    evals, evecs = spsplinalg.eigs(A, k=3, which="LM", maxiter=500, tol=1e-5)
    print(evals)
    maxEVal = np.abs(evals).max()
    print(f"rho: {maxEVal:.4g}")
    return maxEVal


if __name__ == "__main__":

    parser = argparse.ArgumentParser("RecMatCondition")
    parser.add_argument("file_name")

    args = parser.parse_args()
    assert os.path.exists(args.file_name)

    matrixAAInvB, (mat_Rows, mat_Cols) = read_matrix(args.file_name)
    matrixAB = matrix_get_no_inverse(matrixAAInvB)
    print(f"A Cond Max:   {matrix_get_diag_max_cond(matrixAB):.2e}")
    print(f"max Sym deviation {matrix_get_sym_deviation_max(matrixAB):.2e}")

    matrixDIAB = matrix_to_jacobian_preconditioned(matrixAB)
    bsr_matrixAB = matrix_to_bsr(matrixAB, blk_r=mat_Rows, blk_c=mat_Cols)
    print("\n=== Matirx A ===")
    condA, rhoA = scipy_bsr_get_cond(bsr_matrixAB)
    print("\n=== Matirx D^-1 A ===")
    condDIA, rhoDIA = scipy_bsr_get_cond(
        matrix_to_bsr(matrixDIAB, blk_r=mat_Rows, blk_c=mat_Cols)
    )
    print("\n=== Matirx AJ ===")
    bsr_matrixJacobianIter = matrix_to_bsr_Jacobian_mat(
        matrixAB, blk_r=mat_Rows, blk_c=mat_Cols
    )
    rhoAJ = scipy_bsr_get_rho(bsr_matrixJacobianIter)

    print("\n=== Matirx AG ===")
    rhoAJ = scipy_lin_get_rho(
        matrix_to_lin_SOR_mat(matrixAB, blk_r=mat_Rows, blk_c=mat_Cols, omega=1.0)
    )
