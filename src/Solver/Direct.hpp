#pragma once

// #ifndef __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__HEADER_ON__
// #endif
#include "DNDS/JsonUtil.hpp"
#include "DNDS/Defines.hpp"
// #ifdef __DNDS_REALLY_COMPILING__HEADER_ON__
// #undef __DNDS_REALLY_COMPILING__
// #endif

#include <unordered_set>

#define JSON_ASSERT DNDS_assert
#include <json.hpp>

namespace DNDS::Direct
{
    struct DirectPrecControl
    {
        bool useDirectPrec = false;
        int32_t iluCode = 0;              // 0 for no fill, -1 for complete
        int32_t orderingCode = INT32_MIN; // INT32_MIN for auto 0 for natural, 1 for metis, 2 for MMD
        DNDS_NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_ORDERED_JSON(
            DirectPrecControl,
            useDirectPrec,
            iluCode,
            orderingCode)
        int getOrderingCode() const
        {
            return orderingCode == INT32_MIN
                       ? (iluCode == -1 ? 2 : 0) // auto decide
                       : orderingCode;
        }
        int getILUCode() const { return iluCode; }
    };
}

namespace DNDS::Direct
{
    struct SerialSymLUStructure
    {
        MPIInfo mpi;
        index N;
        using tLocalMatStruct = std::vector<std::vector<index>>;
        tLocalMatStruct lowerTriStructure;
        tLocalMatStruct upperTriStructure;
        tLocalMatStruct lowerTriStructureNew;
        tLocalMatStruct upperTriStructureNew;
        tLocalMatStruct lowerTriStructureNewInUpper;
        tLocalMatStruct upperTriStructureNewInLower;
        tLocalMatStruct cell2cellFaceVLocal2FullRowPos; // diag-lower-upper
        std::vector<index> localFillOrderingOld2New;
        std::vector<index> localFillOrderingNew2Old;

        SerialSymLUStructure(const MPIInfo &nMpi, index nN) : mpi(nMpi), N(nN){};

        index Num() const { return N; }

        index FillingReorderOld2New(index v)
        {
            return localFillOrderingOld2New.size() ? localFillOrderingOld2New[v] : v;
        }
        index FillingReorderNew2Old(index v)
        {
            return localFillOrderingNew2Old.size() ? localFillOrderingNew2Old[v] : v;
        }

        /**
         * @brief get symmetric symbolic matrix factorization over cell2cellFaceV
         *
         * @tparam TAdj
         * @param cell2cellFaceV is std::vector<std::vector<index>> -like which holds **symmetric** local adjacency
         * @param iluCode -1: full LU, 0,1,2... incomplete LU defined using expanded stencil
         */
        template <class TAdj>
        void ObtainSymmetricSymbolicFactorization(
            const TAdj &cell2cellFaceV,
            int iluCode)
        {
            std::vector<std::unordered_set<index>> cell2cellFaceVEnlarged;
            if (iluCode > 0) // do expansion of stencil
            {
                DNDS_assert(iluCode <= 5);
                cell2cellFaceVEnlarged.resize(this->Num());
                for (index iCell = 0; iCell < this->Num(); iCell++)
                    for (auto iCO : cell2cellFaceV[iCell])
                        cell2cellFaceVEnlarged[iCell].insert(iCO);
                for (int iFill = 0; iFill < iluCode; iFill++)
                {
                    for (index iCell = 0; iCell < this->Num(); iCell++)
                    {
                        std::unordered_set<index> newNeighbor;
                        for (auto iCO : cell2cellFaceVEnlarged[iCell])
                            for (auto iCOO : cell2cellFaceV[iCO])
                                newNeighbor.insert(iCOO);
                        for (auto iCN : newNeighbor)
                            cell2cellFaceVEnlarged[iCell].insert(iCN);
                    }
                }
                for (index iCell = 0; iCell < this->Num(); iCell++)
                    cell2cellFaceVEnlarged[iCell].erase(iCell); // this is redundant
            }

            std::vector<std::unordered_set<index>> triLowRows;
            std::vector<std::unordered_set<index>> triUppRows;
            std::vector<std::unordered_set<index>> midSymMatCols;
            triLowRows.resize(this->Num());
            triUppRows.resize(this->Num());
            midSymMatCols.resize(this->Num());
            index nnzOrig{this->Num()};

            for (index iCellP = 0; iCellP < this->Num(); iCellP++) // iterate over the columns
            {
                index iCell = this->FillingReorderNew2Old(iCellP);
                midSymMatCols[iCellP].insert(iCellP);
                for (auto iCellOther : cell2cellFaceV[iCell])
                {
                    index iCellOtherP = this->FillingReorderOld2New(iCellOther);
                    midSymMatCols[iCellP].insert(iCellOtherP); // assuming cell2cellFaceV is symmetric
                }
                nnzOrig += cell2cellFaceV[iCell].size();
            }

            for (index iCellP = 0; iCellP < this->Num(); iCellP++) // iterate over the columns
            {
                for (auto iCellOtherP : midSymMatCols[iCellP]) // emulate the symmetric factorization A L1L2L3...LN D LNT...L1T
                    if (iCellOtherP > iCellP)
                    {
                        index iCellOther = this->FillingReorderNew2Old(iCellOtherP);
                        for (auto iCellOtherPP : midSymMatCols[iCellP])
                            if (iCellOtherPP > iCellOtherP)
                            {
                                index iCellOtherOther = this->FillingReorderNew2Old(iCellOtherPP);
                                // to be always symmetric
                                // control over here to get incomplete LU structure
                                if (iluCode < 0 || (iluCode > 0 && cell2cellFaceVEnlarged[iCellOther].count(iCellOtherOther)))
                                {
                                    midSymMatCols[iCellOtherPP].insert(iCellOtherP); // upper part
                                    midSymMatCols[iCellOtherP].insert(iCellOtherPP); // lower part
                                }
                            }
                        triLowRows[iCellOtherP].insert(iCellP); // iCellP is iCol, iCellOtherP is iRow
                        triUppRows[iCellP].insert(iCellOtherP);
                    }
            }

            lowerTriStructure.resize(this->Num());
            upperTriStructure.resize(this->Num());
            lowerTriStructureNew.resize(this->Num());
            upperTriStructureNew.resize(this->Num());
            index nnzLower{0}, nnzUpper{0};
            for (index iCellP = 0; iCellP < this->Num(); iCellP++)
            {
                index iCell = this->FillingReorderNew2Old(iCellP);
                lowerTriStructure[iCell].reserve(triLowRows[iCellP].size());
                upperTriStructure[iCell].reserve(triUppRows[iCellP].size());
                lowerTriStructureNew[iCellP].reserve(triLowRows[iCellP].size());
                upperTriStructureNew[iCellP].reserve(triUppRows[iCellP].size());
                for (auto iCellOtherP : triLowRows[iCellP])
                    lowerTriStructureNew[iCellP].push_back(iCellOtherP);
                for (auto iCellOtherP : triUppRows[iCellP])
                    upperTriStructureNew[iCellP].push_back(iCellOtherP);
                std::sort(lowerTriStructureNew[iCellP].begin(), lowerTriStructureNew[iCellP].end());
                std::sort(upperTriStructureNew[iCellP].begin(), upperTriStructureNew[iCellP].end());
                for (auto iCellOtherP : lowerTriStructureNew[iCellP])
                    lowerTriStructure[iCell].push_back(this->FillingReorderNew2Old(iCellOtherP));
                for (auto iCellOtherP : upperTriStructureNew[iCellP])
                    upperTriStructure[iCell].push_back(this->FillingReorderNew2Old(iCellOtherP));

                nnzLower += lowerTriStructure[iCell].size();
                nnzUpper += upperTriStructure[iCell].size();
                // the lowerTriStructure and upperTriStructure's col indices are corresponding to
                // those of in *New Rows
                // col indices in *New Rows are sorted
            }
            if (mpi.rank == 0)
                log() << "Direct::ObtainSymmetricSymbolicFactorization(): Factorizing Done" << std::endl;
            MPISerialDo(mpi, [&]()
                        { log() << fmt::format("  ({}, {}/{})", mpi.rank,
                                               nnzLower + nnzUpper + this->Num(),
                                               nnzOrig)
                                << std::flush; });
            MPI::Barrier(mpi.comm);
            if (mpi.rank == 0)
                log() << std::endl;
            DNDS_assert(nnzLower == nnzUpper);

            // pre-search
            lowerTriStructureNewInUpper.resize(lowerTriStructureNew.size());
            for (index iP = 0; iP < this->Num(); iP++)
            {
                auto &&lowerRow = lowerTriStructureNew[iP];
                lowerTriStructureNewInUpper[iP].resize(lowerRow.size());
                for (int ijP = 0; ijP < lowerRow.size(); ijP++)
                {
                    index jP = lowerRow[ijP];
                    auto &&upperRow = upperTriStructureNew[jP];
                    auto ret = std::lower_bound(upperRow.begin(), upperRow.end(), iP);
                    // DNDS_assert(ret != upperRow.end()); // has to be found
                    lowerTriStructureNewInUpper[iP][ijP] = ret != upperRow.end()
                                                               ? (ret - upperRow.begin())
                                                               : -1; // !not found in upper
                }
            }

            // pre-search
            upperTriStructureNewInLower.resize(upperTriStructureNew.size());
            for (index iP = 0; iP < this->Num(); iP++)
            {
                auto &&upperRow = upperTriStructureNew[iP];
                upperTriStructureNewInLower[iP].resize(upperRow.size());
                for (int ijP = 0; ijP < upperRow.size(); ijP++)
                {
                    index jP = upperRow[ijP];
                    auto &&lowerRow = lowerTriStructureNew[jP];
                    auto ret = std::lower_bound(lowerRow.begin(), lowerRow.end(), iP);
                    // DNDS_assert(ret != upperRow.end()); // has to be found
                    upperTriStructureNewInLower[iP][ijP] = ret != lowerRow.end()
                                                               ? (ret - lowerRow.begin())
                                                               : -1; // !not found in upper
                }
            }

            cell2cellFaceVLocal2FullRowPos.resize(this->Num());
            for (index i = 0; i < this->Num(); i++)
            {
                index iP = this->FillingReorderOld2New(i);
                cell2cellFaceVLocal2FullRowPos[i].resize(cell2cellFaceV[i].size(), -1);
                for (int ic2c = 0; ic2c < cell2cellFaceV[i].size(); ic2c++)
                {
                    index j = cell2cellFaceV[i][ic2c];
                    index jP = this->FillingReorderOld2New(j);
                    if (jP < iP)
                    {
                        auto &&row = lowerTriStructureNew[iP];
                        auto ret = std::lower_bound(row.begin(), row.end(), jP);
                        DNDS_assert(ret != row.end());
                        cell2cellFaceVLocal2FullRowPos[i][ic2c] =
                            (ret - row.begin()) + 1;
                    }
                    else if (jP > iP)
                    {
                        auto &&row = upperTriStructureNew[iP];
                        auto ret = std::lower_bound(row.begin(), row.end(), jP);
                        DNDS_assert(ret != row.end());
                        cell2cellFaceVLocal2FullRowPos[i][ic2c] =
                            (ret - row.begin()) + lowerTriStructureNew[iP].size() + 1;
                    }
                }
            }
        }
    };

    template <class Derived, class tComponent, class tVec>
    struct LocalLUBase
    {
        ssp<Direct::SerialSymLUStructure> symLU;
        bool isDecomposed = false;

        LocalLUBase(ssp<Direct::SerialSymLUStructure> _symLU) : symLU(_symLU)
        {
        }

        virtual ~LocalLUBase()
        {
        }

        void GetDiag(index i);          // pure "virtual" do not implement
        void GetLower(index i, int ij); // pure "virtual" do not implement
        void GetUpper(index i, int ij); // pure "virtual" do not implement
        void InvertDiag(const tComponent &v);

        void InPlaceDecompose()
        {
            auto dThis = static_cast<Derived *>(this);
            for (index iP = 0; iP < symLU->Num(); iP++)
            {
                index i = symLU->FillingReorderNew2Old(iP);

                auto &&lowerRow = symLU->lowerTriStructureNew[iP];
                /*********************/
                // lower part
                for (int ijP = 0; ijP < lowerRow.size(); ijP++)
                {
                    auto jP = lowerRow[ijP];
                    DNDS_assert(jP < iP);
                    auto j = symLU->FillingReorderNew2Old(jP);
                    // handle last j's job
                    if (ijP > 0)
                    {
                        auto &&lowerRowJP = symLU->lowerTriStructureNew[jP];
                        iterateIdentical(
                            lowerRow.begin(), lowerRow.end(), lowerRowJP.begin(), lowerRowJP.end(),
                            [&](index kP, auto pos1, auto pos2)
                            {
                                if (kP > jP - 1)
                                    return true;                // early end
                                DNDS_assert(kP < symLU->Num()); // a safe guarantee
                                auto k = symLU->FillingReorderNew2Old(kP);
                                int jPInUpperPos = symLU->lowerTriStructureNewInUpper[jP][pos2];
                                if (jPInUpperPos != -1) // in case not symbolically symmetric
                                    dThis->GetLower(i, ijP) -=
                                        dThis->GetLower(i, pos1) * dThis->GetUpper(k, jPInUpperPos);
                                // if (iP < 3)
                                // log() << fmt::format("Lower Add at {},{},{} === {}", iP, jP - 1, kP, (dThis->GetLower(i, pos1) * dThis->GetUpper(k, jPInUpperPos))(0)) << std::endl;
                                return false;
                            });
                    }

                    // auto luDiag = dThis->GetDiag(j).fullPivLu();
                    // tComponent Aij = luDiag.solve(dThis->GetLower(i, ijP));
                    dThis->GetLower(i, ijP) *= dThis->GetDiag(j);
                }
                /*********************/
                // diag part
                for (int ikP = 0; ikP < lowerRow.size(); ikP++)
                {
                    auto kP = lowerRow[ikP];
                    auto k = symLU->FillingReorderNew2Old(kP);
                    int iPInUpperPos = symLU->lowerTriStructureNewInUpper[iP][ikP];
                    if (iPInUpperPos != -1)
                        dThis->GetDiag(i) -= dThis->GetLower(i, ikP) * dThis->GetUpper(k, iPInUpperPos);
                }
                
                dThis->GetDiag(i) = dThis->InvertDiag(dThis->GetDiag(i)); // * note here only stores
                /*********************/
                // upper part
                auto &&upperRow = symLU->upperTriStructureNew[iP];
                for (int ijP = 0; ijP < upperRow.size(); ijP++)
                {
                    auto jP = upperRow[ijP];
                    DNDS_assert(jP > iP);
                    auto j = symLU->FillingReorderNew2Old(jP);
                    auto &&lowerRowJP = symLU->lowerTriStructureNew[jP];

                    iterateIdentical(
                        lowerRow.begin(), lowerRow.end(), lowerRowJP.begin(), lowerRowJP.end(),
                        [&](index kP, auto pos1, auto pos2)
                        {
                            if (kP >= iP)
                                return true;
                            DNDS_assert(kP < symLU->Num());
                            auto k = symLU->FillingReorderNew2Old(kP);
                            int jPInUpperPos = symLU->lowerTriStructureNewInUpper[jP][pos2];
                            if (jPInUpperPos != -1)
                                dThis->GetUpper(i, ijP) -= dThis->GetLower(i, pos1) * dThis->GetUpper(k, jPInUpperPos);
                            // if (iP < 3)
                            // log() << fmt::format("Upper Add at {},{},{} === {}", iP, jP, kP,
                            //                      (this->GetLower(i, pos1) * this->GetUpper(k, jPInUpperPos))(0))
                            // << std::endl;
                            return false;
                        });
                }
            }
            isDecomposed = true;
        }

        void MatMul(tVec &x, tVec &result)
        {
            auto dThis = static_cast<Derived *>(this);
            DNDS_assert(!isDecomposed);
            for (index iCell = 0; iCell < symLU->Num(); iCell++)
            {
                result[iCell] = dThis->GetDiag(iCell) * x[iCell];
                for (int ij = 0; ij < symLU->lowerTriStructure[iCell].size(); ij++)
                    result[iCell] += dThis->GetLower(iCell, ij) * x[symLU->lowerTriStructure[iCell][ij]];
                for (int ij = 0; ij < symLU->upperTriStructure[iCell].size(); ij++)
                    result[iCell] += dThis->GetUpper(iCell, ij) * x[symLU->upperTriStructure[iCell][ij]];
            }
        }

        void Solve(tVec &b, tVec &result)
        {
            auto dThis = static_cast<Derived *>(this);
            DNDS_assert(isDecomposed);
            for (index iP = 0; iP < symLU->Num(); iP++)
            {
                index i = symLU->FillingReorderNew2Old(iP);
                result[i] = b[i];
                auto &&lowerRowOld = symLU->lowerTriStructure[i];
                for (int ij = 0; ij < lowerRowOld.size(); ij++)
                {
                    index j = lowerRowOld[ij];
                    result[i] -= dThis->GetLower(i, ij) * result[j];
                }
            }
            for (index iP = symLU->Num() - 1; iP >= 0; iP--)
            {
                index i = symLU->FillingReorderNew2Old(iP);
                auto &&upperRowOld = symLU->upperTriStructure[i];
                for (int ij = 0; ij < upperRowOld.size(); ij++)
                {
                    index j = upperRowOld[ij];
                    result[i] -= dThis->GetUpper(i, ij) * result[j];
                }
                // auto luDiag = dThis->GetDiag(i).fullPivLu();
                // result[i] = luDiag.solve(result[i]);
                result[i] = dThis->GetDiag(i) * result[i];
            }
        }
    };

    template <class Derived, class tComponent, class tVec>
    struct LocalLDLTBase
    {
        ssp<Direct::SerialSymLUStructure> symLU;
        bool isDecomposed = false;

        LocalLDLTBase(ssp<Direct::SerialSymLUStructure> _symLU) : symLU(_symLU)
        {
        }

        virtual ~LocalLDLTBase()
        {
        }

        void GetDiag(index i);
        void GetLower(index i, int ij);
        void InvertDiag(const tComponent& v);

        void InPlaceDecompose()
        {
            auto dThis = static_cast<Derived *>(this);
            std::vector<tComponent> diagNoInv(symLU->Num());
            for (index iP = 0; iP < symLU->Num(); iP++)
            {
                index i = symLU->FillingReorderNew2Old(iP);

                auto &&lowerRow = symLU->lowerTriStructureNew[iP];
                /*********************/
                // lower part
                for (int ijP = 0; ijP < lowerRow.size(); ijP++)
                {
                    auto jP = lowerRow[ijP];
                    DNDS_assert(jP < iP);
                    auto j = symLU->FillingReorderNew2Old(jP);
                    // handle last j's job
                    if (ijP > 0)
                    {
                        auto &&lowerRowJP = symLU->lowerTriStructureNew[jP];
                        iterateIdentical(
                            lowerRow.begin(), lowerRow.end(), lowerRowJP.begin(), lowerRowJP.end(),
                            [&](index kP, auto pos1, auto pos2)
                            {
                                if (kP > jP - 1)
                                    return true;                // early end
                                DNDS_assert(kP < symLU->Num()); // a safe guarantee
                                auto k = symLU->FillingReorderNew2Old(kP);
                                dThis->GetLower(i, ijP) -=
                                    dThis->GetLower(i, pos1) * diagNoInv[k] * dThis->GetLower(j, pos2).transpose();
                                // if (iP < 3)
                                // log() << fmt::format("Lower Add at {},{},{} === {}", iP, jP - 1, kP, (dThis->GetLower(i, pos1) * dThis->GetUpper(k, jPInUpperPos))(0)) << std::endl;
                                return false;
                            });
                    }

                    // auto luDiag = dThis->GetDiag(j).fullPivLu();
                    // tComponent Aij = luDiag.solve(dThis->GetLower(i, ijP));
                    dThis->GetLower(i, ijP) *= dThis->GetDiag(j);
                }
                /*********************/
                // diag part
                for (int ikP = 0; ikP < lowerRow.size(); ikP++)
                {
                    auto kP = lowerRow[ikP];
                    auto k = symLU->FillingReorderNew2Old(kP);
                    dThis->GetDiag(i) -= dThis->GetLower(i, ikP) * diagNoInv[k] * dThis->GetLower(i, ikP).transpose();
                }
                diagNoInv[i] = dThis->GetDiag(i);
                dThis->GetDiag(i) = dThis->InvertDiag(dThis->GetDiag(i)); // * note here only stores the inverse
            }
            isDecomposed = true;
        }

        void MatMul(tVec &x, tVec &result)
        {
            auto dThis = static_cast<Derived *>(this);
            DNDS_assert(!isDecomposed); // being before the decomposition
            for (index iCell = 0; iCell < symLU->Num(); iCell++)
            {
                result[iCell] = dThis->GetDiag(iCell) * x[iCell];
                for (int ij = 0; ij < symLU->lowerTriStructure[iCell].size(); ij++)
                    result[iCell] += dThis->GetLower(iCell, ij) * x[symLU->lowerTriStructure[iCell][ij]];
            }
            for (index iCell = 0; iCell < symLU->Num(); iCell++)
            {
                for (int ij = 0; ij < symLU->lowerTriStructure[iCell].size(); ij++)
                    result[symLU->lowerTriStructure[iCell][ij]] += dThis->GetLower(iCell, ij).transpose() * x[iCell]; // transposed mat-vec
            }
        }

        void Solve(tVec &b, tVec &result)
        {
            auto dThis = static_cast<Derived *>(this);
            DNDS_assert(isDecomposed);
            for (index iP = 0; iP < symLU->Num(); iP++)
            {
                index i = symLU->FillingReorderNew2Old(iP);
                result[i] = b[i];
                auto &&lowerRowOld = symLU->lowerTriStructure[i];
                for (int ij = 0; ij < lowerRowOld.size(); ij++)
                {
                    index j = lowerRowOld[ij];
                    result[i] -= dThis->GetLower(i, ij) * result[j];
                }
            }
            for (index i = 0; i < symLU->Num(); i++)
            {
                result[i] = dThis->GetDiag(i) * result[i];
            }
            for (index iP = symLU->Num() - 1; iP >= 0; iP--)
            {
                index i = symLU->FillingReorderNew2Old(iP);
                auto &&upperRow = symLU->upperTriStructureNew[iP];
                for (int ij = 0; ij < upperRow.size(); ij++)
                {
                    index jP = upperRow[ij];
                    index j = symLU->FillingReorderNew2Old(jP);
                    index ji = symLU->upperTriStructureNewInLower[iP][ij];
                    DNDS_assert(ji != -1); // has to be found
                    result[i] -= dThis->GetLower(j, ji).transpose() * result[j];
                }
            }
        }
    };
}