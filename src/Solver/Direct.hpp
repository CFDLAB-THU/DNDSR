#pragma once

#include "DNDS/Core/RowRanges.hpp"

// #ifndef __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__
// #define __DNDS_REALLY_COMPILING__HEADER_ON__
// #endif
#include "DNDS/Serializer/JsonUtil.hpp"
#include "DNDS/Config/ConfigParam.hpp"
#include "DNDS/Core/Defines.hpp"
// #ifdef __DNDS_REALLY_COMPILING__HEADER_ON__
// #undef __DNDS_REALLY_COMPILING__
// #endif

#include <unordered_set>

namespace DNDS::Direct
{
    struct DirectPrecControl
    {
        bool useDirectPrec = false;
        int32_t iluCode = 0;              // 0 for no fill, -1 for complete
        int32_t orderingCode = INT32_MIN; // INT32_MIN for auto 0 for natural, 1 for metis, 2 for MMD

        DNDS_DECLARE_CONFIG(DirectPrecControl)
        {
            // clang-format off
            DNDS_FIELD(useDirectPrec, "Enable direct preconditioner");
            DNDS_FIELD(iluCode,       "ILU fill level: 0=no fill, -1=complete");
            DNDS_FIELD(orderingCode,  "Ordering: INT32_MIN=auto, 0=natural, 1=metis, 2=MMD");
            // clang-format on
        }

        [[nodiscard]] int getOrderingCode() const
        {
            return orderingCode == INT32_MIN
                       ? (iluCode == -1 ? 2 : 0) // auto decide
                       : orderingCode;
        }
        [[nodiscard]] int getILUCode() const { return iluCode; }
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

        std::vector<index> localPartStarts;
        /// End of the core prefix in each local part, in factor positions.
        /// Empty unless an explicit core-first permutation was prepared.
        std::vector<index> localPartCoreEnds;
        uint64_t generation = 0;
        bool hasSymbolicPattern = false;

        SerialSymLUStructure(const MPIInfo &nMpi, index nN) : mpi(nMpi), N(nN){};

        [[nodiscard]] index Num() const { return N; }

        index FillingReorderOld2New(index v)
        {
            return localFillOrderingOld2New.size() ? localFillOrderingOld2New[v] : v;
        }
        index FillingReorderNew2Old(index v)
        {
            return localFillOrderingNew2Old.size() ? localFillOrderingNew2Old[v] : v;
        }

        /// Stable partition the requested local ordering before symbolic fill.
        /// Preserve each local part and its relative ordering within both sets.
        /// This changes the factorization permutation, not mesh/DOF storage.
        void SetCoreFirstOrdering(const std::vector<index> &core, std::vector<index> parts = {})
        {
            DNDS_check_throw_info(!hasSymbolicPattern && N >= 0, "core ordering must precede symbolic factorization");
            if (parts.empty())
                parts = {0, N};
            DNDS_check_throw_info(parts.size() >= 2 && parts.front() == 0 && parts.back() == N &&
                                      std::is_sorted(parts.begin(), parts.end()),
                                  "invalid local parts for core ordering");
            DNDS_check_throw_info(localFillOrderingNew2Old.empty() || index(localFillOrderingNew2Old.size()) == N,
                                  "invalid existing factor permutation size");
            std::vector<unsigned char> selected(N, 0), seen(N, 0);
            for (auto iCell : core)
            {
                DNDS_check_throw_info(iCell >= 0 && iCell < N && !selected[iCell], "invalid or duplicate core cell");
                selected[iCell] = 1;
            }
            std::vector<index> order, ends;
            order.reserve(N);
            for (index part = 0; part + 1 < index(parts.size()); ++part)
            {
                for (int group = 1; group >= 0; --group)
                {
                    for (index pos = parts[part]; pos < parts[part + 1]; ++pos)
                    {
                        const index iCell = FillingReorderNew2Old(pos);
                        DNDS_check_throw_info(iCell >= parts[part] && iCell < parts[part + 1], "permutation crosses local part");
                        if (selected[iCell] == group)
                        {
                            DNDS_check_throw_info(!seen[iCell], "duplicate cell in factor permutation");
                            seen[iCell] = 1;
                            order.push_back(iCell);
                        }
                    }
                    if (group == 1)
                        ends.push_back(index(order.size()));
                }
            }
            std::vector<index> inverse(N);
            for (index pos = 0; pos < N; ++pos)
                inverse[order[pos]] = pos;
            localFillOrderingNew2Old = std::move(order);
            localFillOrderingOld2New = std::move(inverse);
            localPartStarts = std::move(parts);
            localPartCoreEnds = std::move(ends);
            ++generation;
        }

        /// Rows whose triangular result is ready from the supplied RHS rows.
        /// Factor support is not a bound on transitive solve dependencies.
        /// Returns old cell indices in forward or reverse factor order.
        std::vector<index> ReadySolveRows(const std::vector<index> &readyRhs, bool forward) const
        {
            DNDS_check_throw_info(hasSymbolicPattern, "triangular readiness needs a symbolic pattern");
            std::vector<unsigned char> rhs(N, 0), ready(N, 0);
            for (auto iCell : readyRhs)
            {
                DNDS_check_throw_info(iCell >= 0 && iCell < N && !rhs[iCell], "invalid or duplicate ready RHS row");
                rhs[iCell] = 1;
            }
            std::vector<index> result;
            for (index scan = 0; scan < N; ++scan)
            {
                const index pos = forward ? scan : N - 1 - scan;
                const index iCell = localFillOrderingNew2Old.empty() ? pos : localFillOrderingNew2Old[pos];
                bool available = rhs[iCell];
                const auto &dependencies = forward ? lowerTriStructureNew[pos] : upperTriStructureNew[pos];
                for (auto other : dependencies)
                    available = available && ready[other];
                ready[pos] = available;
                if (available)
                    result.push_back(iCell);
            }
            return result;
        }

        /**
         * @brief get symmetric symbolic matrix factorization over cell2cellFaceV
         *
         * @tparam TAdj
         * @param cell2cellFaceV is std::vector<std::vector<index>> -like which holds **symmetric** local adjacency
         * @param iluCode -1: full LU, 0,1,2... incomplete LU defined using expanded stencil
         * @param report Preserve communicator-synchronized legacy diagnostic
         * output when true. False performs local construction without those
         * diagnostic MPI collectives. All input rows must already be local.
         */
        template <class TAdj>
        void ObtainSymmetricSymbolicFactorization(
            const TAdj &cell2cellFaceV,
            const std::vector<index> localPartStarts_in,
            int iluCode, bool report = true, const std::vector<WorkerRowRanges> &physicalWorkers = {})
        {
            hasSymbolicPattern = false;
            localPartStarts = localPartStarts_in;
            { // assert on partition
                int nParts = localPartStarts.size() - 1;
                if (nParts >= 1)
                    DNDS_assert(localPartStarts[0] == 0 && localPartStarts.back() == this->Num());
                else // handle when localPartStarts_in is not large enough
                {
                    localPartStarts.resize(2);
                    localPartStarts[0] = 0;
                    localPartStarts[1] = this->Num();
                }

                DNDS_assert(physicalWorkers.empty() || physicalWorkers.size() == size_t(nParts));
                for (int part = 0; part < nParts; ++part)
                {
                    const auto begin = localPartStarts[part], end = localPartStarts[part + 1];
                    const auto worker = physicalWorkers.empty() ? WorkerRowRanges{{begin, end}, {}} : physicalWorkers[part];
                    DNDS_assert(end - begin == worker.Size());
                    for (index pos = begin; pos < end; ++pos)
                    {
                        const auto row = FillingReorderNew2Old(pos);
                        DNDS_assert(worker.Contains(row));
                        DNDS_assert(FillingReorderOld2New(row) == pos);
                    }
                }
            }

            std::vector<std::unordered_set<index>> cell2cellFaceVEnlarged;
            if (iluCode > 0) // do expansion of stencil
            {
                DNDS_assert(iluCode <= 5);
                cell2cellFaceVEnlarged.resize(this->Num());
                for (index iCell = 0; iCell < this->Num(); iCell++)
                    for (auto iCO : cell2cellFaceV[iCell])
                        if (iCO != iCell)
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
                    if (iCellOther == iCell)
                        continue;
                    index iCellOtherP = this->FillingReorderOld2New(iCellOther);
                    midSymMatCols[iCellP].insert(iCellOtherP); // assuming cell2cellFaceV is symmetric
                    nnzOrig++;
                }
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
                // resize() preserves existing rows on a rebuild. Clear their
                // incidences while retaining capacity before publishing the
                // next symbolic generation; otherwise pushes duplicate them.
                lowerTriStructure[iCell].clear();
                upperTriStructure[iCell].clear();
                lowerTriStructureNew[iCellP].clear();
                upperTriStructureNew[iCellP].clear();
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
            if (report)
            {
                if (mpi.rank == 0)
                    log() << "Direct::ObtainSymmetricSymbolicFactorization(): Factorizing Done" << std::endl;
                MPISerialDo(mpi, [&]()
                            { log() << fmt::format("  ({}, {}/{})", mpi.rank,
                                                   nnzLower + nnzUpper + this->Num(),
                                                   nnzOrig)
                                    << (((mpi.rank + 1) % 10) ? "" : "\n")
                                    << std::flush; });
                MPI::Barrier(mpi.comm);
                if (mpi.rank == 0)
                    log() << std::endl;
            }
            DNDS_assert_info(nnzLower == nnzUpper, fmt::format("nnzLower,Upper: [{},{}]", nnzLower, nnzUpper));

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
                    if (jP == iP)
                    {
                        cell2cellFaceVLocal2FullRowPos[i][ic2c] = 0;
                    }
                    else if (jP < iP)
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
            hasSymbolicPattern = true;
            ++generation;
        }
    };

    template <class Derived, class tComponent, class tVec>
    struct LocalLUBase
    {
        ssp<Direct::SerialSymLUStructure> symLU;
        bool isDecomposed = false;
        index nextNumericPivot = 0;

        LocalLUBase(ssp<Direct::SerialSymLUStructure> _symLU) : symLU(_symLU)
        {
        }

        virtual ~LocalLUBase() = default;

        void GetDiag(index i);          // pure "virtual" do not implement
        void GetLower(index i, int ij); // pure "virtual" do not implement
        void GetUpper(index i, int ij); // pure "virtual" do not implement
        void InvertDiag(const tComponent &v);

        void InPlaceDecompose()
        {
            InPlaceDecomposeV1();
        }

        void InPlaceDecomposeV1()
        {
            const auto &localPartStarts = symLU->localPartStarts;
            int nParts = localPartStarts.size() - 1;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (int iPart = 0; iPart < nParts; iPart++)
                for (index iP = localPartStarts.at(iPart); iP < localPartStarts.at(iPart + 1); iP++)
                    DecomposePivotV1(iP);
            isDecomposed = true;
        }

        /// Restart numeric progress after the caller has refilled matrix values.
        /// Existing factors must never be passed here as unfactored values.
        void BeginNumericFactorization()
        {
            isDecomposed = false;
            nextNumericPivot = 0;
        }

        /// Consecutive pivot ranges reuse the same arithmetic as full V1.
        /// Later rows may be updated by a pivot; range does not describe a
        /// disjoint writable row lease. Caller serializes numeric fill/factor.
        void DecomposeRange(index begin, index end)
        {
            DNDS_check_throw_info(!isDecomposed && begin == nextNumericPivot && begin >= 0 && end >= begin && end <= symLU->Num(),
                                  "out-of-order or invalid numeric factor range");
            for (index iP = begin; iP < end; ++iP)
                DecomposePivotV1(iP);
            nextNumericPivot = end;
            isDecomposed = end == symLU->Num();
        }

        void DecomposePivotV1(index iP)
        {
            auto dThis = static_cast<Derived *>(this);
            index i = symLU->FillingReorderNew2Old(iP);

            auto &&lowerRow = symLU->lowerTriStructureNew[iP];
            /*********************/
            // lower part
            for (int ijP = 0; ijP < lowerRow.size(); ijP++) //! for(int jP = 0; jP < iP; jP++)
            {
                auto jP = lowerRow[ijP];
                DNDS_assert(jP < iP);
                auto j = symLU->FillingReorderNew2Old(jP);
                // handle last j's job
                if (ijP > 0) //! if(jP > 0)
                {
                    auto &&lowerRowJP = symLU->lowerTriStructureNew[jP];
                    iterateIdentical( //! for(int kP = 0; kP < jP; kP++)
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
                            //! LU(iP, jP) -= LU(iP, kP) * LU(kP, jP);
                            // if (iP < 3)
                            // log() << fmt::format("Lower Add at {},{},{} === {}", iP, jP - 1, kP, (dThis->GetLower(i, pos1) * dThis->GetUpper(k, jPInUpperPos))(0)) << std::endl;
                            return false;
                        });
                }

                // auto luDiag = dThis->GetDiag(j).fullPivLu();
                // tComponent Aij = luDiag.solve(dThis->GetLower(i, ijP));
                dThis->GetLower(i, ijP) *= dThis->GetDiag(j); //! LU(iP, jP) *= LU(jP, jP);
            }
            /*********************/
            // diag part
            for (int ikP = 0; ikP < lowerRow.size(); ikP++) //! for(int kP = 0; kP < iP; kP++)
            {
                auto kP = lowerRow[ikP];
                auto k = symLU->FillingReorderNew2Old(kP);
                int iPInUpperPos = symLU->lowerTriStructureNewInUpper[iP][ikP];
                if (iPInUpperPos != -1)
                    dThis->GetDiag(i) -= dThis->GetLower(i, ikP) * dThis->GetUpper(k, iPInUpperPos);
                //! LU(iP, iP) -= LU(iP, kP) * LU(kP, iP);
            }

            dThis->GetDiag(i) = dThis->InvertDiag(dThis->GetDiag(i)); // * note here only stores
            //! LU(iP, iP) = inv(LU(iP, iP));
            /*********************/
            // upper part
            auto &&upperRow = symLU->upperTriStructureNew[iP];
            for (int ijP = 0; ijP < upperRow.size(); ijP++) //! for(int jP = iP+1; jP < N; jP++)
            {
                auto jP = upperRow[ijP];
                DNDS_assert(jP > iP);
                auto j = symLU->FillingReorderNew2Old(jP);
                auto &&lowerRowJP = symLU->lowerTriStructureNew[jP];

                iterateIdentical( //! for(int kP = 0; kP < iP; kP++)
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
                        //! LU(iP, jP) -= LU(iP, kP) * LU(kP, jP);
                        // if (iP < 3)
                        // log() << fmt::format("Upper Add at {},{},{} === {}", iP, jP, kP,
                        //                      (this->GetLower(i, pos1) * this->GetUpper(k, jPInUpperPos))(0))
                        // << std::endl;
                        return false;
                    });
            }
        }

        void InPlaceDecomposeV2()
        {
            auto dThis = static_cast<Derived *>(this);
            for (index iP = 0; iP < symLU->Num(); iP++)
            {
                index i = symLU->FillingReorderNew2Old(iP);

                auto &&lowerRow = symLU->lowerTriStructureNew[iP];

                /*********************/
                // diag part
                for (int ikP = 0; ikP < lowerRow.size(); ikP++) //! for(int kP = 0; kP < iP; kP++)
                {
                    auto kP = lowerRow[ikP];
                    auto k = symLU->FillingReorderNew2Old(kP);
                    int iPInUpperPos = symLU->lowerTriStructureNewInUpper[iP][ikP];
                    if (iPInUpperPos != -1)
                        dThis->GetDiag(i) -= dThis->GetLower(i, ikP) * dThis->GetUpper(k, iPInUpperPos);
                    //! LU(iP, iP) -= LU(iP, kP) * LU(kP, iP);
                }

                dThis->GetDiag(i) = dThis->InvertDiag(dThis->GetDiag(i)); // * note here only stores
                //! LU(iP, iP) = inv(LU(iP, iP));
                /*********************/
                // upper and lower part
                auto &&upperRow = symLU->upperTriStructureNew[iP];

                // #if defined(DNDS_DIST_MT_USE_OMP) // this OMP does not improve
                // #pragma omp parallel for schedule(static)
                // #endif
                for (int ijP = 0; ijP < upperRow.size(); ijP++) //! for(int jP = iP+1; jP < N; jP++)
                {
                    auto jP = upperRow[ijP];
                    DNDS_assert(jP > iP);
                    auto j = symLU->FillingReorderNew2Old(jP);
                    auto &&lowerRowJP = symLU->lowerTriStructureNew[jP];
                    int iPInLowerPosOfJRow = symLU->upperTriStructureNewInLower[iP][ijP];

                    iterateIdentical( //! for(int kP = 0; kP < iP; kP++)
                        lowerRow.begin(), lowerRow.end(), lowerRowJP.begin(), lowerRowJP.end(),
                        [&](index kP, auto pos1, auto pos2)
                        {
                            if (kP >= iP)
                                return true;
                            DNDS_assert(kP < symLU->Num());
                            auto k = symLU->FillingReorderNew2Old(kP);
                            int jPInUpperPos = symLU->lowerTriStructureNewInUpper[jP][pos2];
                            int iPInUpperPos = symLU->lowerTriStructureNewInUpper[iP][pos1];
                            if (jPInUpperPos != -1)
                                dThis->GetUpper(i, ijP) -= dThis->GetLower(i, pos1) * dThis->GetUpper(k, jPInUpperPos);
                            //! LU(iP, jP) -= LU(iP, kP) * LU(kP, jP);
                            if (iPInUpperPos != -1)
                                dThis->GetLower(j, iPInLowerPosOfJRow) -= dThis->GetLower(j, pos2) * dThis->GetUpper(k, iPInUpperPos);
                            // !LU(jP, iP) -= LU(jP, kP) * LU(kP, iP);
                            return false;
                        });
                    dThis->GetLower(j, iPInLowerPosOfJRow) *= dThis->GetDiag(i);
                    // !LU(jP, iP) *= inv(LU(iP, iP));
                }
            }
            isDecomposed = true;
        }

        void MatMul(tVec &x, tVec &result)
        {
            auto dThis = static_cast<Derived *>(this);
            DNDS_assert(!isDecomposed);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index iCell = 0; iCell < symLU->Num(); iCell++)
            {
                result[iCell] = dThis->GetDiag(iCell) * x[iCell];
                for (int ij = 0; ij < symLU->lowerTriStructure[iCell].size(); ij++)
                    result[iCell] += dThis->GetLower(iCell, ij) * x[symLU->lowerTriStructure[iCell][ij]];
                for (int ij = 0; ij < symLU->upperTriStructure[iCell].size(); ij++)
                    result[iCell] += dThis->GetUpper(iCell, ij) * x[symLU->upperTriStructure[iCell][ij]];
            }
        }

        /// Partial forward solve. Result predecessors must already be produced
        /// in this factor order; the wrapper owns phase/alias/readiness checks.
        template <class TRhs, class TResult>
        void SolveLowerRange(const TRhs &b, TResult &result, index begin, index end)
        {
            DNDS_check_throw_info(isDecomposed && begin >= 0 && end >= begin && end <= symLU->Num(), "invalid lower solve range or missing factors");
            auto dThis = static_cast<Derived *>(this);
            for (index iP = begin; iP < end; ++iP)
            {
                const index i = symLU->FillingReorderNew2Old(iP);
                result[i] = b[i];
                const auto &row = symLU->lowerTriStructure[i];
                for (index j = 0; j < index(row.size()); ++j)
                    result[i] -= dThis->GetLower(i, j) * result[row[j]];
            }
        }

        /// Partial backward solve using separate forward RHS and output.
        /// Visit contiguous ranges from high to low factor positions.
        template <class TRhs, class TResult>
        void SolveUpperRange(const TRhs &forward, TResult &result, index begin, index end)
        {
            DNDS_check_throw_info(isDecomposed && begin >= 0 && end >= begin && end <= symLU->Num(), "invalid upper solve range or missing factors");
            auto dThis = static_cast<Derived *>(this);
            for (index pos = end; pos > begin; --pos)
            {
                const index i = symLU->FillingReorderNew2Old(pos - 1);
                result[i] = forward[i];
                const auto &row = symLU->upperTriStructure[i];
                for (index j = 0; j < index(row.size()); ++j)
                    result[i] -= dThis->GetUpper(i, j) * result[row[j]];
                result[i] = dThis->GetDiag(i) * result[i];
            }
        }

        void Solve(tVec &b, tVec &result)
        {
            auto dThis = static_cast<Derived *>(this);
            DNDS_assert(isDecomposed);
            const auto &localPartStarts = symLU->localPartStarts;
            int nParts = localPartStarts.size() - 1;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (int iPart = 0; iPart < nParts; iPart++)
                for (index iP = localPartStarts.at(iPart); iP < localPartStarts.at(iPart + 1); iP++)
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
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (int iPart = 0; iPart < nParts; iPart++)
                for (index iP = localPartStarts.at(iPart + 1) - 1; iP >= localPartStarts.at(iPart); iP--)
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

        virtual ~LocalLDLTBase() = default;

        void GetDiag(index i);
        void GetLower(index i, int ij);
        void InvertDiag(const tComponent &v);

        void InPlaceDecompose()
        {
            // todo: add pseudo code
            // todo: make multithread
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
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index iCell = 0; iCell < symLU->Num(); iCell++)
            {
                result[iCell] = dThis->GetDiag(iCell) * x[iCell];
                for (int ij = 0; ij < symLU->lowerTriStructure[iCell].size(); ij++)
                    result[iCell] += dThis->GetLower(iCell, ij) * x[symLU->lowerTriStructure[iCell][ij]];
            }
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index jCell = 0; jCell < symLU->Num(); jCell++)
            {
                index jP = symLU->FillingReorderOld2New(jCell);
                auto &&upperRow = symLU->upperTriStructureNew[jP];
                for (int ji = 0; ji < (int)upperRow.size(); ji++)
                {
                    index iCellP = upperRow[ji];
                    index iCell = symLU->FillingReorderNew2Old(iCellP);
                    index posInLower = symLU->upperTriStructureNewInLower[jP][ji];
                    if (posInLower != -1)
                        result[jCell] += dThis->GetLower(iCell, posInLower).transpose() * x[iCell];
                }
            }
        }

        void Solve(tVec &b, tVec &result)
        {
            auto dThis = static_cast<Derived *>(this);
            DNDS_assert(isDecomposed);
            const auto &localPartStarts = symLU->localPartStarts;
            int nParts = localPartStarts.size() - 1;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (int iPart = 0; iPart < nParts; iPart++)
                for (index iP = localPartStarts.at(iPart); iP < localPartStarts.at(iPart + 1); iP++)
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
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index i = 0; i < symLU->Num(); i++)
            {
                result[i] = dThis->GetDiag(i) * result[i];
            }
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (int iPart = 0; iPart < nParts; iPart++)
                for (index iP = localPartStarts.at(iPart + 1) - 1; iP >= localPartStarts.at(iPart); iP--)
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
