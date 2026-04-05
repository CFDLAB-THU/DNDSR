#pragma once
/// @file ArrayPair.hpp
/// @brief Father-son array pairs with device views and ghost communication.

#include "ArrayTransformer.hpp"
#include "ArrayRedistributor.hpp"
#include "ArrayDerived/ArrayAdjacency.hpp"
#include "ArrayDerived/ArrayEigenVector.hpp"
#include "ArrayDerived/ArrayEigenMatrix.hpp"
#include "ArrayDerived/ArrayEigenMatrixBatch.hpp"
#include "ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "DNDS/Defines.hpp"
#include "DNDS/DeviceStorage.hpp"
#include "DNDS/Errors.hpp"
#include "DeviceView.hpp"
#include <fmt/format.h>
namespace DNDS
{

    /// @brief CRTP base for device views of father-son array pairs.
    template <class Derived>
    struct ArrayPairDeviceView_Base
    {
        DNDS_DEVICE_CALLABLE [[nodiscard]] index Size() const
        {
            auto dThis = static_cast<const Derived *>(this);
            return dThis->father.Size() + dThis->son.Size();
        }

        DNDS_DEVICE_CALLABLE auto RowSize() const
        {
            auto dThis = static_cast<const Derived *>(this);
            return dThis->father.RowSize();
        }

        DNDS_DEVICE_CALLABLE auto RowSize(index i) const
        {
            auto dThis = static_cast<const Derived *>(this);
            if (i >= 0 && i < dThis->father.Size())
                return dThis->father.RowSize(i);
            else
                return dThis->son.RowSize(i - dThis->father.Size());
        }

        DNDS_DEVICE_CALLABLE auto operator[](index i) const
        {
            auto dThis = static_cast<const Derived *>(this);
            if (i >= 0 && i < dThis->father.Size())
                return dThis->father.operator[](i);
            else
                return dThis->son.operator[](i - dThis->father.Size());
        }

        DNDS_DEVICE_CALLABLE auto operator[](index i)
        {
            auto dThis = static_cast<Derived *>(this);
            if (i >= 0 && i < dThis->father.Size())
                return dThis->father.operator[](i);
            else
                return dThis->son.operator[](i - dThis->father.Size());
        }

        template <class... TOthers>
        DNDS_DEVICE_CALLABLE decltype(auto) operator()(index i, TOthers... aOthers)
        {
            auto dThis = static_cast<Derived *>(this);
            if (i >= 0 && i < dThis->father.Size())
                return dThis->father.operator()(i, aOthers...);
            else
                return dThis->son.operator()(i - dThis->father.Size(), aOthers...);
        }

        template <class... TOthers>
        DNDS_DEVICE_CALLABLE decltype(auto) operator()(index i, TOthers... aOthers) const
        {
            auto dThis = static_cast<const Derived *>(this);
            if (i >= 0 && i < dThis->father.Size())
                return dThis->father.operator()(i, aOthers...);
            else
                return dThis->son.operator()(i - dThis->father.Size(), aOthers...);
        }
    };

    /// @brief Mutable device view of a father-son array pair.
    template <DeviceBackend B, class TArray = ParArray<real, 1>>
    struct ArrayPairDeviceView : public ArrayPairDeviceView_Base<ArrayPairDeviceView<B, TArray>>
    {
        using t_arrayDeviceView = typename TArray::template t_deviceView<B>;

        t_arrayDeviceView father;
        t_arrayDeviceView son;

        using t_self = ArrayPairDeviceView<B, TArray>;

        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(ArrayPairDeviceView, t_self)

        DNDS_DEVICE_CALLABLE ArrayPairDeviceView(const t_arrayDeviceView &n_father, const t_arrayDeviceView &n_son)
            : father(n_father), son(n_son) {}
    };

    /// @brief Const device view of a father-son array pair.
    template <DeviceBackend B, class TArray = ParArray<real, 1>>
    struct ArrayPairDeviceViewConst : public ArrayPairDeviceView_Base<ArrayPairDeviceViewConst<B, TArray>>
    {
        using t_arrayDeviceView = typename TArray::template t_deviceViewConst<B>; //! the only difference from non-const

        t_arrayDeviceView father;
        t_arrayDeviceView son;

        using t_self = ArrayPairDeviceViewConst<B, TArray>;

        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(ArrayPairDeviceViewConst, t_self)

        DNDS_DEVICE_CALLABLE ArrayPairDeviceViewConst(const t_arrayDeviceView &n_father, const t_arrayDeviceView &n_son)
            : father(n_father), son(n_son) {}
    };

    /// @brief Father + son array pair with attached ArrayTransformer for ghost communication.
    template <class TArray = ParArray<real, 1>>
    struct ArrayPair
    {
        using t_self = ArrayPair<TArray>;
        using t_arr = TArray;
        static constexpr bool IsCSR() { return t_arr::IsCSR(); }

        ssp<TArray> father;
        ssp<TArray> son;
        using TTrans = typename ArrayTransformerType<TArray>::Type;
        TTrans trans;

        void clone(const t_self &R)
        {
            DNDS_check_throw(R.father && R.son);
            //! rely on TArray's copy ctor!
            DNDS_MAKE_SSP(father, *(R.father)); // call TArray copy ctor
            DNDS_MAKE_SSP(son, *(R.son));       // call TArray copy ctor
            DNDS_check_throw(father->getMPI().comm == son->getMPI().comm);
            //! rely on TTrans's copy assignment!
            trans = R.trans;
            //! if R.trans already attached, then self trans attach self arrays
            if (R.trans.father)
                trans.father = father;
            if (R.trans.son)
                trans.son = son;
        }

        decltype(father->operator[](index(0))) operator[](index i) const
        {
            if (i >= 0 && i < father->Size())
                return father->operator[](i);
            else
                return son->operator[](i - father->Size());
        }

        decltype(father->operator[](index(0))) operator[](index i)
        {
            if (i >= 0 && i < father->Size())
                return father->operator[](i);
            else
                return son->operator[](i - father->Size());
        }

        // decltype(father->operator()(index(0), rowsize(0))) operator()(index i, rowsize j)
        // {
        //     if (i >= 0 && i < father->Size())
        //         return father->operator()(i, j);
        //     else
        //         return son->operator()(i - father->Size(), j);
        // }

        template <class... TOthers>
        decltype(auto) operator()(index i, TOthers... aOthers)
        {
            if (i >= 0 && i < father->Size())
                return father->operator()(i, aOthers...);
            else
                return son->operator()(i - father->Size(), aOthers...);
        }

        template <class... TOthers>
        decltype(auto) operator()(index i, TOthers... aOthers) const
        {
            if (i >= 0 && i < father->Size())
                return father->operator()(i, aOthers...);
            else
                return son->operator()(i - father->Size(), aOthers...);
        }

        template <class TF>
        auto runFunctionAppendedIndex(index i, TF &&F)
        {
            if (i >= 0 && i < father->Size())
                return F(*father, i);
            else
                return F(*son, i - father->Size());
        }

        auto RowSize() const
        {
            return father->RowSize();
        }

        auto RowSize(index i) const
        {
            if (i >= 0 && i < father->Size())
                return father->RowSize(i);
            else
                return son->RowSize(i - father->Size());
        }

        void ResizeRow(index i, rowsize rs)
        {
            if (i >= 0 && i < father->Size())
                father->ResizeRow(i, rs);
            else
                son->ResizeRow(i - father->Size(), rs);
        }

        template <class... TOthers>
        void ResizeRow(index i, TOthers... aOthers)
        {
            if (i >= 0 && i < father->Size())
                father->ResizeRow(i, aOthers...);
            else
                son->ResizeRow(i - father->Size(), aOthers...);
        }

        [[nodiscard]] index Size() const
        {
            DNDS_assert(father && son);
            return father->Size() + son->Size();
        }

        void TransAttach()
        {
            DNDS_check_throw_info(bool(father) && bool(son),
                                  fmt::format("father and son need to be constructed before Trans Attach. Array is {}", TArray::GetArrayName()));
            trans.setFatherSon(father, son);
        }

        void CompressBoth()
        {
            father->Compress();
            son->Compress();
        }

        void CopyFather(t_self &R)
        {
            father->CopyData(*R.father);
        }

        /**
         * \warning force waiting and re initializing persistent
         *
         */
        // TODO: make a data change listener in transformer?
        //! a situation: the data pointer should remain static as long as initPersistentPuxx is done
        void SwapDataFatherSon(t_self &R)
        {
            father->SwapData(*R.father);
            son->SwapData(*R.son);
            trans.reInitPersistentPullPush();
            R.trans.reInitPersistentPullPush();
        }

        std::size_t hash()
        {
            auto fatherHash = father->hash();
            auto sonHash = son->hash();
            index localHash = array_hash<std::size_t, 2>()({fatherHash, sonHash});
            MPIInfo mpi = father->getMPI();
            std::vector<index> hashes;
            hashes.resize(mpi.size);
            MPI::Allgather(&localHash, 1, DNDS_MPI_INDEX, hashes.data(), 1, DNDS_MPI_INDEX, mpi.comm);
            return vector_hash<index>()(hashes);
        }

        /// @brief Writes the ArrayPair (father, optional son, optional ghost mapping).
        ///
        /// Creates a sub-path `name` containing:
        /// - `MPIRank` (per-rank only), `MPISize` — partition metadata.
        /// - `father` — the father array via ParArray::WriteSerializer (Parts offset).
        /// - `son` — the son (ghost) array, if `includeSon` is true.
        /// - `pullingIndexGlobal` — ghost pull indices, if `includePIG` is true.
        ///
        /// All writes are collective for H5 serializers. Every rank must call this.
        void WriteSerialize(Serializer::SerializerBaseSSP serializerP, const std::string &name, bool includePIG = true, bool includeSon = true)
        {
            if (includePIG)
                DNDS_check_throw_info(trans.pLGlobalMapping && trans.pLGhostMapping, "pair's trans not having ghost info");

            auto cwd = serializerP->GetCurrentPath();
            serializerP->CreatePath(name);
            serializerP->GoToPath(name);

            if (serializerP->IsPerRank())
                serializerP->WriteIndex("MPIRank", father->getMPI().rank);
            serializerP->WriteIndex("MPISize", father->getMPI().size);
            // std::cout << trans.pLGlobalMapping->operator()(trans.mpi.rank, 0) << ",,," << trans.pLGlobalMapping->globalSize() << std::endl;
            // ! this is wrong as pLGlobalMapping stores the row index, not the data index!!
            // father->WriteSerializer(serializerP, "father",
            //                         Serializer::ArrayGlobalOffset{
            //                             trans.pLGlobalMapping->globalSize(),
            //                             trans.pLGlobalMapping->operator()(trans.mpi.rank, 0),
            //                         }); // trans.pLGlobalMapping == father->pLGlobalMapping
            // TODO: overwrite all the Resize()/ResizeRow() for ParArray so that it handles global size and offset internally?

            // now using the parts (calculate offsets)
            father->WriteSerializer(serializerP, "father", Serializer::ArrayGlobalOffset_Parts);
            if (includeSon)
                son->WriteSerializer(serializerP, "son", Serializer::ArrayGlobalOffset_Parts);
            /***************************/
            // ghost info
            // static_assert(std::is_same_v<rowsize, MPI_int>);
            // *writing pullingIndexGlobal, trusting the GlobalMapping to remain the same
            if (includePIG)
                serializerP->WriteIndexVector("pullingIndexGlobal", trans.pLGhostMapping->ghostIndex, Serializer::ArrayGlobalOffset_Parts);
            /***************************/

            serializerP->GoToPath(cwd);
        }

        /// @brief Writes the ArrayPair with an origIndex companion dataset for redistribution support.
        ///
        /// When origIndex is provided and the serializer is H5 (collective), an additional
        /// dataset "origIndex" is written alongside the father data. This enables reading
        /// the data back with a different MPI partition.
        ///
        /// @param serializerP  The serializer (H5 for redistribution support, JSON ignores origIndex)
        /// @param name         Path name for this array in the serializer hierarchy
        /// @param origIndex    Partition-independent key for each row (e.g., from cell2cellOrig).
        ///                     Must have size == father->Size().
        /// @param includePIG   Whether to include ghost pull-index-global data
        /// @param includeSon   Whether to include the son (ghost) array
        void WriteSerialize(Serializer::SerializerBaseSSP serializerP, const std::string &name,
                            const std::vector<index> &origIndex,
                            bool includePIG = true, bool includeSon = true)
        {
            DNDS_assert_info(index(origIndex.size()) == father->Size(),
                             fmt::format("origIndex size {} != father size {}", origIndex.size(), father->Size()));

            // Write the array data normally
            WriteSerialize(serializerP, name, includePIG, includeSon);

            // Write the origIndex companion dataset alongside the father data (H5 only)
            if (!serializerP->IsPerRank())
            {
                auto cwd = serializerP->GetCurrentPath();
                serializerP->GoToPath(name);
                serializerP->WriteIndexVector("origIndex", origIndex, Serializer::ArrayGlobalOffset_Parts);
                serializerP->WriteInt("redistributable", 1);
                serializerP->GoToPath(cwd);
            }
        }

        /// @brief Reads an ArrayPair written by WriteSerialize (same partition count).
        ///
        /// Reads father (and optionally son and ghost mapping) from sub-path `name`.
        /// Requires the file to have been written with the same MPI size.
        /// The father and son arrays are resized internally by Array::ReadSerializer.
        ///
        /// All reads are collective for H5 serializers. Every rank must call this,
        /// including ranks whose local size is 0.
        ///
        /// @warning If `includePIG` is true, the caller must call
        ///          `trans.createMPITypes()` after this method returns.
        void ReadSerialize(Serializer::SerializerBaseSSP serializerP, const std::string &name, bool includePIG = true, bool includeSon = true)
        {
            DNDS_check_throw(father && son);
            this->TransAttach();

            auto cwd = serializerP->GetCurrentPath();
            // serializerP->CreatePath(name); //!remember no create!
            serializerP->GoToPath(name);

            index readRank{0}, readSize{0};
            if (serializerP->IsPerRank())
                serializerP->ReadIndex("MPIRank", readRank);
            serializerP->ReadIndex("MPISize", readSize);
            DNDS_check_throw((!serializerP->IsPerRank() || readRank == father->getMPI().rank) &&
                             readSize == father->getMPI().size);
            auto offsetV_father = Serializer::ArrayGlobalOffset_Unknown;
            auto offsetV_son = Serializer::ArrayGlobalOffset_Unknown;
            father->ReadSerializer(serializerP, "father", offsetV_father);
            if (includeSon)
                son->ReadSerializer(serializerP, "son", offsetV_son);
            /***************************/
            // ghost info
            // static_assert(std::is_same_v<rowsize, MPI_int>);
            // *writing pullingIndexGlobal, trusting the GlobalMapping to remain the same
            if (includePIG)
            {
                std::vector<index> pullingIndexGlobal;
                auto offsetV_PIG = Serializer::ArrayGlobalOffset_Unknown; // TODO: check the offsets?
                serializerP->ReadIndexVector("pullingIndexGlobal", pullingIndexGlobal, offsetV_PIG);
                trans.createFatherGlobalMapping();
                trans.createGhostMapping(pullingIndexGlobal);
            }
            /***************************/

            serializerP->GoToPath(cwd);
        }

        /// @brief Reads ArrayPair data from HDF5 with redistribution support.
        ///
        /// Handles three cases depending on the file contents and partition count:
        ///
        /// 1. **No origIndex in file, same np**: falls back to ReadSerialize
        ///    (same-partition read, no redistribution).
        ///
        /// 2. **Has origIndex, same np**: reads father via ReadSerialize, reads
        ///    origIndex, then uses RedistributeArrayWithTransformer to move data
        ///    from the file's partition layout to the caller's partition layout.
        ///
        /// 3. **Has origIndex, different np**: reads father via EvenSplit (each
        ///    rank reads ~nGlobal/nRanks rows regardless of the original partition),
        ///    reads origIndex the same way, then uses
        ///    RedistributeArrayWithTransformer to pull each rank's needed cells.
        ///
        /// In case 3, some ranks may get 0 rows from EvenSplit when nGlobal < nRanks.
        /// This is handled correctly: the H5 collective reads proceed with 0-count
        /// hyperslab selections, and the redistribution operates on empty arrays.
        ///
        /// All operations are collective. Every rank must call this method.
        ///
        /// @param serializerP  The serializer (must be H5 / collective)
        /// @param name         Path name in the serializer hierarchy
        /// @param newOrigIndex Partition-independent keys for this rank's cells
        ///                     (e.g., from cell2cellOrig). Size must equal
        ///                     father->Size(). May be empty for ranks with 0 cells.
        void ReadSerializeRedistributed(
            Serializer::SerializerBaseSSP serializerP,
            const std::string &name,
            const std::vector<index> &newOrigIndex)
        {
            DNDS_check_throw(father && son);
            DNDS_check_throw_info(!serializerP->IsPerRank(),
                                  "Redistribution read only supported for collective (H5) serializers");

            auto cwd = serializerP->GetCurrentPath();
            serializerP->GoToPath(name);

            // Check if origIndex exists in the file
            auto pathContents = serializerP->ListCurrentPath();
            bool hasOrigIndex = pathContents.count("origIndex") > 0;

            index readSize{0};
            serializerP->ReadIndex("MPISize", readSize);
            bool sameNumPartition = (readSize == father->getMPI().size);

            if (!hasOrigIndex)
            {
                // No origIndex in file -- fall back to same-partition read
                serializerP->GoToPath(cwd);
                DNDS_check_throw_info(sameNumPartition,
                                      fmt::format("File has no origIndex and was written with np={}, "
                                                  "but reading with np={}. Cannot redistribute.",
                                                  readSize, father->getMPI().size));
                DNDS_check_throw_info(false, "Redistribution fallback requires same-partition read");
                ReadSerialize(serializerP, name, /*includePIG*/ false, /*includeSon*/ false);
                return;
            }

            // Use redistribution: even-split read + rendezvous.
            // This works for both same-np (different layout) and different-np cases.
            serializerP->GoToPath(cwd); // go back so we can navigate cleanly

            // Read the full ArrayPair normally into a temporary (same-np read)
            // or use even-split for different-np.
            auto mpi = father->getMPI();

            if (sameNumPartition)
            {
                // Same np: we can read normally (rank slices match), then redistribute locally
                t_self readPair;
                DNDS_MAKE_SSP(readPair.father, mpi);
                DNDS_MAKE_SSP(readPair.son, mpi);
                readPair.ReadSerialize(serializerP, name, /*includePIG*/ false, /*includeSon*/ false);

                // Read origIndex from the file
                serializerP->GoToPath(name);
                std::vector<index> fileOrigIndex;
                auto offsetOrigIdx = Serializer::ArrayGlobalOffset_Unknown;
                serializerP->ReadIndexVector("origIndex", fileOrigIndex, offsetOrigIdx);
                serializerP->GoToPath(cwd);

                DNDS_assert_info(readPair.father->Size() == index(fileOrigIndex.size()),
                                 fmt::format("readPair.father size {} != fileOrigIndex size {}",
                                             readPair.father->Size(), fileOrigIndex.size()));

                // Redistribute using ArrayTransformer
                DNDS_assert_info(father->Size() == index(newOrigIndex.size()),
                                 fmt::format("father size {} != newOrigIndex size {}",
                                             father->Size(), newOrigIndex.size()));
                RedistributeArrayWithTransformer<TArray>(
                    mpi, readPair.father, fileOrigIndex, newOrigIndex, father);
            }
            else
            {
                // Different np: use even-split read + rendezvous redistribution
                serializerP->GoToPath(name);

                // Read origIndex from file using even-split
                std::vector<index> fileOrigIndex;
                auto offsetOrigIdx = Serializer::ArrayGlobalOffset_EvenSplit;
                serializerP->ReadIndexVector("origIndex", fileOrigIndex, offsetOrigIdx);
                DNDS_assert(offsetOrigIdx.isDist()); // resolved by even-split read

                // Read father array data using even-split via TArray::ReadSerializer.
                // Pass EvenSplit offset so Array::ReadSerializer reads size correctly.
                auto readFather = std::make_shared<TArray>(mpi);
                {
                    auto offsetFather = Serializer::ArrayGlobalOffset_EvenSplit;
                    readFather->ReadSerializer(serializerP, "father", offsetFather);
                }

                DNDS_assert_info(readFather->Size() == index(fileOrigIndex.size()),
                                 fmt::format("readFather size {} != fileOrigIndex size {}",
                                             readFather->Size(), fileOrigIndex.size()));

                // Redistribute using ArrayTransformer
                DNDS_assert_info(father->Size() == index(newOrigIndex.size()),
                                 fmt::format("father size {} != newOrigIndex size {}",
                                             father->Size(), newOrigIndex.size()));
                RedistributeArrayWithTransformer<TArray>(
                    mpi, readFather, fileOrigIndex, newOrigIndex, father);

                serializerP->GoToPath(cwd);
            }
        }

        template <DeviceBackend B>
        using t_deviceView = ArrayPairDeviceView<B, TArray>;

        template <DeviceBackend B>
        using t_deviceViewConst = ArrayPairDeviceViewConst<B, TArray>;

        template <DeviceBackend B>
        auto deviceView()
        {
            DNDS_check_throw_info(father && son, "need both father and son to exist for device view");
            return t_deviceView<B>{
                father->template deviceView<B>(),
                son->template deviceView<B>()};
        }

        template <DeviceBackend B>
        auto deviceView() const
        {
            DNDS_check_throw_info(father && son, "need both father and son to exist for device view");
            return t_deviceViewConst<B>{
                std::as_const(*father).template deviceView<B>(),
                std::as_const(*son).template deviceView<B>()};
        }

        void to_device(DeviceBackend backend)
        {
            if (father)
                father->to_device(backend);
            if (son)
                son->to_device(backend);
        }

        void to_host()
        {
            if (father)
                father->to_host();
            if (son)
                son->to_host();
        }
    };

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    using ArrayAdjacencyPair = ArrayPair<ArrayAdjacency<_row_size, _row_max, _align>>;

    template <rowsize _vec_size = 1, rowsize _row_max = _vec_size, rowsize _align = NoAlign>
    using ArrayEigenVectorPair = ArrayPair<ArrayEigenVector<_vec_size, _row_max, _align>>;

    template <rowsize _mat_ni = 1, rowsize _mat_nj = 1,
              rowsize _mat_ni_max = _mat_ni, rowsize _mat_nj_max = _mat_nj, rowsize _align = NoAlign>
    using ArrayEigenMatrixPair = ArrayPair<ArrayEigenMatrix<_mat_ni, _mat_nj, _mat_ni_max, _mat_nj_max, _align>>;

    using ArrayEigenMatrixBatchPair = ArrayPair<ArrayEigenMatrixBatch>;

    template <int _n_row, int _n_col>
    using ArrayEigenUniMatrixBatchPair = ArrayPair<ArrayEigenUniMatrixBatch<_n_row, _n_col>>;
}
