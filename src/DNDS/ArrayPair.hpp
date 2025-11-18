#pragma once

#include "ArrayTransformer.hpp"
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
        DNDS_DEVICE_CALLABLE auto operator()(index i, TOthers... aOthers)
        {
            auto dThis = static_cast<Derived *>(this);
            if (i >= 0 && i < dThis->father.Size())
                return dThis->father.operator()(i, aOthers...);
            else
                return dThis->son.operator()(i - dThis->father.Size(), aOthers...);
        }

        template <class... TOthers>
        DNDS_DEVICE_CALLABLE auto operator()(index i, TOthers... aOthers) const
        {
            auto dThis = static_cast<const Derived *>(this);
            if (i >= 0 && i < dThis->father.Size())
                return dThis->father.operator()(i, aOthers...);
            else
                return dThis->son.operator()(i - dThis->father.Size(), aOthers...);
        }
    };

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
            DNDS_assert(R.father && R.son);
            //! rely on TArray's copy ctor!
            DNDS_MAKE_SSP(father, *(R.father)); // call TArray copy ctor
            DNDS_MAKE_SSP(son, *(R.son));       // call TArray copy ctor
            DNDS_assert(father->getMPI().comm == son->getMPI().comm);
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
            DNDS_assert_info(bool(father) && bool(son),
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

        void WriteSerialize(Serializer::SerializerBaseSSP serializerP, const std::string &name, bool includePIG = true, bool includeSon = true)
        {
            if (includePIG)
                DNDS_assert_info(trans.pLGlobalMapping && trans.pLGhostMapping, "pair's trans not having ghost info");

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

        /**
         * @warning if includePIG == true, need to createMPITypes after this
         */
        void ReadSerialize(Serializer::SerializerBaseSSP serializerP, const std::string &name, bool includePIG = true, bool includeSon = true)
        {
            DNDS_assert(father && son);
            this->TransAttach();

            auto cwd = serializerP->GetCurrentPath();
            // serializerP->CreatePath(name); //!remember no create!
            serializerP->GoToPath(name);

            index readRank{0}, readSize{0};
            if (serializerP->IsPerRank())
                serializerP->ReadIndex("MPIRank", readRank);
            serializerP->ReadIndex("MPISize", readSize);
            DNDS_assert((!serializerP->IsPerRank() || readRank == father->getMPI().rank) &&
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

        template <DeviceBackend B>
        using t_deviceView = ArrayPairDeviceView<B, TArray>;

        template <DeviceBackend B>
        using t_deviceViewConst = ArrayPairDeviceViewConst<B, TArray>;

        template <DeviceBackend B>
        auto deviceView()
        {
            DNDS_assert_info(father && son, "need both father and son to exist for device view");
            return t_deviceView<B>{
                father->template deviceView<B>(),
                son->template deviceView<B>()};
        }

        template <DeviceBackend B>
        auto deviceView() const
        {
            DNDS_assert_info(father && son, "need both father and son to exist for device view");
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
