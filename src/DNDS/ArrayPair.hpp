#include "ArrayTransformer.hpp"
#include "ArrayDerived/ArrayAdjacency.hpp"

namespace DNDS
{
    template <class TArray = ParArray<real, 1>>
    struct ArrayPair
    {
        using t_self = ArrayPair<TArray>;
        ssp<TArray> father;
        ssp<TArray> son;
        using TTrans = typename ArrayTransformerType<TArray>::Type;
        TTrans trans;

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

        auto RowSize()
        {
            return father->RowSize();
        }

        auto RowSize(index i)
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
            return father->Size() + son->Size();
        }

        void TransAttach()
        {
            DNDS_assert(bool(father) && bool(son));
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
            index localHash = std::hash<std::array<std::size_t, 2>>()({fatherHash, sonHash});
            MPIInfo mpi = father->getMPI();
            std::vector<index> hashes;
            hashes.resize(mpi.size);
            MPI::Allgather(&localHash, 1, DNDS_MPI_INDEX, hashes.data(), 1, DNDS_MPI_INDEX, mpi.comm);
            return std::hash<decltype(hashes)>()(hashes);
        }

        void WriteSerialize(Serializer::SerializerBaseSSP serializerP, const std::string &name, bool includePIG = true)
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
        void ReadSerialize(Serializer::SerializerBaseSSP serializerP, const std::string &name, bool includePIG = true)
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
    };

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    using ArrayAdjacencyPair = ArrayPair<ArrayAdjacency<_row_size, _row_max, _align>>;
}
