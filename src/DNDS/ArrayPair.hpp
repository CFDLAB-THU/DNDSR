#include "ArrayTransformer.hpp"
#include "ArrayDerived/ArrayAdjacency.hpp"

namespace DNDS
{
    template <class TArray = ParArray<real, 1>>
    struct ArrayPair
    {
        using t_self = ArrayPair<TArray>;
        std::shared_ptr<TArray> father;
        std::shared_ptr<TArray> son;
        using TTrans = typename ArrayTransformerType<TArray>::Type;
        TTrans trans;

        decltype(father->operator[](0)) operator[](index i) const
        {
            if (i >= 0 && i < father->Size())
                return father->operator[](i);
            else
                return son->operator[](i - father->Size());
        }

        decltype(father->operator()(0, 0)) operator()(index i, rowsize j)
        {
            if (i >= 0 && i < father->Size())
                return father->operator()(i, j);
            else
                return son->operator()(i - father->Size(), j);
        }

        template <class... TOthers>
        auto operator()(index i, TOthers... aOthers)
        {
            if (i >= 0 && i < father->Size())
                return father->operator()(i, aOthers...);
            else
                return son->operator()(i - father->Size(), aOthers...);
        }

        template <class... TOthers>
        auto operator()(index i, TOthers... aOthers) const
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

        index Size() const
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

        void WriteSerialize(SerializerBase *serializer, const std::string &name)
        {
            DNDS_assert_info(trans.pLGlobalMapping && trans.pLGhostMapping, "pair's trans not having ghost info");

            auto cwd = serializer->GetCurrentPath();
            serializer->CreatePath(name);
            serializer->GoToPath(name);

            serializer->WriteIndex("MPIRank", father->getMPI().rank);
            serializer->WriteIndex("MPISize", father->getMPI().size);
            father->WriteSerializer(serializer, "father");
            son->WriteSerializer(serializer, "son");
            /***************************/
            // ghost info
            // static_assert(std::is_same_v<rowsize, MPI_int>);
            // *writing pullingIndexGlobal, trusting the GlobalMapping to remain the same
            serializer->WriteIndexVector("pullingIndexGlobal", trans.pLGhostMapping->ghostIndex);
            /***************************/

            serializer->GoToPath(cwd);
        }

        /**
         * @warning need to createMPITypes after this
         */
        void ReadSerialize(SerializerBase *serializer, const std::string &name)
        {
            DNDS_assert(father && son);
            this->TransAttach();

            auto cwd = serializer->GetCurrentPath();
            // serializer->CreatePath(name); //!remember no create!
            serializer->GoToPath(name);

            index readRank, readSize;
            serializer->ReadIndex("MPIRank", readRank);
            serializer->ReadIndex("MPISize", readSize);
            DNDS_assert(readRank == father->getMPI().rank && readSize == father->getMPI().size);
            father->ReadSerializer(serializer, "father");
            son->ReadSerializer(serializer, "son");
            /***************************/
            // ghost info
            // static_assert(std::is_same_v<rowsize, MPI_int>);
            // *writing pullingIndexGlobal, trusting the GlobalMapping to remain the same
            std::vector<index> pullingIndexGlobal;
            serializer->ReadIndexVector("pullingIndexGlobal", pullingIndexGlobal);
            trans.createFatherGlobalMapping();
            trans.createGhostMapping(pullingIndexGlobal);
            /***************************/

            serializer->GoToPath(cwd);
        }
    };

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    using ArrayAdjacencyPair = ArrayPair<ArrayAdjacency<_row_size, _row_max, _align>>;
}
