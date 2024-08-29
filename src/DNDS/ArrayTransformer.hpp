#pragma once

#include "Array.hpp"
#include "IndexMapping.hpp"
#include "Profiling.hpp"

namespace DNDS
{
    using t_pLGlobalMapping = std::shared_ptr<GlobalOffsetsMapping>;
    using t_pLGhostMapping = std::shared_ptr<OffsetAscendIndexMapping>; // TODO: change to unique_ptr and modify corresponding copy constructor/assigner

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class ParArray : public Array<T, _row_size, _row_max, _align>
    {
    public:
        using TArray = Array<T, _row_size, _row_max, _align>;
        using t_pArray = std::shared_ptr<TArray>;
        static const DataLayout _dataLayout = TArray::_dataLayout;

        using TArray::Array;
        // TODO: privatize these
        t_pLGlobalMapping pLGlobalMapping;
        MPIInfo mpi;
        using t_pRowSizes = typename TArray::t_pRowSizes;

    public: //! use this with caution
        using TArray::ReadSerializer;
        using TArray::WriteSerializer;

    private:
        MPI_Datatype dataType = BasicType_To_MPIIntType<T>().first;
        MPI_int typeMult = BasicType_To_MPIIntType<T>().second;

    public:
        MPI_Datatype getDataType() { return dataType; }
        MPI_int getTypeMult() { return typeMult; }

    public:
        /**
         * @brief allows using default constructor and set MPI after
         *
         * @param n_mpi
         */
        void setMPI(const MPIInfo &n_mpi)
        {
            mpi = n_mpi;
            AssertDataType();
        }

        MPIInfo getMPI()
        {
            return mpi;
        }

        void setDataType(MPI_Datatype n_dType, MPI_int n_TypeMult)
        {
            dataType = n_dType;
            typeMult = n_TypeMult;
        }

        ParArray()
        {
        }

        ParArray(const MPIInfo &n_mpi) : mpi(n_mpi)
        {
            AssertDataType();
        }
        ParArray(MPI_Datatype n_dType, MPI_int n_TypeMult, const MPIInfo &n_mpi)
            : mpi(n_mpi), dataType(n_dType), typeMult(n_TypeMult)
        {
            AssertDataType();
        }

        void AssertDataType()
        {
            DNDS_assert(dataType != MPI_DATATYPE_NULL);
            MPI_Aint lb;
            MPI_Aint extent;
            MPI_Type_get_extent(dataType, &lb, &extent);
            DNDS_assert(lb == 0 && extent * typeMult == sizeof(T));
        }

        /**
         * @brief asserts on the consistencies
         *
         *
         * @warning //! warning, complexity O(Nproc); must collectively call
         * @return true currently only true
         * @return false
         */
        bool AssertConsistent()
        {
            DNDS_assert(mpi.comm != MPI_COMM_NULL);
            MPI::Barrier(mpi.comm); // must be globally existent
            if constexpr (_dataLayout == TABLE_Max ||
                          _dataLayout == TABLE_Fixed) // must have the same dynamic size
            {
                // checking if is uniform across all procs
                t_RowsizeVec uniformSizes(mpi.size);
                MPI_int rowsizeC = this->RowSizeField();
                static_assert(sizeof(MPI_int) == sizeof(rowsize));
                MPI::Allgather(&rowsizeC, 1, MPI_INT, uniformSizes.data(), 1, MPI_INT, mpi.comm);
                for (auto i : uniformSizes)
                    DNDS_assert_info(i == rowsizeC, "sizes not uniform across procs");
            }

            std::vector<MPI_int> uniform_typeMult(mpi.size);
            MPI::Allgather(&typeMult, 1, MPI_INT, uniform_typeMult.data(), 1, MPI_INT, mpi.comm);
            for (auto i : uniform_typeMult)
                DNDS_assert_info(i == typeMult, "typeMults not uniform across procs");

            return true; // currently all errors aborts inside
        }

        /// @warning must collectively call
        void createGlobalMapping() // collective;
        {
            DNDS_assert_info(mpi.comm != MPI_COMM_NULL, "MPI unset");
            // phase1.1: create localGlobal mapping (broadcast)
            pLGlobalMapping = std::make_shared<GlobalOffsetsMapping>();
            pLGlobalMapping->setMPIAlignBcast(mpi, this->Size());
        }

        /// @warning must collectively call
        index globalSize()
        {
            index gSize = 0;
            index cSize = this->Size();
            MPI::Allreduce(&cSize, &gSize, 1, DNDS_MPI_INDEX, MPI_SUM, mpi.comm);
            return gSize;
        }
    };
    /********************************************************************************************************/

    /********************************************************************************************************/

    template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    // template <class TArray>
    class ArrayTransformer
    {
        MPI::CommStrategy::ArrayCommType commTypeCurrent = MPI::CommStrategy::UnknownArrayCommType;

    public:
        using TArray = ParArray<T, _row_size, _row_max, _align>;
        using TSelf = ArrayTransformer<T, _row_size, _row_max, _align>;
        // using T = TArray::value_type;
        // static const rowsize _align = TArray::al;
        // static const rowsize _row_size = TArray::rs;
        // static const rowsize _row_max = TArray::rm;

        using t_pArray = std::shared_ptr<TArray>;
        static const DataLayout _dataLayout = TArray::_dataLayout;

        /*********************************/
        /*          MEMBER               */
        /*********************************/

        MPIInfo mpi;
        t_pLGhostMapping pLGhostMapping;
        t_pArray father;
        t_pArray son;

        t_pLGlobalMapping pLGlobalMapping; // reference from father

        std::shared_ptr<tMPI_typePairVec> pPushTypeVec;
        std::shared_ptr<tMPI_typePairVec> pPullTypeVec;

        // ** comm aux info: comm running structures **
        ssp<MPIReqHolder> PushReqVec;
        ssp<MPIReqHolder> PullReqVec;
        MPI_int nRecvPushReq{-1};
        MPI_int nRecvPullReq{-1};
        tMPI_statVec PushStatVec;
        tMPI_statVec PullStatVec;
        MPI_Aint pushSendSize;
        MPI_Aint pullSendSize;

        tMPI_intVec pushingSizes;
        tMPI_AintVec pushingDisps;
        std::vector<index> pushingIndexLocal;

        std::vector<std::vector<T>> inSituBuffer;

        /*********************************/
        /*          MEMBER               */
        /*********************************/

        void setFatherSon(const t_pArray &n_father, const t_pArray &n_son)
        {
            father = n_father;
            son = n_son;
            mpi = father->getMPI();
            DNDS_assert_info(son->getMPI() == father->getMPI(), "MPI inconsistent between father & son");
            DNDS_assert_info(father->getDataType() == son->getDataType(), "MPI datatype inconsistent between father & son");
            DNDS_assert_info(father->getTypeMult() == son->getTypeMult(), "MPI datatype multiplication inconsistent between father & son");
            DNDS_assert_info(father->getDataType() != MPI_DATATYPE_NULL, "MPI datatype invalid");
            DNDS_assert_info(father->getTypeMult() > 0, "MPI datatype multiplication invalid");
            pLGhostMapping.reset();
            pLGlobalMapping.reset();
            pLGlobalMapping = father->pLGlobalMapping;
        }

        template <class TRArrayTrans>
        void BorrowGGIndexing(const TRArrayTrans &RArrayTrans)
        {
            // DNDS_assert(father && Rarray.father); // Rarray's father is not visible...
            // DNDS_assert(father->obtainTotalSize() == Rarray.father->obtainTotalSize());
            DNDS_assert(RArrayTrans.pLGhostMapping && RArrayTrans.pLGlobalMapping);
            DNDS_assert(RArrayTrans.father->Size() == father->Size());
            pLGhostMapping = RArrayTrans.pLGhostMapping;
            pLGlobalMapping = RArrayTrans.pLGlobalMapping;
            father->pLGlobalMapping = RArrayTrans.pLGlobalMapping;
        }

        void createFatherGlobalMapping()
        {
            father->createGlobalMapping();
            pLGlobalMapping = father->pLGlobalMapping;
        }

        /** @brief create ghost by pulling data
         * @details
         * pulling data indicates the data put in son (received in pulling operation)
         * pullingIndexGlobal is the global indices in son
         * pullingIndexGlobal should be mutually different, otherwise behavior undefined
         */
        template <class TPullSet>
        void createGhostMapping(TPullSet &&pullingIndexGlobal) // collective;
        {
            DNDS_assert(bool(father) && bool(son));
            DNDS_assert_info(bool(father->pLGlobalMapping), "Father needs to createGlobalMapping");
            pLGlobalMapping = father->pLGlobalMapping;
            // phase1.2: count how many to pull and allocate the localGhost mapping, fill the mapping
            // counting could overflow
            // tMPI_intVec ghostSizes(mpi.size, 0); // == pulling sizes
            pLGhostMapping = std::make_shared<OffsetAscendIndexMapping>(
                (*pLGlobalMapping)(mpi.rank, 0), father->Size(),
                std::forward<TPullSet>(pullingIndexGlobal),
                *pLGlobalMapping,
                mpi);
        }

        /** @brief create ghost by pushing data
         * @details
         * pulling data indicates the data distributed in father (received in pushing operation)
         * CSR(pushingIndexLocal,pushStarts) indicates the local indices in father, for each rank to push from or pull to
         * CSR(pushingIndexLocal,pushStarts) is required to be mutually different entries of father, otherwise behavior undefined
         */
        template <class TPushSet, class TPushStart>
        void createGhostMapping(TPushSet &&pushingIndexLocal, TPushStart &&pushStarts) // collective;
        {
            DNDS_assert(bool(father) && bool(son));
            DNDS_assert_info(bool(father->pLGlobalMapping), "Father needs to createGlobalMapping");
            pLGlobalMapping = father->pLGlobalMapping;
            // phase1.2: calculate over pushing
            // counting could overflow
            pLGhostMapping = std::make_shared<OffsetAscendIndexMapping>(
                (*pLGlobalMapping)(mpi.rank, 0), father->Size(),
                std::forward<TPushSet>(pushingIndexLocal),
                std::forward<TPushStart>(pushStarts),
                *pLGlobalMapping,
                mpi);
        }
        /******************************************************************************************************************************/
        /**
         * \brief get real element byte info into account, with ghost indexer and comm types built, need two mappings
         * \pre has pLGlobalMapping pLGhostMapping, and resize the son
         * \post my indexer pPullTypeVec pPushTypeVec established
         */
        void createMPITypes() // collective;
        {
            DNDS_assert(bool(father) && bool(son));
            DNDS_assert(pLGlobalMapping && pLGhostMapping);
            commTypeCurrent = MPI::CommStrategy::Instance().GetArrayStrategy();
            if (commTypeCurrent == MPI::CommStrategy::HIndexed)
                father->Compress(); //! assure CSR is in compressed form
            // TODO: support comm for uncompressed: add in-situ packaging working mode
            // TODO: support actual MAX arrays' size communicating: append comm types: ? needed?
            // TODO: add manual packaging mode

            /*********************************************/ // starts to deal with actual byte sizes

            //*phase2.1: build push sizes and push disps
            index nSend = pLGhostMapping->pushingIndexGlobal.size();
            pushingSizes.resize(nSend); // pushing sizes  xx in bytes xx now in num of remove_all_extents_t<T>

            pushingDisps.resize(nSend); // pushing disps in bytes

            DNDS_assert(nSend < DNDS_INDEX_MAX);
            auto fatherDataStart = father->operator[](0);
            if (commTypeCurrent == MPI::CommStrategy::InSituPack)
                pushingIndexLocal.resize(nSend);
            for (index i = 0; i < index(nSend); i++)
            {
                MPI_int rank = -1;
                index loc = -1;
                bool found = pLGhostMapping->search(pLGhostMapping->pushingIndexGlobal[i], rank, loc);
                DNDS_assert_info(found && rank == -1, "must be at local main");                  // must be at local main
                pushingDisps[i] = (father->operator[](loc) - father->operator[](0)) * sizeof(T); //* in bytes
                if constexpr (_dataLayout == CSR)
                    pushingSizes[i] = father->RowSizeField(loc) * father->getTypeMult();
                if constexpr (isTABLE_Max(_dataLayout)) //! init sizes
                    pushingSizes[i] = father->RowSize(loc) * father->getTypeMult();
                if constexpr (isTABLE_Fixed(_dataLayout))
                    pushingSizes[i] = father->RowSizeField() * father->getTypeMult();

                if (commTypeCurrent == MPI::CommStrategy::InSituPack)
                    pushingIndexLocal[i] = loc;
            }
            // PrintVec(pushingSizes, std::cout);
            // std::cout << std::endl;

            //*phase2.2: be informed of pulled sub-indexer
            // equals to: building pullingSizes and pullingDisps, bytes size and disps of ghost
            // - legacy: indexer.buildAsGhostAlltoall(father->indexer, pushingSizes, *pLGhostMapping, mpi); // cascade from father
            auto do_son_resizing = [&]()
            {
                auto &LGhostMapping = *pLGhostMapping;
                index ghostArraySiz = LGhostMapping.ghostStart[LGhostMapping.ghostStart.size() - 1];
                DNDS_assert(mpi.size == LGhostMapping.ghostStart.size() - 1);
                if constexpr (_dataLayout == TABLE_StaticFixed)
                {
                    son->Resize(ghostArraySiz);
                    return;
                }
                if constexpr (_dataLayout == TABLE_Fixed)
                {
                    son->Resize(ghostArraySiz, father->RowSize()); // using father's row size
                    return;
                }
                if constexpr (_dataLayout == TABLE_Max)
                {
                    son->Resize(ghostArraySiz, father->RowSizeMax());
                    // and go on for non-uniform resizing
                }
                if constexpr (_dataLayout == TABLE_StaticMax)
                {
                    son->Resize(ghostArraySiz);
                    // and go on for non-uniform resizing
                }

                // obtain pulling sizes with pushing sizes
                tMPI_intVec pullingSizes(ghostArraySiz);
                MPI_Alltoallv(pushingSizes.data(), LGhostMapping.pushIndexSizes.data(), LGhostMapping.pushIndexStarts.data(), MPI_INT,
                              pullingSizes.data(), LGhostMapping.ghostSizes.data(), LGhostMapping.ghostStart.data(), MPI_INT,
                              mpi.comm);

                // std::cout << LGhostMapping.gStarts().size() << std::endl;
                if constexpr (_dataLayout == CSR)
                    son->Resize(ghostArraySiz, [&](index i)
                                { return pullingSizes[i] / father->getTypeMult(); });
                if constexpr (_dataLayout == TABLE_Max)
                {
                    son->Resize(ghostArraySiz, father->RowSizeMax());
                    for (index i = 0; i < son->Size(); i++)
                        son->ResizeRow(i, pullingSizes[i] / father->getTypeMult());
                }
                if constexpr (_dataLayout == TABLE_StaticMax)
                {
                    son->Resize(ghostArraySiz);
                    for (index i = 0; i < son->Size(); i++)
                        son->ResizeRow(i, pullingSizes[i] / father->getTypeMult());
                }
                // is actually pulling disps, but is contiguous anyway

                // DNDS_MPI_InsertCheck(mpi);
                // std::cout << mpi.rank << " VEC ";
                // PrintVec(pullingSizes, std::cout);
                // std::cout << std::endl;
                // DNDS_MPI_InsertCheck(mpi);

                // note that Rowstart and pullingSizes are in bytes
                // pullingSizes is actual but Rowstart is before indexModder(), use indexModder[] to invert
            };
            do_son_resizing();

            // phase3: create and register MPI types of pushing and pulling
            if constexpr (isTABLE_Max(_dataLayout)) // convert back to full pushing sizes
            {
                for (auto &i : pushingSizes)
                    i = son->RowSizeField() * father->getTypeMult();
            }

            if (commTypeCurrent == MPI::CommStrategy::HIndexed) // record types
            {
                pPushTypeVec = std::make_shared<MPITypePairHolder>(0);
                pPullTypeVec = std::make_shared<MPITypePairHolder>(0);
                for (MPI_int r = 0; r < mpi.size; r++)
                {
                    /************************************************************/
                    // push
                    MPI_int pushNumber = pLGhostMapping->pushIndexSizes[r];
                    // std::cout << "PN" << pushNumber << std::endl;
                    MPI_Aint *pPushDisps;
                    MPI_int *pPushSizes;
                    pPushDisps = pushingDisps.data() + pLGhostMapping->pushIndexStarts[r];
                    pPushSizes = pushingSizes.data() + pLGhostMapping->pushIndexStarts[r];
                    index sumPushSizes = 0; // using upgraded integer
                    for (MPI_int i = 0; i < pushNumber; i++)
                        sumPushSizes += pPushSizes[i];
                    if (sumPushSizes > 0) // if no actuall data is to be sent
                    {
                        // std::cout <<mpi.rank<< " pushSlice " << pPushDisps[0] << outputDelim << pPushSizes[0] << std::endl;

                        // if (mpi.rank == 0)
                        // {
                        //     std::cout << "pushing to " << r << "  size" << pushNumber << "\n";
                        //     for (int i = 0; i < pushNumber; i++)
                        //         std::cout << "b[" << i << "] = " << pPushSizes[i] << std::endl;
                        //     for (int i = 0; i < pushNumber; i++)
                        //         std::cout << "d[" << i << "] = " << pPushDisps[i] << std::endl;
                        // }
                        // std::cout << "=== PUSH TYPE : " << mpi.rank << " from " << r << std::endl;

                        MPI_Datatype dtype;

                        MPI_Type_create_hindexed(pushNumber, pPushSizes, pPushDisps, father->getDataType(), &dtype);
                        // MPI_Type_create_hindexed(PushDispsMPI.size(), PushSizesMPI.data(), PushDispsMPI.data(), father->getDataType(), &dtype);

                        MPI_Type_commit(&dtype);
                        pPushTypeVec->push_back(std::make_pair(r, dtype));
                        // OPT: could use MPI_Type_create_hindexed_block to save some space
                    }
                    /************************************************************/
                    // pull
                    MPI_Aint pullDisp[1];

                    MPI_int pullSizes[1]; // same as pushSizes
                    auto gRPtr = son->operator[](index(pLGhostMapping->ghostStart[r + 1]));
                    auto gLPtr = son->operator[](index(pLGhostMapping->ghostStart[r]));
                    auto gStartPtr = son->operator[](index(0));
                    auto ghostSpan = gRPtr - gLPtr;
                    auto ghostStart = gLPtr - gStartPtr;
                    DNDS_assert(ghostSpan < INT_MAX && ghostStart < INT_MAX);
                    pullSizes[0] = MPI_int(ghostSpan) * father->getTypeMult();
                    pullDisp[0] = ghostStart * sizeof(T);
                    if (pullSizes[0] > 0)
                    {
                        // std::cout << "=== PULL TYPE : " << mpi.rank << " from " << r << std::endl;
                        MPI_Datatype dtype;

                        MPI_Type_create_hindexed(1, pullSizes, pullDisp, father->getDataType(), &dtype);

                        // std::cout << mpi.rank << " pullSlice " << pullDisp[0] << outputDelim << pullBytes[0] << std::endl;
                        MPI_Type_commit(&dtype);
                        pPullTypeVec->push_back(std::make_pair(r, dtype));
                    }
                }
                pPullTypeVec->shrink_to_fit();
                pPushTypeVec->shrink_to_fit();

                pushingDisps.clear();
                pushingSizes.clear(); // no need
            }
            else if (commTypeCurrent == MPI::CommStrategy::CommStrategy::InSituPack)
            {
                // could simplify some info on sparse comm?
                pushingDisps.clear();
            }
            else
            {
                DNDS_assert(false);
            }
        }
        /******************************************************************************************************************************/

        /******************************************************************************************************************************/
        /**
         * \brief when established push and pull types, init persistent-nonblocked-nonbuffered MPI reqs
         * \pre has pPullTypeVec pPushTypeVec
         * \post PushReqVec established
         * \warning after init, raw buffers of data for both father and son/ghost should remain static
         */
        void initPersistentPush() // collective;
        {
            if (commTypeCurrent == MPI::CommStrategy::HIndexed)
            {
                // DNDS_assert(pPullTypeVec && pPushTypeVec);
                DNDS_assert(pPullTypeVec.use_count() > 0 && pPushTypeVec.use_count() > 0);
                pushSendSize = 0;
                auto nReqs = pPullTypeVec->size() + pPushTypeVec->size();
                // DNDS_assert(nReqs > 0);
                DNDS_MAKE_SSP(PushReqVec);
                PushReqVec->resize(nReqs, (MPI_REQUEST_NULL)), PushStatVec.resize(nReqs);
                nRecvPushReq = 0;
                for (auto ip = 0; ip < pPushTypeVec->size(); ip++)
                {
                    auto dtypeInfo = (*pPushTypeVec)[ip];
                    MPI_int rankOther = dtypeInfo.first;
                    MPI_int tag = rankOther + mpi.rank;
                    MPI_Recv_init(father->data(), 1, dtypeInfo.second, rankOther, tag, mpi.comm, PushReqVec->data() + pPullTypeVec->size() + ip);
                    // cascade from father
                    nRecvPushReq++;
                }
                for (auto ip = 0; ip < pPullTypeVec->size(); ip++)
                {
                    auto dtypeInfo = (*pPullTypeVec)[ip];
                    MPI_int rankOther = dtypeInfo.first;
                    MPI_int tag = rankOther + mpi.rank;
                    
#ifndef ARRAY_COMM_USE_BUFFERED_SEND
                    // MPI_Ssend_init
                    MPI_Send_init
#else
                    MPI_Bsend_init
#endif
                        (son->data(), 1, dtypeInfo.second, rankOther, tag, mpi.comm, PushReqVec->data() + ip);

                    // cascade from father

                    // // buffer calculate //!deprecated because of size limit
                    // MPI_Aint csize;
                    // MPI_Pack_external_size(1, dtypeInfo.second, mpi.comm, &csize);
                    // csize += MPI_BSEND_OVERHEAD;
                    // DNDS_assert(MAX_MPI_Aint - pushSendSize >= csize && csize > 0);
                    // pushSendSize += csize * 2;
                }
#ifdef ARRAY_COMM_USE_BUFFERED_SEND
                // MPIBufferHandler::Instance().claim(pushSendSize, mpi.rank);
#endif
            }
            else if (commTypeCurrent == MPI::CommStrategy::CommStrategy::InSituPack)
            {
                // could simplify some info on sparse comm?
                DNDS_MAKE_SSP(PushReqVec);
            }
            else
            {
                DNDS_assert(false);
            }
        }
        /******************************************************************************************************************************/

        /******************************************************************************************************************************/
        /**
         * \brief when established push and pull types, init persistent-nonblocked-nonbuffered MPI reqs
         * \pre has pPullTypeVec pPushTypeVec
         * \post PullReqVec established
         * \warning after init, raw buffers of data for both father and son/ghost should remain static
         */
        void initPersistentPull() // collective;
        {
            if (commTypeCurrent == MPI::CommStrategy::HIndexed)
            {
                // DNDS_assert(pPullTypeVec && pPushTypeVec);
                DNDS_assert(pPullTypeVec.use_count() > 0 && pPushTypeVec.use_count() > 0);
                auto nReqs = pPullTypeVec->size() + pPushTypeVec->size();
                pullSendSize = 0;
                // DNDS_assert(nReqs > 0);
                DNDS_MAKE_SSP(PullReqVec);
                PullReqVec->resize(nReqs, (MPI_REQUEST_NULL)), PullStatVec.resize(nReqs);
                nRecvPullReq = 0;
                for (typename decltype(pPullTypeVec)::element_type::size_type ip = 0; ip < pPullTypeVec->size(); ip++)
                {
                    auto dtypeInfo = (*pPullTypeVec)[ip];
                    MPI_int rankOther = dtypeInfo.first;
                    MPI_int tag = rankOther + mpi.rank; //! receives a lot of messages, this distinguishes them
                    // std::cout << mpi.rank << " Recv " << rankOther << std::endl;
                    MPI_Recv_init(son->data(), 1, dtypeInfo.second, rankOther, tag, mpi.comm, PullReqVec->data() + ip);
                    nRecvPullReq++;
                    // std::cout << *(real *)(dataGhost.data() + 8 * 0) << std::endl;
                    // cascade from father
                }
                for (typename decltype(pPullTypeVec)::element_type::size_type ip = 0; ip < pPushTypeVec->size(); ip++)
                {
                    auto dtypeInfo = (*pPushTypeVec)[ip];
                    MPI_int rankOther = dtypeInfo.first;
                    MPI_int tag = rankOther + mpi.rank;
                    // std::cout << mpi.rank << " Send " << rankOther << std::endl;
#ifndef ARRAY_COMM_USE_BUFFERED_SEND
                    // MPI_Ssend_init
                    MPI_Send_init
#else
                    MPI_Bsend_init
#endif
                        (father->data(), 1, dtypeInfo.second, rankOther, tag, mpi.comm, PullReqVec->data() + pPullTypeVec->size() + ip);
                    // std::cout << *(real *)(data.data() + 8 * 1) << std::endl;
                    // cascade from father

                    // // buffer calculate //!deprecated because of size limit
                    // MPI_Aint csize;
                    // MPI_Pack_external_size(1, dtypeInfo.second, mpi.comm, &csize);
                    // csize += MPI_BSEND_OVERHEAD * 8;
                    // DNDS_assert(MAX_MPI_Aint - pullSendSize >= csize && csize > 0);
                    // pullSendSize += csize * 2;
                }
#ifdef ARRAY_COMM_USE_BUFFERED_SEND
                // MPIBufferHandler::Instance().claim(pullSendSize, mpi.rank);
#endif
            }
            else if (commTypeCurrent == MPI::CommStrategy::CommStrategy::InSituPack)
            {
                // could simplify some info on sparse comm?
                DNDS_MAKE_SSP(PullReqVec);
            }
            else
            {
                DNDS_assert(false);
            }
        }
        /******************************************************************************************************************************/

        void __InSituPackStartPush()
        {
            nRecvPushReq = 0;
            for (MPI_int r = 0; r < mpi.size; r++)
            {
                // push
                MPI_int pushNumber = pLGhostMapping->pushIndexSizes[r];
                // std::cout << "PN" << pushNumber << std::endl;
                if (pushNumber > 0)
                {
                    index nPushData{0};
                    for (index i = 0; i < pushNumber; i++)
                    {
                        auto loc = pushingIndexLocal.at(pLGhostMapping->pushIndexStarts[r] + i);
                        index nPush = 0;
                        if constexpr (_dataLayout == CSR)
                            nPush = father->RowSizeField(loc);
                        if constexpr (isTABLE_Max(_dataLayout)) //! init sizes
                            nPush = father->RowSize(loc);
                        if constexpr (isTABLE_Fixed(_dataLayout))
                            nPush = father->RowSizeField();
                        nPushData += nPush;
                    }
                    inSituBuffer.emplace_back(nPushData);
                    PushReqVec->emplace_back(MPI_REQUEST_NULL);
                    MPI_Irecv(inSituBuffer.back().data(), nPushData * father->getTypeMult(), father->getDataType(),
                              r, mpi.rank + r, mpi.comm, &PushReqVec->back());
                    nRecvPushReq++;
                }
            }
            for (MPI_int r = 0; r < mpi.size; r++)
            {
                // pull
                MPI_Aint pullDisp;
                MPI_int pullSize; // same as pushSizes
                auto gRPtr = son->operator[](index(pLGhostMapping->ghostStart[r + 1]));
                auto gLPtr = son->operator[](index(pLGhostMapping->ghostStart[r]));
                auto ghostSpan = gRPtr - gLPtr;
                pullSize = MPI_int(ghostSpan);

                if (pullSize > 0)
                {
                    PushReqVec->emplace_back(MPI_REQUEST_NULL);
                    MPI_Issend(gLPtr, pullSize * father->getTypeMult(), father->getDataType(), r, r + mpi.rank, mpi.comm, &PushReqVec->back());
                }
            }
        }

        void startPersistentPush() // collective;
        {
            if (commTypeCurrent == MPI::CommStrategy::HIndexed)
            {
                // req already ready
                DNDS_assert(nRecvPushReq <= PushReqVec->size());
                if (PushReqVec->size())
                {
                    if (MPI::CommStrategy::Instance().GetUseAsyncOneByOne())
                    {
                    }
                    else
                        MPI_Startall(PushReqVec->size(), PushReqVec->data());
                }
            }
            else if (commTypeCurrent == MPI::CommStrategy::InSituPack)
            {
                __InSituPackStartPush();
            }
            else
            {
                DNDS_assert(false);
            }
            PerformanceTimer::Instance().StartTimer(PerformanceTimer::TimerType::Comm);
#ifdef ARRAY_COMM_USE_BUFFERED_SEND
            MPIBufferHandler::Instance().claim(pushSendSize, mpi.rank);
#endif

            PerformanceTimer::Instance().StopTimer(PerformanceTimer::TimerType::Comm);
        }

        void __InSituPackStartPull()
        {
            nRecvPullReq = 0;
            for (MPI_int r = 0; r < mpi.size; r++)
            {
                // pull
                MPI_Aint pullDisp;
                MPI_int pullSize; // same as pushSizes
                auto gRPtr = son->operator[](index(pLGhostMapping->ghostStart[r + 1]));
                auto gLPtr = son->operator[](index(pLGhostMapping->ghostStart[r]));
                auto ghostSpan = gRPtr - gLPtr;
                pullSize = MPI_int(ghostSpan);

                if (pullSize > 0)
                {
                    PullReqVec->emplace_back(MPI_REQUEST_NULL);
                    MPI_Irecv(gLPtr, pullSize * father->getTypeMult(), father->getDataType(), r, r + mpi.rank, mpi.comm, &PullReqVec->back());
                    nRecvPullReq++;
                }
            }
            for (MPI_int r = 0; r < mpi.size; r++)
            {
                // push
                MPI_int pushNumber = pLGhostMapping->pushIndexSizes[r];
                // std::cout << "PN" << pushNumber << std::endl;
                if (pushNumber > 0)
                {
                    index nPushData{0};
                    for (index i = 0; i < pushNumber; i++)
                    {
                        auto loc = pushingIndexLocal.at(pLGhostMapping->pushIndexStarts[r] + i);
                        index nPush = 0;
                        if constexpr (_dataLayout == CSR)
                            nPush = father->RowSizeField(loc);
                        if constexpr (isTABLE_Max(_dataLayout)) //! init sizes
                            nPush = father->RowSize(loc);
                        if constexpr (isTABLE_Fixed(_dataLayout))
                            nPush = father->RowSizeField();
                        nPushData += nPush;
                    }
                    inSituBuffer.emplace_back(nPushData);
                    nPushData = 0;
                    for (index i = 0; i < pushNumber; i++)
                    {
                        auto loc = pushingIndexLocal.at(pLGhostMapping->pushIndexStarts[r] + i);
                        index nPush = 0;
                        if constexpr (_dataLayout == CSR)
                            nPush = father->RowSizeField(loc);
                        if constexpr (isTABLE_Max(_dataLayout)) //! init sizes
                            nPush = father->RowSize(loc);
                        if constexpr (isTABLE_Fixed(_dataLayout))
                            nPush = father->RowSizeField();
                        std::copy((*father)[loc], (*father)[loc] + nPush, inSituBuffer.back().begin() + nPushData);
                        nPushData += nPush;
                    }
                    PullReqVec->emplace_back(MPI_REQUEST_NULL);
                    MPI_Issend(inSituBuffer.back().data(), nPushData * father->getTypeMult(), father->getDataType(),
                               r, mpi.rank + r, mpi.comm, &PullReqVec->back());
                }
            }
        }

        void startPersistentPull() // collective;
        {
            PerformanceTimer::Instance().StartTimer(PerformanceTimer::TimerType::Comm);
            if (commTypeCurrent == MPI::CommStrategy::HIndexed)
            {
                DNDS_assert(nRecvPullReq <= PullReqVec->size());
                // req already ready
                if (PullReqVec->size())
                {
                    if (MPI::CommStrategy::Instance().GetUseAsyncOneByOne())
                    {
                    }
                    else
                        MPI_Startall(int(PullReqVec->size()), PullReqVec->data());
                }
            }
            else if (commTypeCurrent == MPI::CommStrategy::InSituPack)
            {
                __InSituPackStartPull();
            }
            else
            {
                DNDS_assert(false);
            }
#ifdef ARRAY_COMM_USE_BUFFERED_SEND
            MPIBufferHandler::Instance().claim(pullSendSize, mpi.rank);
#endif

            PerformanceTimer::Instance().StopTimer(PerformanceTimer::TimerType::Comm);
        }

        void waitPersistentPush() // collective;
        {
            if (MPI::CommStrategy::Instance().GetUseStrongSyncWait())
                MPI::Barrier(mpi.comm);
            PerformanceTimer::Instance().StartTimer(PerformanceTimer::TimerType::Comm);
            PushStatVec.resize(PushReqVec->size());
#ifdef ARRAY_COMM_USE_BUFFERED_SEND
            MPIBufferHandler::Instance().unclaim(pushSendSize);
#endif
            if (commTypeCurrent == MPI::CommStrategy::HIndexed)
            {
                // data alright
                if (PushReqVec->size())
                {
                    DNDS_assert(nRecvPushReq <= PushReqVec->size());
                    if (MPI::CommStrategy::Instance().GetUseAsyncOneByOne())
                    {
                        MPI_Startall(nRecvPushReq, PushReqVec->data());
                        for (int iReq = nRecvPushReq; iReq < PushReqVec->size(); iReq++)
                        {
                            MPI_Start(&PushReqVec->operator[](iReq));
                            MPI_Wait(&PushReqVec->operator[](iReq), MPI_STATUS_IGNORE);
                        }
                        MPI::WaitallAuto(nRecvPushReq, PushReqVec->data(), MPI_STATUSES_IGNORE);
                    }
                    else
                        MPI::WaitallAuto(PushReqVec->size(), PushReqVec->data(), MPI_STATUSES_IGNORE);
                }
            }
            else if (commTypeCurrent == MPI::CommStrategy::InSituPack)
            {
                if (PushReqVec->size())
                    MPI::WaitallAuto(PushReqVec->size(), PushReqVec->data(), PushStatVec.data());
                auto bufferVec = inSituBuffer.begin();
                for (MPI_int r = 0; r < mpi.size; r++)
                {
                    // push
                    DNDS_assert(bufferVec < inSituBuffer.end());
                    MPI_int pushNumber = pLGhostMapping->pushIndexSizes[r];
                    // std::cout << "PN" << pushNumber << std::endl;
                    if (pushNumber > 0)
                    {
                        index nPushData = 0;
                        for (index i = 0; i < pushNumber; i++)
                        {
                            auto loc = pushingIndexLocal.at(pLGhostMapping->pushIndexStarts[r] + i);
                            index nPush = 0;
                            if constexpr (_dataLayout == CSR)
                                nPush = father->RowSizeField(loc);
                            if constexpr (isTABLE_Max(_dataLayout)) //! init sizes
                                nPush = father->RowSize(loc);
                            if constexpr (isTABLE_Fixed(_dataLayout))
                                nPush = father->RowSizeField();
                            std::copy(bufferVec->begin() + nPushData, bufferVec->begin() + nPushData + nPush, (*father)[loc]);
                            nPushData += nPush;
                        }
                        bufferVec++;
                    }
                }
                inSituBuffer.clear();
                PushReqVec->clear();
            }
            else
            {
                DNDS_assert(false);
            }
            PerformanceTimer::Instance().StopTimer(PerformanceTimer::TimerType::Comm);
            if (MPI::CommStrategy::Instance().GetUseStrongSyncWait())
                MPI::Barrier(mpi.comm);
        }
        void waitPersistentPull() // collective;
        {
            PerformanceTimer::Instance().StartTimer(PerformanceTimer::TimerType::Comm);
            PullStatVec.resize(PullReqVec->size());

#ifdef ARRAY_COMM_USE_BUFFERED_SEND
            MPIBufferHandler::Instance().unclaim(pullSendSize);
#endif
            if (commTypeCurrent == MPI::CommStrategy::HIndexed)
            {
                // data alright
                if (PullReqVec->size())
                {
                    DNDS_assert(nRecvPullReq <= PullReqVec->size());
                    if (MPI::CommStrategy::Instance().GetUseAsyncOneByOne())
                    {
                        MPI_Startall(nRecvPullReq, PullReqVec->data());
                        for (int iReq = nRecvPullReq; iReq < PullReqVec->size(); iReq++)
                        {
                            MPI_Start(&PullReqVec->operator[](iReq));
                            MPI_Wait(&PullReqVec->operator[](iReq), MPI_STATUS_IGNORE);
                            // if (mpi.rank == 0)
                            //     log() << "waited a req" << std::endl;
                        }
                        MPI::WaitallAuto(nRecvPullReq, PullReqVec->data(), MPI_STATUSES_IGNORE);
                    }
                    else
                    {
                        MPI::WaitallAuto(PullReqVec->size(), PullReqVec->data(), MPI_STATUSES_IGNORE);
                    }
                }
            }
            else if (commTypeCurrent == MPI::CommStrategy::InSituPack)
            {
                if (PullReqVec->size())
                    MPI::WaitallAuto(PullReqVec->size(), PullReqVec->data(), PullStatVec.data());
                // std::cout << "waiting DONE" << std::endl;
                inSituBuffer.clear();
                PullReqVec->clear();
            }
            else
            {
                DNDS_assert(false);
            }
            PerformanceTimer::Instance().StopTimer(PerformanceTimer::TimerType::Comm);
        }

        void clearPersistentPush() // collective;
        {
            waitPersistentPush();
            PushReqVec->clear(); // stat vec is left untouched here
        }
        void clearPersistentPull() // collective;
        {
            waitPersistentPull();
            PullReqVec->clear();
        }

        void clearMPITypes() // collective;
        {
            pPullTypeVec.reset();
            pPushTypeVec.reset();
        }

        void clearGlobalMapping() // collective;
        {
            pLGlobalMapping.reset();
        }

        void clearGhostMapping() // collective;
        {
            pLGhostMapping.reset();
        }

        void pullOnce() // collective;
        {
            initPersistentPull();
            startPersistentPull();
            waitPersistentPull();
            clearPersistentPull();
        }

        void pushOnce() // collective;
        {
            initPersistentPush();
            startPersistentPush();
            waitPersistentPush();
            clearPersistentPush();
        }

        void reInitPersistentPullPush()
        {
            bool clearedPull{false}, clearedPush{false};
            if (PullReqVec->size())
            {
                clearedPull = true;
                waitPersistentPull();
                clearPersistentPull();
            }
            if (PushReqVec->size())
            {
                clearedPush = true;
                waitPersistentPush();
                clearPersistentPush();
            }
            if (clearedPull)
                initPersistentPull();
            if (clearedPush)
                initPersistentPush();
        }
    };

    template <class TArray>
    struct ArrayTransformerType
    {
        using Type = ArrayTransformer<typename TArray::value_type, TArray::rs, TArray::rm, TArray::al>;
    };

}
