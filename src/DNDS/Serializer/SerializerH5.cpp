#include "SerializerH5.hpp"
#include <algorithm>
#include <string>
#include <fmt/core.h>
#include "DNDS/Serializer/HDF5.hpp"

namespace DNDS::Serializer
{
#define H5CHECK_Set DNDS_assert_info(herr >= 0, "H5 setting err")
#define H5CHECK_Close DNDS_assert_info(herr >= 0, "H5 closing err")
#define H5CHECK_Iter DNDS_assert_info(herr >= 0, "H5 iteration err")
    /**
     * @brief returns good path and if the path is absolute
     */
    static bool processPath(std::vector<std::string> &pth)
    {
        bool ifAbs = pth.empty() || pth[0].empty();
        pth.erase(
            std::remove_if(pth.begin(), pth.end(), [](const std::string &v)
                           { return v.empty(); }), // only keep non-zero sized path names
            pth.end());

        return ifAbs;
    }

    static std::string constructPath(std::vector<std::string> &pth)
    {
        std::string ret;
        for (auto &name : pth)
            ret.append(std::string("/") + name);
        return ret;
    }

    /**
     * @brief Get the Group Of File If Exist
     *
     * @param file_id
     * @param read
     * @param groupName
     * @return hid_t group_id, need releasing
     * @warning remember releasing returned group_id!
     */
    static hid_t GetGroupOfFileIfExist(hid_t file_id, bool read, const std::string &groupName, bool coll_on_meta)
    {
        hid_t group_id{-1};
        herr_t herr{0};

        auto pth = splitSString(groupName, '/');
        bool isAbs = processPath(pth);
        // std::cout << groupName << std::endl;
        DNDS_assert_info(isAbs, fmt::format("groupName: {}", groupName));
        if (pth.empty()) // is root itself
        {
            group_id = H5Gopen(file_id, "/", H5P_DEFAULT);
            DNDS_assert(group_id >= 0);
        }
        for (int i = 0; i < pth.size(); i++)
        {
            std::vector<std::string> pth_parent;
            for (int j = 0; j <= i; j++)
                pth_parent.push_back(pth[j]);
            auto parentGroupName = constructPath(pth_parent);
            hid_t lapl_id = H5Pcreate(H5P_LINK_ACCESS);
            DNDS_assert(lapl_id >= 0);
            if (coll_on_meta)
                herr = H5Pset_all_coll_metadata_ops(lapl_id, true), H5CHECK_Set;
            htri_t group_exists = H5Lexists(file_id, parentGroupName.c_str(), lapl_id);
            herr = H5Pclose(lapl_id), H5CHECK_Close;
            if (group_exists > 0)
            {
                // If the group exists, open it if it the deepest level
                if (i == pth.size() - 1)
                {
                    hid_t gapl_id = H5Pcreate(H5P_GROUP_ACCESS);
                    DNDS_assert(gapl_id >= 0);
                    if (coll_on_meta)
                        herr = H5Pset_all_coll_metadata_ops(gapl_id, true), H5CHECK_Set;
                    group_id = H5Gopen(file_id, parentGroupName.c_str(), gapl_id);
                    herr = H5Pclose(gapl_id), H5CHECK_Close;
                    // std::cout << parentGroupName << " exists" << std::endl;
                }
                else
                    group_id = -1;
            }
            else
            {
                DNDS_assert_info(!read, "file is read only, cannot create new group: " + groupName);

                // If the group does not exist, create it
                hid_t gapl_id = H5Pcreate(H5P_GROUP_ACCESS);
                DNDS_assert(gapl_id >= 0);
                if (coll_on_meta)
                    herr = H5Pset_all_coll_metadata_ops(gapl_id, true), H5CHECK_Set;
                group_id = H5Gcreate(file_id, parentGroupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, gapl_id);
                DNDS_assert(group_id >= 0);
                herr = H5Pclose(gapl_id), H5CHECK_Close;
                // std::cout << parentGroupName << " created" << std::endl;
                if (i < pth.size() - 1)
                {
                    herr = H5Gclose(group_id), H5CHECK_Close;
                    group_id = -1;
                }
            }
        }

        return group_id;
    }

    void SerializerH5::OpenFile(const std::string &fName, bool read)
    {
        reading = read;
        cP = "";
        herr_t herr{0};
        hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
        herr = H5Pset_fapl_mpio(plist_id, commDup, MPI_INFO_NULL), H5CHECK_Set; // Set up file access property list with parallel I/O access
        herr = H5Pset_all_coll_metadata_ops(plist_id, true), H5CHECK_Set;
        herr = H5Pset_coll_metadata_write(plist_id, true), H5CHECK_Set;
        if (read)
            h5file = H5Fopen(fName.c_str(), H5F_ACC_RDONLY, plist_id);
        else
            h5file = H5Fcreate(fName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
        DNDS_assert_info(H5I_INVALID_HID != h5file, fmt::format(" attempted to {} file [{}]", read ? "read" : "write", fName));
        herr = H5Pclose(plist_id), H5CHECK_Close;
    }
    void SerializerH5::CloseFile()
    {
        CloseFileNonVirtual();
    }
    void SerializerH5::CloseFileNonVirtual()
    {
        cPathSplit.clear();
        dedupClear();
        cP.clear();

        if (H5I_INVALID_HID != h5file)
            H5Fclose(h5file), h5file = H5I_INVALID_HID;
    }
    void SerializerH5::CreatePath(const std::string &p)
    {
        herr_t herr{0};
        auto pth = splitSString(p, '/');
        bool isAbs = processPath(pth);
        std::vector<std::string> newPath = cPathSplit;
        if (isAbs)
            newPath = std::move(pth);
        else
            for (auto &name : pth)
                newPath.push_back(name);
        std::string nP = constructPath(cPathSplit);
        hid_t group_id = GetGroupOfFileIfExist(h5file, reading, nP, collectiveMetadataRW);
        herr = H5Gclose(group_id), H5CHECK_Close;
    }
    void SerializerH5::GoToPath(const std::string &p)
    {
        auto pth = splitSString(p, '/');
        bool isAbs = processPath(pth);
        if (isAbs)
            cPathSplit = std::move(pth);
        else
            for (auto &name : pth)
                cPathSplit.push_back(name);
        cP = constructPath(cPathSplit);
        // DNDS_assert(jObj[nlohmann::json::json_pointer(cP)].is_object());
    }
    std::string SerializerH5::GetCurrentPath()
    {
        return cP;
    }

    // Structure to hold collected HDF5 information
    struct H5Contents
    {
        std::vector<std::string> groups;
        std::vector<std::string> datasets;
        std::vector<std::string> attributes;
    };

    struct TraverseData
    {
        H5Contents &contents;
        std::string current_path;
        bool coll_on_meta;
        std::string get_indent() const
        {
            return std::string(std::count(current_path.begin(), current_path.end(), '/') * 2, ' ');
        }
    };

    static herr_t link_iterate_cb(hid_t group_id, const char *name, const H5L_info_t *info, void *op_data)
    {
        herr_t herr;
        auto *data = static_cast<TraverseData *>(op_data);
        bool coll_on_meta = data->coll_on_meta;
        std::string full_name = data->current_path + "/" + name;
        // We only care about hard links that represent HDF5 objects
        // std::cout << "name is " << name << std::endl;
        if (info->type == H5L_TYPE_HARD)
        {
            data->contents.groups.push_back(name);
            return 0;
            //! now it seems obj_info is not correctly retrieved with type=Unknown
            //! TODO: fix this and get actual object type
            H5O_info_t obj_info;
            hid_t lapl_id = H5Pcreate(H5P_LINK_ACCESS);
            DNDS_assert(lapl_id >= 0);
            herr = H5Pset_all_coll_metadata_ops(lapl_id, true), H5CHECK_Set;
            // if (coll_on_meta)
            //     herr = H5Pset_all_coll_metadata_ops(lapl_id, true), H5CHECK_Set;
            if (H5Oget_info_by_name(group_id, name, &obj_info, H5P_DEFAULT, lapl_id) >= 0)
            {
                log() << "name is " << name << ", \n"
                      << obj_info.type << "\n"
                      << obj_info.mtime << "\n"
                      << obj_info.rc << "\n"
                      << obj_info.fileno << "\n"
                      << std::endl;
                switch (obj_info.type)
                {
                case H5O_TYPE_GROUP:
                {
                    data->contents.groups.push_back(name);
                    // std::cout << data->get_indent() << "Group: " << full_name << std::endl;
                    break;
                }
                case H5O_TYPE_DATASET:
                {
                    data->contents.datasets.push_back(name);
                    // std::cout << data->get_indent() << "Dataset: " << full_name << std::endl;
                    break;
                }
                default:
                    // Other types (e.g., symbols, but less common for direct user interaction)
                    break;
                }
            }
            else
            {
                log() << "Error getting object info for: " << full_name << std::endl;
                H5Eprint2(H5E_DEFAULT, stderr);
            }
            herr = H5Pclose(lapl_id), H5CHECK_Close;
        }
        else
        {
            // Handle soft links or external links if their names are also desired
            // std::cout << data->get_indent() << "Link: " << full_name << " (Non-hard link)" << std::endl;
        }
        return 0; // Continue iteration
    }

    // Callback for H5Aiterate (iterating attributes)
    static herr_t attribute_iterate_cb(hid_t obj_id, const char *attr_name, const H5A_info_t *info, void *op_data)
    {
        TraverseData *data = static_cast<TraverseData *>(op_data);
        data->contents.attributes.push_back(attr_name);
        // std::string full_attr_name = data->current_path + "@" + attr_name; // Common convention for attribute paths
        // std::cout << data->get_indent() << "  Attribute: " << full_attr_name << std::endl; // Indent more for attributes
        return 0; // Continue iteration
    }

    std::set<std::string> SerializerH5::ListCurrentPath()
    {
        hid_t group_id = GetGroupOfFileIfExist(h5file, reading, cP, collectiveMetadataRW);
        H5Contents contents;
        TraverseData data{contents, cP, collectiveMetadataRW};
        herr_t herr{0};
        herr = H5Aiterate(group_id, H5_INDEX_NAME, H5_ITER_INC, NULL, attribute_iterate_cb, &data), H5CHECK_Iter;
        herr = H5Literate(group_id, H5_INDEX_NAME, H5_ITER_INC, NULL, link_iterate_cb, &data), H5CHECK_Iter;
        H5Gclose(group_id), H5CHECK_Close; // don't forget this
        std::set<std::string> ret;
        for (auto &v : contents.attributes)
            ret.insert(v);
        for (auto &v : contents.groups)
            ret.insert(v);
        for (auto &v : contents.datasets)
            ret.insert(v);
        return ret;
    }

    template <typename T>
    // using T = int;
    void WriteAttributeScalar(const std::string &name, const T &v,
                              hid_t h5file, bool reading, const std::string &cP,
                              bool coll_on_meta)
    {
        hid_t T_H5TYPE = -1;
        if constexpr (std::is_same_v<T, int>)
            T_H5TYPE = H5T_NATIVE_INT;
        else if constexpr (std::is_same_v<T, index>)
            T_H5TYPE = DNDS_H5T_INDEX();
        else if constexpr (std::is_same_v<T, rowsize>)
            T_H5TYPE = DNDS_H5T_ROWSIZE();
        else if constexpr (std::is_same_v<T, real>)
            T_H5TYPE = DNDS_H5T_REAL();
        else if constexpr (std::is_same_v<T, std::string>)
            ; // pass
        else
            static_assert(std::is_same_v<T, real>);

        T vV = v;

        if constexpr (!std::is_same_v<T, std::string>)
        {
            herr_t herr{0};
            hid_t group_id = GetGroupOfFileIfExist(h5file, reading, cP, coll_on_meta);
            hsize_t attr_size = 1;
            hid_t attr_space = H5Screate(H5S_SCALAR);
            hid_t aapl_id = H5Pcreate(H5P_ATTRIBUTE_ACCESS);
            if (coll_on_meta)
                herr = H5Pset_all_coll_metadata_ops(aapl_id, true), H5CHECK_Set;
            hid_t attr_id = H5Acreate(group_id, name.c_str(), T_H5TYPE, attr_space, H5P_DEFAULT, aapl_id);
            herr = H5Awrite(attr_id, T_H5TYPE, &vV), H5CHECK_Set;
            H5Aclose(attr_id);
            H5Pclose(aapl_id);
            H5Sclose(attr_space);
            H5Gclose(group_id);
        }
        else
        {
            hid_t group_id = GetGroupOfFileIfExist(h5file, reading, cP, coll_on_meta);

            herr_t herr{0};
            hid_t aapl_id = H5Pcreate(H5P_ATTRIBUTE_ACCESS);
            if (coll_on_meta)
                herr = H5Pset_all_coll_metadata_ops(aapl_id, true), H5CHECK_Set;
            hid_t scalar_space = H5Screate(H5S_SCALAR);
            hid_t string_type = H5Tcreate(H5T_STRING, v.length());
            herr = H5Tset_strpad(string_type, H5T_STR_NULLPAD), H5CHECK_Set;
            hid_t type_attr_id = H5Acreate(group_id, name.c_str(), string_type, scalar_space, H5P_DEFAULT, aapl_id);
            DNDS_assert_info(type_attr_id >= 0, fmt::format("{}, {}, {}", name, v, cP));
            herr = H5Awrite(type_attr_id, string_type, v.data()), H5CHECK_Set;

            herr = H5Aclose(type_attr_id), H5CHECK_Close;
            herr = H5Tclose(string_type), H5CHECK_Close;
            herr = H5Pclose(aapl_id), H5CHECK_Close;
            herr = H5Sclose(scalar_space), H5CHECK_Close;
            herr = H5Gclose(group_id), H5CHECK_Close;
        }
    }

    void SerializerH5::WriteInt(const std::string &name, int v)
    {
        WriteAttributeScalar<int>(name, v, h5file, reading, cP, collectiveMetadataRW);
    }
    void SerializerH5::WriteIndex(const std::string &name, index v)
    {
        WriteAttributeScalar<index>(name, v, h5file, reading, cP, collectiveMetadataRW);
    }
    void SerializerH5::WriteReal(const std::string &name, real v)
    {
        WriteAttributeScalar<real>(name, v, h5file, reading, cP, collectiveMetadataRW);
    }

    static void H5_WriteDataset(hid_t loc, const char *name, index nGlobal, index nOffset, index nLocal,
                                hid_t file_dataType, hid_t mem_dataType, hid_t dxpl_id, hid_t dapl_id, int64_t chunksize, int deflateLevel,
                                const void *buf, int dim2 = -1)
    {
        int herr{0};
        DNDS_assert_info(nGlobal >= 0 && nLocal >= 0 && nOffset >= 0,
                         fmt::format("{},{},{}", nGlobal, nLocal, nOffset));
        int rank = dim2 >= 0 ? 2 : 1;
        std::array<hsize_t, 2> ranksFull{hsize_t(nGlobal), hsize_t(dim2)};
        std::array<hsize_t, 2> ranksFullUnlim{chunksize > 0 ? H5S_UNLIMITED : hsize_t(nGlobal), hsize_t(dim2)};
        std::array<hsize_t, 2> offset{hsize_t(nOffset), 0};
        std::array<hsize_t, 2> siz{hsize_t(nLocal), hsize_t(dim2)};
        hid_t memSpace = H5Screate_simple(rank, siz.data(), NULL);
        hid_t fileSpace = H5Screate_simple(rank, ranksFull.data(), ranksFullUnlim.data());
        std::array<hsize_t, 2> chunk_dims{hsize_t(chunksize > 0 ? chunksize : 0), dim2 >= 0 ? hsize_t(dim2) : 0};
        hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
        if (chunk_dims[0] > 0)
            herr = H5Pset_chunk(dcpl_id, dim2 > 0 ? 2 : 1, chunk_dims.data()), H5CHECK_Set;
#ifdef H5_HAVE_FILTER_DEFLATE
        if (deflateLevel > 0)
            herr = H5Pset_deflate(dcpl_id, deflateLevel), H5CHECK_Set;
#endif

        hid_t dset_id = H5Dcreate(loc, name, file_dataType, fileSpace, H5P_DEFAULT, dcpl_id, dapl_id);
        DNDS_assert_info(H5I_INVALID_HID != dset_id, "dataset create failed");
        herr = H5Sclose(fileSpace);
        fileSpace = H5Dget_space(dset_id);
        herr |= H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, offset.data(), NULL, siz.data(), NULL);
        herr |= H5Dwrite(dset_id, mem_dataType, memSpace, fileSpace, dxpl_id, buf);
        herr |= H5Dclose(dset_id);
        herr |= H5Pclose(dcpl_id);
        herr |= H5Sclose(fileSpace);
        herr |= H5Sclose(memSpace);
        DNDS_assert_info(herr >= 0,
                         "h5 error " + fmt::format(
                                           "nGlobal {}, nOffset {}, nLocal {}, name {}",
                                           nGlobal, nOffset, nLocal, name));
    } //! currently a direct copy

    template <typename T = index>
    // using T = index;
    static void WriteDataVector(const std::string &name, const T *v, size_t size, ArrayGlobalOffset offset, int64_t chunksize, int deflateLevel,
                                hid_t h5file, bool reading, const std::string &cP, const MPIInfo &mpi, MPI_Comm commDup,
                                bool coll_on_meta, bool coll_on_data)
    {
        hid_t T_H5TYPE = H5T_NATIVE_INT;
        if constexpr (std::is_same_v<T, index>)
            T_H5TYPE = DNDS_H5T_INDEX();
        else if constexpr (std::is_same_v<T, rowsize>)
            T_H5TYPE = DNDS_H5T_ROWSIZE();
        else if constexpr (std::is_same_v<T, real>)
            T_H5TYPE = DNDS_H5T_REAL();
        else if constexpr (std::is_same_v<T, uint8_t>)
            T_H5TYPE = H5T_NATIVE_UINT8;
        else
            static_assert(std::is_same_v<T, uint8_t>);

        herr_t herr{0};
        hid_t dxpl_id = H5Pcreate(H5P_DATASET_XFER);
        //! is this necessary? seems even global-unique vector needs this if deflate is on!
        if (((offset.isDist() || offset == ArrayGlobalOffset_Parts) && (coll_on_data)) || deflateLevel > 0)
            herr = H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE), H5CHECK_Set;
        else
            herr = H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_INDEPENDENT), H5CHECK_Set;
        DNDS_assert_info(herr >= 0, "h5 error");
        // if is dist array, we use coll metadata
        hid_t dapl_id = H5Pcreate(H5P_DATASET_ACCESS);
        if (coll_on_meta)
            herr = H5Pset_all_coll_metadata_ops(dapl_id, true), H5CHECK_Set;

        hid_t group_id = GetGroupOfFileIfExist(h5file, reading, cP, coll_on_meta);
        if (offset == ArrayGlobalOffset_One)
            H5_WriteDataset(group_id, name.c_str(), size, 0, mpi.rank == 0 ? size : 0,
                            T_H5TYPE, T_H5TYPE, dxpl_id, dapl_id, chunksize, deflateLevel,
                            v);
        else if (offset == ArrayGlobalOffset_Parts) // now we force it to be
        {
            uint64_t sizeU{size}, sizeOff{0}, sizeGlobal{0};
            MPI::Scan(&sizeU, &sizeOff, 1, MPI_UINT64_T, MPI_SUM, commDup);
            sizeGlobal = sizeOff;
            MPI::Bcast(&sizeGlobal, 1, MPI_UINT64_T, mpi.size - 1, commDup);
            sizeOff -= sizeU;

            H5_WriteDataset(group_id, name.c_str(), sizeGlobal, sizeOff, size,
                            T_H5TYPE, T_H5TYPE, dxpl_id, dapl_id, chunksize, deflateLevel,
                            v);
            // rank offset array
            std::array<index, 2> offsetC = {index(sizeOff), index(sizeGlobal)};
            H5_WriteDataset(group_id, (name + "::rank_offsets").c_str(), mpi.size + 1, mpi.rank, (mpi.rank == mpi.size - 1) ? 2 : 1,
                            DNDS_H5T_INDEX(), DNDS_H5T_INDEX(), dxpl_id, dapl_id, chunksize, deflateLevel,
                            offsetC.data());
        }
        else if (offset.isDist())
        {

            H5_WriteDataset(group_id, name.c_str(), offset.size(), offset.offset(), size,
                            T_H5TYPE, T_H5TYPE, dxpl_id, dapl_id, chunksize, deflateLevel,
                            v);
            // rank offset array
            std::array<index, 2> offsetC = {offset.offset(), offset.size()};
            H5_WriteDataset(group_id, (name + "::rank_offsets").c_str(), mpi.size + 1, mpi.rank, (mpi.rank == mpi.size - 1) ? 2 : 1,
                            DNDS_H5T_INDEX(), DNDS_H5T_INDEX(), dxpl_id, dapl_id, chunksize, deflateLevel,
                            offsetC.data());
        }
        else
            DNDS_assert_info(false, "offset ill-formed");

        herr = H5Pclose(dxpl_id), H5CHECK_Close;
        herr = H5Pclose(dapl_id), H5CHECK_Close;
        herr = H5Gclose(group_id), H5CHECK_Close;
    }

    void SerializerH5::WriteIndexVector(const std::string &name, const std::vector<index> &v, ArrayGlobalOffset offset)
    {
        WriteDataVector<index>(name, v.data(), v.size(), offset, chunksize, deflateLevel, h5file, reading, cP, mpi, commDup, collectiveMetadataRW, collectiveDataRW);
    }
    void SerializerH5::WriteRowsizeVector(const std::string &name, const std::vector<rowsize> &v, ArrayGlobalOffset offset)
    {
        WriteDataVector<rowsize>(name, v.data(), v.size(), offset, chunksize, deflateLevel, h5file, reading, cP, mpi, commDup, collectiveMetadataRW, collectiveDataRW);
    }
    void SerializerH5::WriteRealVector(const std::string &name, const std::vector<real> &v, ArrayGlobalOffset offset)
    {
        WriteDataVector<real>(name, v.data(), v.size(), offset, chunksize, deflateLevel, h5file, reading, cP, mpi, commDup, collectiveMetadataRW, collectiveDataRW);
    }
    void SerializerH5::WriteString(const std::string &name, const std::string &v)
    {
        WriteAttributeScalar<std::string>(name, v, h5file, reading, cP, collectiveMetadataRW);
    }
    void SerializerH5::WriteSharedIndexVector(const std::string &name, const ssp<host_device_vector<index>> &v, ArrayGlobalOffset offset)
    {
        std::string refPath;
        index CanNotShare = dedupLookup(v, refPath) ? 0 : 1;
        MPI::AllreduceOneIndex(CanNotShare, MPI_MAX, MPIInfo{commDup, mpi.rank, mpi.size});
        if (!CanNotShare)
            this->WriteString(name + "::ref", refPath);
        else
        {
            WriteDataVector<index>(name, v->data(), v->size(), offset, chunksize, deflateLevel, h5file, reading, cP, mpi, commDup, collectiveMetadataRW, collectiveDataRW);
            dedupRegister(v, cP + "/" + name);
        }
    }
    void SerializerH5::WriteSharedRowsizeVector(const std::string &name, const ssp<host_device_vector<rowsize>> &v, ArrayGlobalOffset offset)
    {
        std::string refPath;
        index CanNotShare = dedupLookup(v, refPath) ? 0 : 1;
        MPI::AllreduceOneIndex(CanNotShare, MPI_MAX, MPIInfo{commDup, mpi.rank, mpi.size});
        if (!CanNotShare)
            this->WriteString(name + "::ref", refPath);
        else
        {
            WriteDataVector<rowsize>(name, v->data(), v->size(), offset, chunksize, deflateLevel, h5file, reading, cP, mpi, commDup, collectiveMetadataRW, collectiveDataRW);
            dedupRegister(v, cP + "/" + name);
        }
    }

    template <typename T>
    // using T = int;
    void ReadAttributeScalar(const std::string &name, T &v,
                             hid_t h5file, bool reading, const std::string &cP,
                             bool coll_on_meta)
    {
        hid_t T_H5TYPE = -1;
        if constexpr (std::is_same_v<T, int>)
            T_H5TYPE = H5T_NATIVE_INT;
        else if constexpr (std::is_same_v<T, index>)
            T_H5TYPE = DNDS_H5T_INDEX();
        else if constexpr (std::is_same_v<T, rowsize>)
            T_H5TYPE = DNDS_H5T_ROWSIZE();
        else if constexpr (std::is_same_v<T, real>)
            T_H5TYPE = DNDS_H5T_REAL();
        else if constexpr (std::is_same_v<T, std::string>)
            ; // pass
        else
            static_assert(std::is_same_v<T, real>);

        T vV = v;

        if constexpr (!std::is_same_v<T, std::string>)
        {
            herr_t herr{0};
            hid_t group_id = GetGroupOfFileIfExist(h5file, reading, cP, coll_on_meta);
            hid_t aapl_id = H5Pcreate(H5P_ATTRIBUTE_ACCESS);
            herr = H5Pset_all_coll_metadata_ops(aapl_id, true), H5CHECK_Set;
            // herr = H5Pset_coll_metadata_write(aapl_id, true), H5CHECK_Set;  // setting to present in dapl
            hid_t attr_id = H5Aopen(group_id, name.c_str(), aapl_id);
            DNDS_assert_info(attr_id >= 0, fmt::format("attempting to open attribute [{}/{}] failed", cP, name));
            hid_t attr_space = H5Aget_space(attr_id);
            int ndims = H5Sget_simple_extent_ndims(attr_space);
            DNDS_assert_info(ndims == 0, fmt::format("attempting to read attribute [{}/{}] which is not scalar", cP, name));
            herr = H5Aread(attr_id, T_H5TYPE, &v), H5CHECK_Set;
            herr = H5Aclose(attr_id), H5CHECK_Close;
            herr = H5Pclose(aapl_id), H5CHECK_Close;
            herr = H5Sclose(attr_space), H5CHECK_Close;
            herr = H5Gclose(group_id), H5CHECK_Close;
        }
        else
        {
            herr_t herr{0};
            hid_t group_id = GetGroupOfFileIfExist(h5file, reading, cP, coll_on_meta);
            hid_t aapl_id = H5Pcreate(H5P_ATTRIBUTE_ACCESS);
            if (coll_on_meta)
            {
                herr = H5Pset_all_coll_metadata_ops(aapl_id, true), H5CHECK_Set;
                // herr = H5Pset_coll_metadata_write(aapl_id, true), H5CHECK_Set; // setting not present in aapl
            }
            hid_t attr_id = H5Aopen(group_id, name.c_str(), aapl_id);
            DNDS_assert_info(attr_id >= 0, fmt::format("attempting to open attribute [{}/{}] failed", cP, name));
            hid_t attr_space = H5Aget_space(attr_id);
            int ndims = H5Sget_simple_extent_ndims(attr_space);
            DNDS_assert_info(ndims == 0, fmt::format("attempting to read attribute [{}/{}] which is not scalar", cP, name));
            hid_t dtype_id = H5Aget_type(attr_id);
            bool is_varlen = H5Tis_variable_str(dtype_id);

            if (is_varlen)
            {
                // Variable-length string: HDF5 will allocate memory
                char *attr_value = nullptr;
                herr = H5Aread(attr_id, dtype_id, &attr_value), H5CHECK_Set;
                // std::cout << "Read Attribute (Variable-Length): " << attr_value << "\n";
                v = attr_value;            // copy as null-terminated string
                H5free_memory(attr_value); // Must free memory manually
            }
            else
            {
                // Fixed-length string: Allocate buffer based on datatype size
                size_t size = H5Tget_size(dtype_id);
                std::vector<char> buffer(size + 1, '\0'); // +1 for null terminator
                herr = H5Aread(attr_id, dtype_id, buffer.data()), H5CHECK_Set;
                // std::cout << "Read Attribute (Fixed-Length): " << buffer.data() << "\n";
                v = buffer.data(); // copy as null-terminated string
            }

            herr = H5Tclose(dtype_id), H5CHECK_Close;
            herr = H5Aclose(attr_id), H5CHECK_Close;
            herr = H5Pclose(aapl_id), H5CHECK_Close;
            herr = H5Sclose(attr_space), H5CHECK_Close;
            herr = H5Gclose(group_id), H5CHECK_Close;
        }
    }

    void SerializerH5::ReadInt(const std::string &name, int &v)
    {
        ReadAttributeScalar<int>(name, v, h5file, reading, cP, collectiveMetadataRW);
    }
    void SerializerH5::ReadIndex(const std::string &name, index &v)
    {
        ReadAttributeScalar<index>(name, v, h5file, reading, cP, collectiveMetadataRW);
    }
    void SerializerH5::ReadReal(const std::string &name, real &v)
    {
        ReadAttributeScalar<real>(name, v, h5file, reading, cP, collectiveMetadataRW);
    }

    /**
     * @brief
     *
     * when #buf == 2, only set nGlobal and dim2
     *
     * @param loc
     * @param name
     * @param nGlobal
     * @param nOffset
     * @param nLocal
     * @param file_dataType
     * @param mem_dataType
     * @param plist_id
     * @param dcpl_id
     * @param dapl_id
     * @param buf
     * @param dim2
     */
    // When dxpl_id uses collective I/O (H5FD_MPIO_COLLECTIVE), every rank MUST
    // call H5Dread even if nLocal==0.  Callers must provide a valid (possibly
    // dummy) buf pointer; do NOT skip this function based on nLocal==0.
    static void H5_ReadDataset(hid_t loc, const char *name, index &nGlobal, index nOffset, index nLocal,
                               hid_t mem_dataType, hid_t dxpl_id, hid_t dapl_id,
                               void *buf, int &dim2)
    {
        int herr{0};
        hid_t dset_id = H5Dopen(loc, name, dapl_id);
        DNDS_assert_info(dset_id >= 0, fmt::format("dataset [{}] open failed", name));
        hid_t fileSpace = H5Dget_space(dset_id);
        DNDS_assert_info(fileSpace >= 0, fmt::format("dataset [{}] filespace open failed", name));
        int ndims = H5Sget_simple_extent_ndims(fileSpace);
        DNDS_assert_info(ndims == 1 || ndims == 2, fmt::format("dataset [{}] not having 1 or 2 dims!", name));
        std::array<hsize_t, 2> sizes;
        ndims = H5Sget_simple_extent_dims(fileSpace, sizes.data(), nullptr);
        if (ndims == 2)
            dim2 = sizes[1];
        else
            dim2 = -1;
        nGlobal = sizes[0];

        if (buf != nullptr)
        {
            int rank = dim2 >= 0 ? 2 : 1;
            std::array<hsize_t, 2> offset{hsize_t(nOffset), 0};
            std::array<hsize_t, 2> siz{hsize_t(nLocal), hsize_t(dim2)};
            hid_t memSpace = H5Screate_simple(rank, siz.data(), NULL);
            DNDS_assert(memSpace > 0);
            herr = H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, offset.data(), NULL, siz.data(), NULL), H5CHECK_Set;
            herr = H5Dread(dset_id, mem_dataType, memSpace, fileSpace, dxpl_id, buf), H5CHECK_Set;
            herr = H5Sclose(memSpace), H5CHECK_Close;
        }
        herr = H5Dclose(dset_id), H5CHECK_Close;
        herr = H5Sclose(fileSpace), H5CHECK_Close;
    }

    template <typename T = index>
    // using T = index;
    static void ReadDataVector(const std::string &name, T *v, size_t &size, ArrayGlobalOffset &offset,
                               hid_t h5file, bool reading, const std::string &cP, const MPIInfo &mpi,
                               bool coll_on_meta, bool coll_on_data)
    {
        hid_t T_H5TYPE = H5T_NATIVE_INT;
        if constexpr (std::is_same_v<T, index>)
            T_H5TYPE = DNDS_H5T_INDEX();
        else if constexpr (std::is_same_v<T, rowsize>)
            T_H5TYPE = DNDS_H5T_ROWSIZE();
        else if constexpr (std::is_same_v<T, real>)
            T_H5TYPE = DNDS_H5T_REAL();
        else if constexpr (std::is_same_v<T, uint8_t>)
            T_H5TYPE = H5T_NATIVE_UINT8;
        else
            static_assert(std::is_same_v<T, uint8_t>);

        herr_t herr{0};
        hid_t dxpl_id = H5Pcreate(H5P_DATASET_XFER);
        if ((offset.isDist() || offset == ArrayGlobalOffset_One) && coll_on_data) //! is this necessary?
            herr = H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE), H5CHECK_Set;
        else
            herr = H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_INDEPENDENT), H5CHECK_Set;

        // if is dist array, we use coll metadata
        hid_t dapl_id = H5Pcreate(H5P_DATASET_ACCESS);
        if (coll_on_meta)
        {
            herr = H5Pset_all_coll_metadata_ops(dapl_id, offset.isDist() || offset == ArrayGlobalOffset_One), H5CHECK_Set;
            // herr = H5Pset_coll_metadata_write(dapl_id, offset.isDist() || offset == ArrayGlobalOffset_One), H5CHECK_Set;  // tsetting to present in dapl
        }
        hid_t group_id = GetGroupOfFileIfExist(h5file, reading, cP, coll_on_meta);

        if (offset == ArrayGlobalOffset_Unknown) // need detection, auto ArrayGlobalOffset_Parts or distributed array
        {
            DNDS_assert(v == nullptr); // must be a size-query call

            hid_t lapl_id = H5Pcreate(H5P_LINK_ACCESS);
            DNDS_assert(lapl_id >= 0);
            if (coll_on_meta)
                herr = H5Pset_all_coll_metadata_ops(lapl_id, offset.isDist() || offset == ArrayGlobalOffset_One), H5CHECK_Set;
            htri_t exists_single = H5Lexists(group_id, name.c_str(), lapl_id);
            htri_t exists_rank_offsets = H5Lexists(group_id, (name + "::rank_offsets").c_str(), lapl_id);
            herr = H5Pclose(lapl_id), H5CHECK_Close;

            if (exists_single > 0)
            {
                if (exists_rank_offsets > 0)
                {
                    index nGlobal_offsets{-1};
                    int dim2_offsets{-1};
                    H5_ReadDataset(group_id, (name + "::rank_offsets").c_str(), nGlobal_offsets, -1, -1, DNDS_H5T_INDEX(), dxpl_id, dapl_id, nullptr, dim2_offsets);
                    DNDS_assert(dim2_offsets == -1);
                    if (nGlobal_offsets == mpi.size + 1)
                    {
                        std::array<index, 2> offsets = {-1, -1};
                        H5_ReadDataset(group_id, (name + "::rank_offsets").c_str(), nGlobal_offsets, mpi.rank, 2, DNDS_H5T_INDEX(), dxpl_id, dapl_id, offsets.data(), dim2_offsets);
                        index nGlobal{-1};
                        int dim2{-1};
                        H5_ReadDataset(group_id, name.c_str(), nGlobal, -1, -1, T_H5TYPE, dxpl_id, dapl_id, nullptr, dim2);
                        DNDS_assert(dim2 == -1);
                        DNDS_assert(offsets[1] >= offsets[0]);
                        offset = ArrayGlobalOffset{nGlobal, offsets[0]};
                        size = offsets[1] - offsets[0];
                    }
                    else
                    {
                        DNDS_assert_info(false, "no valid determination" +
                                                    fmt::format(" [{} {}], ", exists_single, exists_rank_offsets) + name);
                    }
                }
                else
                {
                    DNDS_assert_info(false, "no valid determination; arrays of ArrayGlobalOffset_One need to be explicitly designated" +
                                                fmt::format(" [{} {}], ", exists_single, exists_rank_offsets) + name);
                }
            }
            else
                DNDS_assert_info(false, "no valid determination; no dataset found " +
                                            fmt::format(" [{} {}], ", exists_single, exists_rank_offsets) + name);
        }
        else
        {
            if (offset == ArrayGlobalOffset_One)
            {
                index nGlobal{-1};
                int dim2{-1};
                if (v != nullptr)
                {
                    H5_ReadDataset(group_id, name.c_str(), nGlobal, 0, size, T_H5TYPE, dxpl_id, dapl_id, v, dim2);
                    DNDS_assert(dim2 == -1);
                    DNDS_assert(size == nGlobal);
                }
                else
                {
                    H5_ReadDataset(group_id, name.c_str(), nGlobal, -1, -1, T_H5TYPE, dxpl_id, dapl_id, nullptr, dim2);
                    DNDS_assert(dim2 == -1);
                    size = nGlobal;
                }
            }
            // else if (offset == ArrayGlobalOffset_Parts)
            // {
            //     index nGlobal{-1};
            //     int dim2{-1};
            //     if (v != nullptr)
            //     {
            //         H5_ReadDataset(group_id, (name + "::" + std::to_string(mpi.rank)).c_str(), nGlobal, 0, size, T_H5TYPE, dxpl_id, dapl_id, v, dim2);
            //         DNDS_assert(dim2 == -1);
            //         DNDS_assert_info(size == nGlobal, fmt::format(", {}, {}", size, nGlobal));
            //     }
            //     else
            //     {
            //         H5_ReadDataset(group_id, (name + "::" + std::to_string(mpi.rank)).c_str(), nGlobal, -1, -1, T_H5TYPE, dxpl_id, dapl_id, nullptr, dim2);
            //         DNDS_assert(dim2 == -1);
            //         size = nGlobal;
            //     }
            // }
            else if (offset == ArrayGlobalOffset_EvenSplit)
            {
                // Even-split read: each rank reads ~N_global/nRanks rows.
                // First pass (v==nullptr): query global size, compute local range, set offset.
                // Second pass (v!=nullptr): read the data slice.
                index nGlobal{-1};
                int dim2{-1};
                if (v == nullptr)
                {
                    H5_ReadDataset(group_id, name.c_str(), nGlobal, -1, -1, T_H5TYPE, dxpl_id, dapl_id, nullptr, dim2);
                    DNDS_assert(dim2 == -1);
                    auto [start, end] = EvenSplitRange(mpi.rank, mpi.size, nGlobal);
                    size = end - start;
                    offset = ArrayGlobalOffset{index(size), start};
                }
                else
                {
                    DNDS_assert_info(offset.isDist(), "EvenSplit second pass must have a resolved offset");
                    H5_ReadDataset(group_id, name.c_str(), nGlobal, offset.offset(), size, T_H5TYPE, dxpl_id, dapl_id, v, dim2);
                    DNDS_assert(dim2 == -1);
                }
            }
            else if (offset.isDist())
            {
                if (v == nullptr)
                {
                    // First pass (size query): local size is already in offset.size().
                    size = offset.size();
                }
                else
                {
                    // Second pass (data read): read `size` elements at offset.offset().
                    // size==0 is valid for ranks with empty partitions (nGlobal < nRanks).
                    index nGlobal{-1};
                    int dim2{-1};
                    H5_ReadDataset(group_id, name.c_str(), nGlobal, offset.offset(), size, T_H5TYPE, dxpl_id, dapl_id, v, dim2);
                    DNDS_assert(dim2 == -1);
                }
            }
            else
                DNDS_assert_info(false, "offset ill-formed");
        }

        herr = H5Pclose(dxpl_id), H5CHECK_Close;
        herr = H5Pclose(dapl_id), H5CHECK_Close;
        herr = H5Gclose(group_id), H5CHECK_Close;
    }

    void SerializerH5::ReadIndexVector(const std::string &name, std::vector<index> &v, ArrayGlobalOffset &offset)
    {
        size_t size{0};
        index dummy{};
        ReadDataVector<index>(name, nullptr, size, offset, h5file, reading, cP, mpi, collectiveMetadataRW, collectiveDataRW);
        v.resize(size);
        DNDS_assert(!(offset == ArrayGlobalOffset_Unknown));
        ReadDataVector<index>(name, size == 0 ? &dummy : v.data(), size, offset, h5file, reading, cP, mpi, collectiveMetadataRW, collectiveDataRW);
    }
    void SerializerH5::ReadRowsizeVector(const std::string &name, std::vector<rowsize> &v, ArrayGlobalOffset &offset)
    {
        size_t size{0};
        rowsize dummy{};
        ReadDataVector<rowsize>(name, nullptr, size, offset, h5file, reading, cP, mpi, collectiveMetadataRW, collectiveDataRW);
        v.resize(size);
        DNDS_assert(!(offset == ArrayGlobalOffset_Unknown));
        ReadDataVector<rowsize>(name, size == 0 ? &dummy : v.data(), size, offset, h5file, reading, cP, mpi, collectiveMetadataRW, collectiveDataRW);
    }
    void SerializerH5::ReadRealVector(const std::string &name, std::vector<real> &v, ArrayGlobalOffset &offset)
    {
        size_t size{0};
        real dummy{};
        ReadDataVector<real>(name, nullptr, size, offset, h5file, reading, cP, mpi, collectiveMetadataRW, collectiveDataRW);
        v.resize(size);
        DNDS_assert(!(offset == ArrayGlobalOffset_Unknown));
        ReadDataVector<real>(name, size == 0 ? &dummy : v.data(), size, offset, h5file, reading, cP, mpi, collectiveMetadataRW, collectiveDataRW);
    }
    void SerializerH5::ReadString(const std::string &name, std::string &v)
    {
        ReadAttributeScalar<std::string>(name, v, h5file, reading, cP, collectiveMetadataRW);
    }
    void SerializerH5::ReadSharedIndexVector(const std::string &name, ssp<host_device_vector<index>> &v, ArrayGlobalOffset &offset)
    {
        using tValue = host_device_vector<index>;
        herr_t herr;
        std::string refPath;
        hid_t group_id = GetGroupOfFileIfExist(h5file, reading, cP, collectiveMetadataRW);
        htri_t exists_ref = H5Aexists(group_id, (name + "::ref").c_str());
        herr = H5Gclose(group_id), H5CHECK_Close;
        if (exists_ref > 0)
        {
            this->ReadString(name + "::ref", refPath);
        }
        else
        {
            refPath = cP + "/" + name;
        }

        if (pth_2_ssp.count(refPath))
        {
            v = *((ssp<tValue> *)(pth_2_ssp[refPath])); // ! reform this (and in json counterpart) to use reinterpret_cast or use STL's tools
        }
        else
        {
            v = std::make_shared<tValue>();
            pth_2_ssp[refPath] = &v;

            size_t size;
            index dummy{};
            ReadDataVector<index>(refPath, nullptr, size, offset, h5file, reading, "/", mpi, collectiveMetadataRW, collectiveDataRW);
            v->resize(size);
            DNDS_assert(!(offset == ArrayGlobalOffset_Unknown));
            ReadDataVector<index>(refPath, size == 0 ? &dummy : v->data(), size, offset, h5file, reading, "/", mpi, collectiveMetadataRW, collectiveDataRW);
        }
    }
    void SerializerH5::ReadSharedRowsizeVector(const std::string &name, ssp<host_device_vector<rowsize>> &v, ArrayGlobalOffset &offset)
    {
        using tValue = host_device_vector<rowsize>;
        herr_t herr;
        std::string refPath;
        hid_t group_id = GetGroupOfFileIfExist(h5file, reading, cP, collectiveMetadataRW);
        htri_t exists_ref = H5Aexists(group_id, (name + "::ref").c_str());
        herr = H5Gclose(group_id), H5CHECK_Close;
        if (exists_ref > 0)
        {
            this->ReadString(name + "::ref", refPath);
        }
        else
        {
            refPath = cP + "/" + name;
        }

        if (pth_2_ssp.count(refPath))
        {
            v = *((ssp<tValue> *)(pth_2_ssp[refPath])); // ! reform this (and in json counterpart) to use reinterpret_cast or use STL's tools
        }
        else
        {
            v = std::make_shared<tValue>();
            pth_2_ssp[refPath] = &v;

            size_t size;
            rowsize dummy{};
            ReadDataVector<rowsize>(refPath, nullptr, size, offset, h5file, reading, "/", mpi, collectiveMetadataRW, collectiveDataRW);
            v->resize(size);
            DNDS_assert(!(offset == ArrayGlobalOffset_Unknown));
            ReadDataVector<rowsize>(refPath, size == 0 ? &dummy : v->data(), size, offset, h5file, reading, "/", mpi, collectiveMetadataRW, collectiveDataRW);
        }
    }
    void SerializerH5::WriteUint8Array(const std::string &name, const uint8_t *data, index size, ArrayGlobalOffset offset)
    {
        WriteDataVector<uint8_t>(name, data, size, offset, chunksize, deflateLevel, h5file, reading, cP, mpi, commDup, collectiveMetadataRW, collectiveDataRW);
    }
    void SerializerH5::ReadUint8Array(const std::string &name, uint8_t *data, index &size, ArrayGlobalOffset &offset)
    {
        size_t size_size_t{0};
        if (data == nullptr)
        {
            ReadDataVector<uint8_t>(name, nullptr, size_size_t, offset, h5file, reading, cP, mpi, collectiveMetadataRW, collectiveDataRW);
            size = size_size_t; // todo: check overflow?
            DNDS_assert(!(offset == ArrayGlobalOffset_Unknown));
        }
        else
        {
            DNDS_assert(!(offset == ArrayGlobalOffset_Unknown));
            size_size_t = size;
            ReadDataVector<uint8_t>(name, data, size_size_t, offset, h5file, reading, cP, mpi, collectiveMetadataRW, collectiveDataRW);
        }
    }
}