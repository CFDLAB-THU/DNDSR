/// @file SerializerBase.cpp
/// @brief Out-of-line definition of the virtual destructor of @ref DNDS::SerializerBase "SerializerBase";
/// separated so the vtable has a single translation-unit home.

#include "SerializerBase.hpp"
#include <algorithm>

namespace DNDS::Serializer
{
    SerializerBase::~SerializerBase() = default;

    bool SerializerBase::RowStartAll(bool value)
    {
        if (IsPerRank())
            return value;
        int local = value, global = 0;
        MPI::Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, getMPI().comm);
        return global != 0;
    }

    void SerializerBase::RowStartCheck(bool valid, const std::string &message)
    {
        DNDS_check_throw_info(RowStartAll(valid), message);
    }

    bool SerializerBase::RowStartSamePath(const std::string &path)
    {
        if (IsPerRank())
            return true;
        RowStartCheck(path.size() <= size_t(std::numeric_limits<int>::max()), "Row-start reference path is too long");
        int size = int(path.size());
        MPI::Bcast(&size, 1, MPI_INT, 0, getMPI().comm);
        std::string rootPath = path;
        rootPath.resize(size);
        MPI::Bcast(rootPath.data(), size, MPI_CHAR, 0, getMPI().comm);
        return RowStartAll(path == rootPath);
    }

    void SerializerBase::ValidateRowStartWrite(const ssp<const RowStartVector> &v, ArrayGlobalOffset data)
    {
        bool valid = v && v->size() > 0 && v->size() <= size_t(std::numeric_limits<index>::max());
        if (valid)
            valid = v->at(0) == 0 && std::is_sorted(v->begin(), v->end());
        if (valid && !IsPerRank())
            valid = data.isDist() && data.size() == v->at(v->size() - 1) &&
                    data.size() <= std::numeric_limits<index>::max() - data.offset();
        if (IsPerRank())
            valid = valid && data == ArrayGlobalOffset_Unknown;
        RowStartCheck(valid, "Invalid local CSR row starts or flat-data region");
        auto found = rowStartWrites.find(v.get());
        RowStartCheck(found == rowStartWrites.end() ||
                          (found->second.data == data && found->second.size == v->size()),
                      "Shared row-start write changed its encoding within one session");
    }

    void SerializerBase::ValidateRowStartRead(const std::string &path, ArrayGlobalOffset rows)
    {
        bool valid = rows.isDist() && rows.size() >= 0 &&
                     rows.size() < std::numeric_limits<index>::max() &&
                     rows.offset() <= std::numeric_limits<index>::max() - rows.size() - 1;
        auto found = rowStartReads.find(path);
        RowStartCheck(valid && (found == rowStartReads.end() || found->second.rows == rows),
                      "Shared row-start reads require one resolved row slice per dataset per session");
    }

    ArrayGlobalOffset SerializerBase::NormalizeRowStarts(RowStartVector &v, bool distributed)
    {
        bool valid = v.size() > 0 && v.at(0) >= 0 && std::is_sorted(v.begin(), v.end());
        if (!distributed)
            valid = valid && v.at(0) == 0;
        RowStartCheck(valid, "Invalid stored CSR row starts");
        index base = v.at(0);
        index count = v.at(v.size() - 1) - base;
        if (distributed)
            for (auto &value : v)
                value -= base;
        return distributed ? ArrayGlobalOffset{count, base} : ArrayGlobalOffset_Unknown;
    }
}
