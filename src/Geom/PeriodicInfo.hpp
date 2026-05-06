#pragma once

#include "DNDS/ArrayTransformer.hpp"
#include "DNDS/ArrayPair.hpp"
#include "Geometric.hpp"
#include "BoundaryCondition.hpp"
#include "DNDS/Serializer/SerializerBase.hpp"

namespace DNDS::Geom
{
    struct NodePeriodicBits
    {
        uint8_t __v{0U};
        [[nodiscard]] bool getP1() const { return __v & 0x01U; }
        [[nodiscard]] bool getP2() const { return __v & 0x02U; }
        [[nodiscard]] bool getP3() const { return __v & 0x04U; }
        void setP1True() { __v |= 0x01U; }
        void setP2True() { __v |= 0x02U; }
        void setP3True() { __v |= 0x04U; }
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(NodePeriodicBits, NodePeriodicBits)
        DNDS_DEVICE_CALLABLE uint8_t operator^(const NodePeriodicBits &r) const
        {
            return uint8_t(__v ^ r.__v);
        }
        DNDS_DEVICE_CALLABLE NodePeriodicBits operator&(const NodePeriodicBits &r) const
        {
            return NodePeriodicBits{uint8_t(__v & r.__v)};
        }
        DNDS_DEVICE_CALLABLE operator uint8_t() const
        {
            return uint8_t{__v};
        }
        DNDS_DEVICE_CALLABLE operator bool() const
        {
            return bool(__v);
        }
        DNDS_DEVICE_CALLABLE bool operator==(const NodePeriodicBits &r) const
        {
            return uint8_t(r) == uint8_t(*this);
        }
        static MPI_Datatype CommType() { return MPI_UINT8_T; }
        static int CommMult() { return 1; }

        friend std::ostream &operator<<(std::ostream &o, const NodePeriodicBits &b)
        {
            o << int(b.__v);
            return o;
        }
    };

    static const NodePeriodicBits nodePB1{0x01U};
    static const NodePeriodicBits nodePB2{0x02U};
    static const NodePeriodicBits nodePB3{0x04U};

}
namespace DNDS
{
    //     DNDS_DEVICE_STORAGE_BASE_DELETER_INST(Geom::NodePeriodicBits, extern)
    //     DNDS_DEVICE_STORAGE_INST(Geom::NodePeriodicBits, DeviceBackend::Host, extern)
    // #ifdef DNDS_USE_CUDA
    //     DNDS_DEVICE_STORAGE_INST(Geom::NodePeriodicBits, DeviceBackend::CUDA, extern)
    // #endif
}
namespace DNDS::Geom
{

    struct NodeIndexPBI
    {
        index i;
        NodePeriodicBits pbi;

        bool operator<(const NodeIndexPBI &r) const
        {
            if (i < r.i)
                return true;
            else if (i == r.i)
                return uint8_t(pbi) < uint8_t(r.pbi);
            else
                return false;
        }

        bool operator==(const NodeIndexPBI &r) const
        {
            return r.i == i && r.pbi == pbi;
        }

        bool operator!=(const NodeIndexPBI &r) const { return !(*this == r); }
    };

    inline bool isCollaborativeNodePeriodicBits(const std::vector<NodePeriodicBits> &a, const std::vector<NodePeriodicBits> &b)
    {
        size_t n = a.size();
        DNDS_assert(n == b.size());
        if (n == 0)
            return true;
        auto v0 = a[0] ^ b[0];
        for (size_t i = 1; i < n; i++)
            if ((a.at(i) ^ b.at(i)) != v0)
                return false;
        return true;
    }

    class NodePeriodicBitsRow // instead of std::vector<NodePeriodicBits> for building on raw buffer as a "mapping" object
    {
        NodePeriodicBits *__p_indices;
        rowsize __Row_size;

    public:
        NodePeriodicBitsRow(NodePeriodicBits *ptr, rowsize siz) : __p_indices(ptr), __Row_size(siz) {} // default actually

        NodePeriodicBits &operator[](rowsize j)
        {
            DNDS_assert(j >= 0 && j < __Row_size);
            return __p_indices[j];
        }

        NodePeriodicBits operator[](rowsize j) const
        {
            DNDS_assert(j >= 0 && j < __Row_size);
            return __p_indices[j];
        }

        operator std::vector<NodePeriodicBits>() const // copies to a new std::vector<index>
        {
            return std::vector<NodePeriodicBits>(__p_indices, __p_indices + __Row_size);
        }

        void operator=(const std::vector<NodePeriodicBits> &r)
        {
            DNDS_assert(__Row_size == r.size());
            std::copy(r.begin(), r.end(), __p_indices);
        }

        void operator=(const NodePeriodicBitsRow &r)
        {
            DNDS_assert(__Row_size == r.size());
            std::copy(r.cbegin(), r.cend(), __p_indices);
        }

        NodePeriodicBits bitandReduce()
        {
            NodePeriodicBits ret;
            ret.setP1True();
            ret.setP2True();
            ret.setP3True();
            for (auto &v : *this)
                ret = ret & v;
            return ret;
        }

        NodePeriodicBits *begin() { return __p_indices; }
        NodePeriodicBits *end() { return __p_indices + __Row_size; } // past-end
        const NodePeriodicBits *cbegin() const { return __p_indices; }
        const NodePeriodicBits *cend() const { return __p_indices + __Row_size; } // past-end
        [[nodiscard]] rowsize size() const { return __Row_size; }
    };

    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    class ArrayNodePeriodicBits : public ParArray<NodePeriodicBits, _row_size, _row_max, _align>
    {
    public:
        using t_base = ParArray<NodePeriodicBits, _row_size, _row_max, _align>;
        using t_base::t_base;

        NodePeriodicBitsRow operator[](index i)
        {
            DNDS_assert(i < this->Size()); //! disable past-end input
            return NodePeriodicBitsRow(t_base::operator[](i), t_base::RowSize(i));
        }

        NodePeriodicBits *rowPtr(index i) { return t_base::operator[](i); }
    };
    template <rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
    using ArrayNodePeriodicBitsPair = ArrayPair<ArrayNodePeriodicBits<_row_size, _row_max, _align>>;

    struct Periodicity
    {
        std::array<tGPointPortable, 4> rotation;
        std::array<tPointPortable, 4> translation;
        std::array<tPointPortable, 4> rotationCenter;
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE_NO_EMPTY_CTOR(Periodicity, Periodicity)
        DNDS_DEVICE_CALLABLE Periodicity()
        {
            for (auto &r : rotation)
                r.map().setIdentity();
            for (auto &r : rotationCenter)
                r.map().setZero();
            translation[0].map().setZero();
            translation[1].map() = tPoint{1, 0, 0};
            translation[2].map() = tPoint{0, 1, 0};
            translation[3].map() = tPoint{0, 0, 1};

            //     translation[1] = tPoint{0, 0, 0};
            //     rotation[1] << 0, 1, 0,
            //         -1, 0, 0,
            //         0, 0, 1;
        }

        DNDS_HOST void WriteSerializer(Serializer::SerializerBaseSSP serializerP, const std::string &name)
        {
            auto cwd = serializerP->GetCurrentPath();
            serializerP->CreatePath(name);
            serializerP->GoToPath(name);

            for (int i = 1; i <= 3; i++)
            {
                serializerP->WriteRealVector("rotation" + std::to_string(i), Geom::JacobiToSTDVector(rotation.at(i).map()), Serializer::ArrayGlobalOffset_One);
                serializerP->WriteRealVector("rotationCenter" + std::to_string(i), Geom::VectorToSTDVector(rotationCenter.at(i).map()), Serializer::ArrayGlobalOffset_One);
                serializerP->WriteRealVector("translation" + std::to_string(i), Geom::VectorToSTDVector(translation.at(i).map()), Serializer::ArrayGlobalOffset_One);
            }

            serializerP->GoToPath(cwd);
        }

        DNDS_HOST void ReadSerializer(Serializer::SerializerBaseSSP serializerP, const std::string &name)
        {
            auto cwd = serializerP->GetCurrentPath();
            // serializerP->CreatePath(name); // * no create
            serializerP->GoToPath(name);

            for (int i = 1; i <= 3; i++)
            {
                std::vector<real> rotRead, rotCRead, transRead;
                Serializer::ArrayGlobalOffset offsetV = Serializer::ArrayGlobalOffset_One; // must explicitly set as One
                serializerP->ReadRealVector("rotation" + std::to_string(i), rotRead, offsetV);
                serializerP->ReadRealVector("rotationCenter" + std::to_string(i), rotCRead, offsetV);
                serializerP->ReadRealVector("translation" + std::to_string(i), transRead, offsetV);
                rotation.at(i).map() = Geom::STDVectorToJacobi(rotRead);
                rotationCenter.at(i).map() = Geom::STDVectorToVector(rotCRead);
                translation.at(i).map() = Geom::STDVectorToVector(transRead);
            }

            serializerP->GoToPath(cwd);
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] tPoint TransCoord(const tPoint &c, t_index id) const
        {
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            return rotation.at(i).map() * (c - rotationCenter.at(i).map()) + rotationCenter.at(i).map() + translation.at(i).map();
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] tPoint TransCoordBack(const tPoint &c, t_index id) const
        {
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            return rotation.at(i).map().transpose() * ((c - translation.at(i).map()) - rotationCenter.at(i).map()) + rotationCenter.at(i).map();
        }

        ///@todo //TODO: add support for cartesian tensor transformation

        template <int dim, int nVec>
        DNDS_DEVICE_CALLABLE Eigen::Matrix<real, dim, nVec> TransVector(const Eigen::Matrix<real, dim, nVec> &v, t_index id)
        {
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            if constexpr (dim == 3)
                return rotation.at(i).map() * v;
            else
                return rotation.at(i).map()({0, 1}, {0, 1}) * v;
        }

        template <int dim, int nVec>
        DNDS_DEVICE_CALLABLE Eigen::Matrix<real, dim, nVec> TransVectorBack(const Eigen::Matrix<real, dim, nVec> &v, t_index id)
        {
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            if constexpr (dim == 3)
                return rotation.at(i).map().transpose() * v;
            else
                return rotation.at(i).map()({0, 1}, {0, 1}).transpose() * v;
        }

        template <int dim>
        DNDS_DEVICE_CALLABLE Eigen::Matrix<real, dim, dim> TransMat(const Eigen::Matrix<real, dim, dim> &m, t_index id)
        {
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            if constexpr (dim == 3)
                return rotation.at(i).map() * m * rotation.at(i).map().transpose();
            else
                return rotation.at(i).map()({0, 1}, {0, 1}) * m * rotation.at(i).map()({0, 1}, {0, 1}).transpose();
        }

        template <int dim>
        DNDS_DEVICE_CALLABLE Eigen::Matrix<real, dim, dim> TransMatBack(const Eigen::Matrix<real, dim, dim> &m, t_index id)
        {
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            if constexpr (dim == 3)
                return rotation.at(i).map().transpose() * m * rotation.at(i).map();
            else
                return rotation.at(i).map()({0, 1}, {0, 1}).transpose() * m * rotation.at(i).map()({0, 1}, {0, 1});
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] tPoint GetCoordByBits(const tPoint &c, const NodePeriodicBits &bits) const
        {
            if (!bool(bits))
                return c;
            tPoint ret = c;
            if (bits.getP3())
                ret = this->TransCoord(ret, BC_ID_PERIODIC_3);
            if (bits.getP2())
                ret = this->TransCoord(ret, BC_ID_PERIODIC_2);
            if (bits.getP1())
                ret = this->TransCoord(ret, BC_ID_PERIODIC_1);
            return ret;
        }

        template <int dim, int nVec>
        DNDS_DEVICE_CALLABLE auto GetVectorByBits(const Eigen::Matrix<real, dim, nVec> &v, const NodePeriodicBits &bits)
        {
            if (!bool(bits))
                return v;
            Eigen::Matrix<real, dim, nVec> ret = v;
            if (bits.getP3())
                ret = this->TransVector<dim, nVec>(ret, BC_ID_PERIODIC_3);
            if (bits.getP2())
                ret = this->TransVector<dim, nVec>(ret, BC_ID_PERIODIC_2);
            if (bits.getP1())
                ret = this->TransVector<dim, nVec>(ret, BC_ID_PERIODIC_1);
            return ret;
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] tPoint GetCoordBackByBits(const tPoint &c, const NodePeriodicBits &bits) const
        {
            if (!bool(bits))
                return c;
            tPoint ret = c;
            if (bits.getP1())
                ret = this->TransCoordBack(ret, BC_ID_PERIODIC_1);
            if (bits.getP2())
                ret = this->TransCoordBack(ret, BC_ID_PERIODIC_2);
            if (bits.getP3())
                ret = this->TransCoordBack(ret, BC_ID_PERIODIC_3);
            return ret;
        }

        template <int dim, int nVec>
        DNDS_DEVICE_CALLABLE auto GetVectorBackByBits(const Eigen::Matrix<real, dim, nVec> &v, const NodePeriodicBits &bits)
        {
            if (!bool(bits))
                return v;
            Eigen::Matrix<real, dim, nVec> ret = v;
            if (bits.getP1())
                ret = this->TransVectorBack<dim, nVec>(ret, BC_ID_PERIODIC_1);
            if (bits.getP2())
                ret = this->TransVectorBack<dim, nVec>(ret, BC_ID_PERIODIC_2);
            if (bits.getP3())
                ret = this->TransVectorBack<dim, nVec>(ret, BC_ID_PERIODIC_3);
            return ret;
        }
    };
} // namespace DNDS::Geom
