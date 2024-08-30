#pragma once

#include "DNDS/ArrayTransformer.hpp"
#include "Geometric.hpp"
#include "BoundaryCondition.hpp"
#include "DNDS/SerializerBase.hpp"

namespace DNDS::Geom
{
    struct NodePeriodicBits
    {
        uint8_t __v{0u};
        bool getP1() const { return __v & 0x01u; }
        bool getP2() const { return __v & 0x02u; }
        bool getP3() const { return __v & 0x04u; }
        void setP1True() { __v |= 0x01u; }
        void setP2True() { __v |= 0x02u; }
        void setP3True() { __v |= 0x04u; }
        uint8_t operator^(const NodePeriodicBits &r) const
        {
            return uint8_t(__v ^ r.__v);
        }
        NodePeriodicBits operator&(const NodePeriodicBits &r) const
        {
            return NodePeriodicBits{uint8_t(__v & r.__v)};
        }
        operator uint8_t() const
        {
            return uint8_t{__v};
        }
        operator bool() const
        {
            return bool(__v);
        }
        static MPI_Datatype CommType() { return MPI_UINT8_T; }
        static int CommMult() { return 1; }

        inline friend std::ostream &operator<<(std::ostream &o, const NodePeriodicBits &b)
        {
            o << int(b.__v);
            return o;
        }
    };

    static const NodePeriodicBits nodePB1{0x01u};
    static const NodePeriodicBits nodePB2{0x02u};
    static const NodePeriodicBits nodePB3{0x04u};

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
        rowsize size() const { return __Row_size; }
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

    struct Periodicity
    {
        std::array<tGPoint, 4> rotation;
        std::array<tPoint, 4> translation;
        std::array<tPoint, 4> rotationCenter;
        Periodicity()
        {
            for (auto &r : rotation)
                r.setIdentity();
            for (auto &r : rotationCenter)
                r.setZero();
            translation[0].setZero();
            translation[1] = tPoint{1, 0, 0};
            translation[2] = tPoint{0, 1, 0};
            translation[3] = tPoint{0, 0, 1};

            //     translation[1] = tPoint{0, 0, 0};
            //     rotation[1] << 0, 1, 0,
            //         -1, 0, 0,
            //         0, 0, 1;
        }

        void WriteSerializer(SerializerBase *serializer, const std::string &name)
        {
            auto cwd = serializer->GetCurrentPath();
            serializer->CreatePath(name);
            serializer->GoToPath(name);

            for (int i = 1; i <= 3; i++)
            {
                serializer->WriteRealVector("rotation" + std::to_string(i), Geom::JacobiToSTDVector(rotation.at(i)));
                serializer->WriteRealVector("rotationCenter" + std::to_string(i), Geom::VectorToSTDVector(rotationCenter.at(i)));
                serializer->WriteRealVector("translation" + std::to_string(i), Geom::VectorToSTDVector(translation.at(i)));
            }

            serializer->GoToPath(cwd);
        }

        void ReadSerializer(SerializerBase *serializer, const std::string &name)
        {
            auto cwd = serializer->GetCurrentPath();
            // serializer->CreatePath(name); // * no create
            serializer->GoToPath(name);

            for (int i = 1; i <= 3; i++)
            {
                std::vector<real> rotRead, rotCRead, transRead;
                serializer->ReadRealVector("rotation" + std::to_string(i), rotRead);
                serializer->ReadRealVector("rotationCenter" + std::to_string(i), rotCRead);
                serializer->ReadRealVector("translation" + std::to_string(i), transRead);
                rotation.at(i) = Geom::STDVectorToJacobi(rotRead);
                rotationCenter.at(i) = Geom::STDVectorToVector(rotCRead);
                translation.at(i) = Geom::STDVectorToVector(transRead);
            }

            serializer->GoToPath(cwd);
        }

        tPoint TransCoord(const tPoint &c, t_index id) const
        {
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            return rotation.at(i) * (c - rotationCenter.at(i)) + rotationCenter.at(i) + translation.at(i);
        }

        tPoint TransCoordBack(const tPoint &c, t_index id) const
        {
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            return rotation.at(i).transpose() * ((c - translation.at(i)) - rotationCenter.at(i)) + rotationCenter.at(i);
        }

        ///@todo //TODO: add support for cartesian tensor transformation

        template <int dim, int nVec>
        Eigen::Matrix<real, dim, nVec> TransVector(const Eigen::Matrix<real, dim, nVec> &v, t_index id)
        {
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            if constexpr (dim == 3)
                return rotation.at(i) * v;
            else
                return rotation.at(i)({0, 1}, {0, 1}) * v;
        }

        template <int dim, int nVec>
        Eigen::Matrix<real, dim, nVec> TransVectorBack(const Eigen::Matrix<real, dim, nVec> &v, t_index id)
        {
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            if constexpr (dim == 3)
                return rotation.at(i).transpose() * v;
            else
                return rotation.at(i)({0, 1}, {0, 1}).transpose() * v;
        }

        template <int dim>
        Eigen::Matrix<real, dim, dim> TransMat(const Eigen::Matrix<real, dim, dim> &m, t_index id)
        {
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            if constexpr (dim == 3)
                return rotation.at(i) * m * rotation.at(i).transpose();
            else
                return rotation.at(i)({0, 1}, {0, 1}) * m * rotation.at(i)({0, 1}, {0, 1}).transpose();
        }

        template <int dim>
        Eigen::Matrix<real, dim, dim> TransMatBack(const Eigen::Matrix<real, dim, dim> &m, t_index id)
        {
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            if constexpr (dim == 3)
                return rotation.at(i).transpose() * m * rotation.at(i);
            else
                return rotation.at(i)({0, 1}, {0, 1}).transpose() * m * rotation.at(i)({0, 1}, {0, 1});
        }

        tPoint GetCoordByBits(const tPoint &c, const NodePeriodicBits &bits)
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
        auto GetVectorByBits(const Eigen::Matrix<real, dim, nVec> &v, const NodePeriodicBits &bits)
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

        tPoint GetCoordBackByBits(const tPoint &c, const NodePeriodicBits &bits)
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
        auto GetVectorBackByBits(const Eigen::Matrix<real, dim, nVec> &v, const NodePeriodicBits &bits)
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
}
