#pragma once
/// @file ArrayEigenUniMatrixBatch.hpp
/// @brief Batch of uniform-sized Eigen matrices per row, with variable batch count.
/// @par Unit Test Coverage (test_ArrayDerived.cpp, MPI np=1,2,4)
/// - Static sizes (ArrayEigenUniMatrixBatch<2,3>) and dynamic sizes
/// - Resize, ResizeBatch, BatchSize, Rows, Cols, MSize, Compress
/// - Element access: operator()(i,j) returning Eigen::Map
/// @par Not Yet Tested
/// - GetDerivedArraySignature, WriteSerializer / ReadSerializer
/// - Device views

#include "../ArrayTransformer.hpp"
#include "DNDS/ArrayBasic.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch_DeviceView.hpp"

namespace DNDS
{
    /**
     * @brief CSR array whose rows store a *batch* of identically-sized Eigen matrices.
     *
     * @details Each row holds `BatchSize(i)` matrices of shape `_n_row x _n_col`,
     * contiguous in memory. The raw CSR row width is `BatchSize(i) * _n_row * _n_col`,
     * but the public API exposes batched semantics: @ref BatchSize reports the matrix
     * count, `operator()(i, j)` returns the `j`-th matrix in row `i`'s batch.
     *
     * Used heavily by @ref FiniteVolume and @ref VariationalReconstruction to store
     * per-quadrature-point Jacobians and basis-function coefficients: each cell
     * contributes one row, each quadrature point contributes one matrix.
     *
     * @tparam _n_row Row count of each stored matrix (compile time or `Eigen::Dynamic`).
     * @tparam _n_col Column count of each stored matrix.
     */
    template <int _n_row, int _n_col>
    class ArrayEigenUniMatrixBatch : public ParArray<real, NonUniformSize, NonUniformSize, NoAlign> // use CSR array
    {
        static_assert(_n_row >= 0 || _n_row == Eigen::Dynamic, "invalid _n_row");
        static_assert(_n_col >= 0 || _n_col == Eigen::Dynamic, "invalid _n_col");

    public:
        using t_self = ArrayEigenUniMatrixBatch<_n_row, _n_col>;
        using t_base = ParArray<real, NonUniformSize>;
        using t_base::t_base;

        // NOLINTBEGIN(bugprone-branch-clone): both non-row-vector arms of
        // the options ternary intentionally select ColMajor; mirrors the
        // pattern in EigenUtil.hpp::MatrixFMTSafe.
        using t_EigenMatrix = Eigen::Matrix<real, _n_row, _n_col,
                                            Eigen::AutoAlign |
                                                ((_n_row == 1 && _n_col != 1) ? Eigen ::RowMajor : (_n_col == 1 && _n_row != 1) ? Eigen ::ColMajor // ColMajor except for row-vector
                                                                                                                                : Eigen ::ColMajor)>;
        // NOLINTEND(bugprone-branch-clone)
        using t_EigenMap = Eigen::Map<t_EigenMatrix, Eigen::Unaligned>;             // default no buffer align and stride
        using t_EigenMap_const = Eigen::Map<const t_EigenMatrix, Eigen::Unaligned>; // default no buffer align and stride

    private:
        int _row_dynamic = _n_row > 0 ? _n_row : 0;
        int _col_dynamic = _n_col > 0 ? _n_col : 0;
        int _m_size = this->Rows() * this->Cols(); //! extra data!

    private:
        using t_base::Resize;
        using t_base::ResizeRow; // privatize basic resizing
        using t_base::operator();
        using t_base::RowSize;
        // void Resize(index, rowsize) = delete;
        // void ResizeRow(index iRow, rowsize nRowSize) = delete;
    public:
        // default copy
        ArrayEigenUniMatrixBatch(const t_self &R) = default;
        t_self &operator=(const t_self &R) = default;
        // Rule-of-five closure: all value-semantic members.
        ArrayEigenUniMatrixBatch(t_self &&) noexcept = default;
        t_self &operator=(t_self &&) noexcept = default;
        ~ArrayEigenUniMatrixBatch() = default;
        // operator= handled automatically

        void clone(const t_self &R)
        {
            this->operator=(R);
        }

        /**
         * @brief resizes all matrices to be used;
         * -1 means no change
         * @param r
         * @param c
         */
        void ResizeMatrix(int r = -1, int c = -1)
        {
            if constexpr (_n_row >= 0)
                DNDS_assert(r == -1 || r == _n_row);
            if constexpr (_n_col >= 0)
                DNDS_assert(c == -1 || c == _n_col);
            if (r >= 0)
                _row_dynamic = r;
            if (c >= 0)
                _col_dynamic = c;
            // TODO: multiplication overflow detect
            _m_size = this->Rows() * this->Cols();
            this->t_base::Resize(0);
        }

        void Resize(index n_size, int r, int c)
        {
            this->ResizeMatrix(r, c);
            this->t_base::Resize(n_size);
        }

        void Resize(index n_size)
        {
            if constexpr (_n_row > 0 && _n_col > 0)
                this->Resize(n_size, -1, -1);
            else
                DNDS_assert_info(false, "invalid call");
        }

        template <class TFRowSize>
        void Resize(index n_size, int r, int c, TFRowSize &&rsf)
        {
            this->ResizeMatrix(r, c);
            this->t_base::Resize(n_size, [&](index i)
                                 { return rsf(i) * this->MSize(); });
        }

    public:
        auto view()
        {
            return ArrayEigenUniMatrixBatchDeviceView<DeviceBackend::Host, real, _n_row, _n_col>{
                t_base::view(), _row_dynamic, _col_dynamic, _m_size};
        }

        [[nodiscard]] int Rows() const { return _n_row > 0 ? _n_row : _row_dynamic; }
        [[nodiscard]] int Cols() const { return _n_col > 0 ? _n_col : _col_dynamic; }
        [[nodiscard]] int MSize() const
        {
            if constexpr (_n_row >= 0 && _n_col >= 0)
                return _n_row * _n_col;
            else
                return _m_size;
        }

        void ResizeBatch(index i, rowsize b_size)
        {
            this->t_base::ResizeRow(i, b_size * MSize());
        }

        void ResizeRow(index i, rowsize b_size)
        {
            this->t_base::ResizeRow(i, b_size * MSize());
        }

        [[nodiscard]] rowsize BatchSize(index i) const
        {
            return this->RowSize(i);
        }

        [[nodiscard]] rowsize RowSize(index i) const
        {
            rowsize row_size_c = this->t_base::RowSize(i);
            DNDS_assert(MSize() != 0 && row_size_c % MSize() == 0);
            return row_size_c / MSize();
        }

        t_EigenMap operator()(index i, rowsize j)
        {
            DNDS_assert(j >= 0 && j < this->RowSize(i));
            // if constexpr (_n_row >= 0 && _n_col >= 0)
            return {this->t_base::operator[](i) + MSize() * j, Rows(), Cols()};
        }

        t_EigenMap_const operator()(index i, rowsize j) const
        {
            DNDS_assert(j >= 0 && j < this->RowSize(i));
            // if constexpr (_n_row >= 0 && _n_col >= 0)
            return {this->t_base::operator[](i) + MSize() * j, Rows(), Cols()};
        }

        std::vector<t_EigenMap> operator[](index i)
        {
            std::vector<t_EigenMap> ret;
            ret.reserve(this->BatchSize(i));
            for (rowsize j = 0; j < this->BatchSize(i); j++)
                ret.emplace_back(this->t_base::operator[](i) + MSize() * j, Rows(), Cols());
            return ret;
        }

        // TODO: getting sub matrix ?

        static std::string GetDerivedArraySignature()
        {
            return "ArrayEigenUniMatrixBatch__" + std::to_string(_n_row) +
                   "_" + std::to_string(_n_col);
        }

        void WriteSerializer(Serializer::SerializerBaseSSP serializerP, const std::string &name, Serializer::ArrayGlobalOffset offset)
        {
            auto cwd = serializerP->GetCurrentPath();
            serializerP->CreatePath(name);
            serializerP->GoToPath(name);

            serializerP->WriteString("DerivedType", GetDerivedArraySignature());
            serializerP->WriteInt("row_dynamic", _row_dynamic);
            serializerP->WriteInt("col_dynamic", _col_dynamic);
            serializerP->WriteInt("m_size", _m_size);
            this->t_base::WriteSerializer(serializerP, "array", offset);

            serializerP->GoToPath(cwd);
        }

        void ReadSerializer(Serializer::SerializerBaseSSP serializerP, const std::string &name, Serializer::ArrayGlobalOffset &offset)
        {
            auto cwd = serializerP->GetCurrentPath();
            // serializerP->CreatePath(name); //!remember no create path
            serializerP->GoToPath(name);

            std::string readDerivedType;
            serializerP->ReadString("DerivedType", readDerivedType);
            DNDS_assert(readDerivedType == this->GetDerivedArraySignature());
            serializerP->ReadInt("row_dynamic", _row_dynamic);
            serializerP->ReadInt("col_dynamic", _col_dynamic);
            serializerP->ReadInt("m_size", _m_size);
            this->t_base::ReadSerializer(serializerP, "array", offset);

            serializerP->GoToPath(cwd);
        }

        struct ReadSerializerMetaResult : t_base::ReadSerializerMetaResult
        {
            std::string derived_type;
            rowsize row_dynamic{0};
            rowsize col_dynamic{0};
            index m_size{0};
        };

        ReadSerializerMetaResult ReadSerializerMeta(Serializer::SerializerBaseSSP serializerP, const std::string &name)
        {
            auto cwd = serializerP->GetCurrentPath();
            serializerP->GoToPath(name);

            ReadSerializerMetaResult result;
            static_cast<typename t_base::ReadSerializerMetaResult &>(result) =
                t_base::ReadSerializerMeta(serializerP, "array");
            serializerP->ReadString("DerivedType", result.derived_type);
            serializerP->ReadInt("row_dynamic", result.row_dynamic);
            serializerP->ReadInt("col_dynamic", result.col_dynamic);
            serializerP->ReadInt("m_size", result.m_size);

            serializerP->GoToPath(cwd);
            return result;
        }

        template <DeviceBackend B>
        using t_deviceView = ArrayEigenUniMatrixBatchDeviceView<B, real, _n_row, _n_col>;

        template <DeviceBackend B>
        using t_deviceViewConst = ArrayEigenUniMatrixBatchDeviceView<B, const real, _n_row, _n_col>;

        template <DeviceBackend B>
        auto deviceView()
        {
            return t_deviceView<B>{t_base::template deviceView<B>(), _row_dynamic, _col_dynamic, _m_size};
        }

        template <DeviceBackend B>
        [[nodiscard]] auto deviceView() const
        {
            return t_deviceView<B>{t_base::template deviceView<B>(), _row_dynamic, _col_dynamic, _m_size};
        }

        using t_base::to_device;
        using t_base::to_host;

        /// @brief Non-owning view of one row's matrix batch in ArrayEigenUniMatrixBatch.
        template <DeviceBackend B>
        class UniMatrixRowView
        {
        protected:
            t_deviceView<B> view; // todo: optimize so that no need to store whole arrayview?
            index iRow;
            rowsize row_size;

        public:
            DNDS_DEVICE_CALLABLE UniMatrixRowView(const t_deviceView<B> &n_view, index n_iRow, rowsize n_row_size)
                : view(n_view), iRow(n_iRow), row_size(n_row_size)
            {
            }

            DNDS_DEVICE_CALLABLE t_EigenMap operator[](rowsize j)
            {
                DNDS_assert(j >= 0 && j < row_size);
                return view.operator()(iRow, j);
            }

            DNDS_DEVICE_CALLABLE t_EigenMap_const operator[](rowsize j) const
            {
                DNDS_assert(j >= 0 && j < row_size);
                return view.operator()(iRow, j);
            }
        };

        /// @brief Element iterator for ArrayEigenUniMatrixBatch, yielding UniMatrixRowView per row.
        template <DeviceBackend B>
        class iterator : public ArrayIteratorBase<iterator<B>>
        {
        public:
            using view_type = t_deviceView<B>;
            using t_base_iter = ArrayIteratorBase<iterator<B>>;
            using typename t_base_iter::difference_type;
            using reference = UniMatrixRowView<B>;
            using iterator_category = std::random_access_iterator_tag;

        protected:
            view_type view;

        public:
            auto getView() const { return view; }
            DNDS_DEVICE_CALLABLE iterator(const iterator &) = default;
            DNDS_DEVICE_CALLABLE iterator &operator=(const iterator &) = default;
            DNDS_DEVICE_CALLABLE iterator(iterator &&) noexcept = default;
            DNDS_DEVICE_CALLABLE iterator &operator=(iterator &&) noexcept = default;
            DNDS_DEVICE_CALLABLE ~iterator() = default;
            DNDS_DEVICE_CALLABLE iterator(const view_type &n_view, index n_iRow) : view(n_view), t_base_iter(n_iRow)
            {
            }

            DNDS_DEVICE_CALLABLE reference operator*()
            {
                return {view, this->iRow, this->RowSize(this->iRow)};
            }
        };

        template <DeviceBackend B>
        iterator<B> begin()
        {
            return {deviceView<B>(), 0};
        }

        template <DeviceBackend B>
        iterator<B> end()
        {
            return {deviceView<B>(), this->Size()};
        }
    };
}
