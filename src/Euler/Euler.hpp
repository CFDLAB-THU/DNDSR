/**
 * @file Euler.hpp
 * @brief Core type definitions, enumerations, and distributed array wrappers for compressible
 *        Navier-Stokes / RANS solvers in the DNDSR CFD framework.
 *
 * This header defines:
 * - @ref EulerModel and @ref RANSModel enumerations for selecting physics models and
 *   turbulence closures.
 * - Compile-time query functions (@ref getnVarsFixed, @ref getDim_Fixed, @ref getGeomDim_Fixed)
 *   that map model enumerants to their variable counts and dimensionality.
 * - @ref EulerModelTraits, a trait struct that bundles all model properties for use in
 *   template-parameterized solver code.
 * - MPI-parallel distributed array wrappers (@ref ArrayDOFV, @ref ArrayRECV, @ref ArrayGRADV)
 *   that extend CFV storage types with element-wise arithmetic, dot products, norms, and
 *   MPI collective reductions. These are the primary data containers for conservative
 *   variables, reconstruction coefficients, and gradients throughout the Euler/RANS solvers.
 * - @ref JacobianValue, a placeholder for implicit-solver Jacobian storage in various formats.
 * - The @ref DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS macro, which declares compile-time
 *   Eigen sequence objects for efficient sub-vector indexing into conservative state vectors.
 */
#pragma once
#include "DNDS/Defines.hpp"
#include "DNDS/Device/DeviceStorage.hpp"
#include "DNDS/EigenUtil.hpp"
#include "DNDS/Config/ConfigEnum.hpp"
#include "CFV/VRDefines.hpp"
#include "DNDS/Errors.hpp"

namespace DNDS::Euler
{
/**
 * @brief Declares compile-time Eigen index sequences for sub-vector access into conservative state vectors.
 *
 * This macro is intended to be expanded inside EulerEvaluator methods (or any templated
 * context where @c dim, @c gDim, and @c I4 are available). It creates `static const`
 * Eigen fixed-size sequences that enable zero-overhead slicing of Eigen vectors representing
 * the conservative variable state \f$\mathbf{U} = [\rho,\; \rho u,\; \rho v,\; \rho w,\; E,\; \ldots]\f$.
 *
 * The naming convention encodes the index range (inclusive on both ends):
 * | Name        | Range (indices)  | Typical use                                      |
 * |-------------|------------------|--------------------------------------------------|
 * | Seq012      | [0, dim-1]       | All spatial indices (0-based), e.g. x,y,z coords |
 * | Seq12       | [1, dim-1]       | Spatial indices excluding the first               |
 * | Seq123      | [1, dim]         | Momentum components (rhoU, rhoV, rhoW)           |
 * | Seq23       | [2, dim]         | Momentum components excluding rhoU                |
 * | Seq234      | [2, dim+1]       | Momentum tail through energy index                |
 * | Seq34       | [3, dim+1]       | Last momentum component(s) and energy             |
 * | Seq01234    | [0, dim+1]       | Full base NS variables (rho through E)            |
 * | SeqG012     | [0, gDim-1]      | Geometry-dimension spatial indices                |
 * | SeqI52Last  | [I4+1, last]     | Turbulence/extension variables after energy       |
 * | I4          | dim+1            | Index of the energy variable in the state vector  |
 *
 * @note Requires @c dim (physics dimension), @c gDim (geometry dimension), and
 *       @c EigenLast (alias for `Eigen::last`) to be in scope.
 */
#define DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS                              \
    static const auto Seq012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim - 1>);   \
    static const auto Seq12 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim - 1>);    \
    static const auto Seq123 = Eigen::seq(Eigen::fix<1>, Eigen::fix<dim>);       \
    static const auto Seq23 = Eigen::seq(Eigen::fix<2>, Eigen::fix<dim>);        \
    static const auto Seq234 = Eigen::seq(Eigen::fix<2>, Eigen::fix<dim + 1>);   \
    static const auto Seq34 = Eigen::seq(Eigen::fix<3>, Eigen::fix<dim + 1>);    \
    static const auto Seq01234 = Eigen::seq(Eigen::fix<0>, Eigen::fix<dim + 1>); \
    static const auto SeqG012 = Eigen::seq(Eigen::fix<0>, Eigen::fix<gDim - 1>); \
    static const auto SeqI52Last = Eigen::seq(Eigen::fix<I4 + 1>, EigenLast);    \
    static const auto I4 = dim + 1;

    /**
     * @brief MPI-parallel distributed array of per-cell Degrees-of-Freedom (conservative variable vectors).
     *
     * Wraps `CFV::tUDof<nVarsFixed>` (which is `ArrayDof<nVarsFixed, 1>`), adding element-wise
     * arithmetic operators, norms, dot products, and MPI-collective reductions that operate
     * across all ranks transparently.
     *
     * Each cell stores a single column vector of @p nVarsFixed conservative variables
     * (e.g. \f$[\rho,\; \rho u,\; \rho v,\; \rho w,\; E]\f$ for a 3D NS model).
     * Instances of this class are used for the solution vector @c u, the solution increment
     * @c uInc, the right-hand-side @c rhs, and other per-cell vector quantities throughout
     * the EulerEvaluator.
     *
     * The "father" sub-array holds locally-owned cells; the "son" sub-array holds ghost
     * cells received from neighboring MPI ranks. Reduction operations (norm, dot, min)
     * only iterate over father cells and then call `MPI_Allreduce` for global results.
     *
     * @tparam nVarsFixed Compile-time number of conservative variables per cell, or
     *                    `Eigen::Dynamic` for runtime-sized vectors.
     *
     * @see ArrayRECV for reconstruction coefficient storage.
     * @see ArrayGRADV for gradient storage.
     */
    template <int nVarsFixed>
    class ArrayDOFV : public CFV::tUDof<nVarsFixed>
    {
    public:
        using t_self = ArrayDOFV<nVarsFixed>;         ///< Self type alias for CRTP-style forwarding.
        using t_base = CFV::tUDof<nVarsFixed>;        ///< Base distributed DOF array type.
        using t_element_mat = typename t_base::t_element_mat; ///< Per-element Eigen matrix/vector type.

        /**
         * @brief Set all DOF entries to a uniform scalar value.
         * @param R The scalar value to assign to every component of every cell.
         */
        void setConstant(real R)
        {
            this->t_base::setConstant(R);
        }
        /**
         * @brief Set all DOF entries to copies of a given vector.
         * @param R Reference vector; each cell's DOF vector is set to this value.
         */
        void setConstant(const Eigen::Ref<const t_element_mat> &R )
        {
            this->t_base::setConstant(R);
        }
        /**
         * @brief Element-wise addition: `this[i] += R[i]` for every cell.
         * @param R Source array with the same distribution layout.
         */
        void operator+=(const t_self &R)
        {
            this->t_base::operator+=(R);
        }
        /**
         * @brief Element-wise subtraction: `this[i] -= R[i]` for every cell.
         * @param R Source array with the same distribution layout.
         */
        void operator-=(const t_self &R)
        {
            this->t_base::operator-=(R);
        }
        /**
         * @brief Uniform scalar multiplication: `this[i] *= R` for every cell.
         * @param R Scalar multiplier applied to all components of all cells.
         */
        void operator*=(real R)
        {
            this->t_base::operator*=(R);
        }
        /**
         * @brief Deep copy assignment from another ArrayDOFV.
         * @param R Source array to copy from. Must have the same distribution layout.
         */
        void operator=(const t_self &R)
        {
            this->t_base::operator=(R);
        }

        /**
         * @brief Scaled addition: `this[i] += r * R[i]` for every cell (axpy-style).
         * @param R Source array to add.
         * @param r Scalar multiplier applied to @p R before addition.
         */
        void addTo(const t_self &R, real r)
        {
            this->t_base::addTo(R, r);
        }

        /**
         * @brief Per-cell scalar multiplication from a vector of cell-wise weights.
         *
         * Multiplies each cell's DOF vector by the corresponding scalar in @p R:
         * `this[i] *= R[i]`. Useful for applying cell-volume or time-step scaling.
         *
         * @param R Vector of per-cell scalar multipliers. Must have size >= number of
         *          locally-owned cells.
         *
         * @note Only operates on host memory. Asserts if device storage is active.
         * @note OpenMP parallelized when `DNDS_DIST_MT_USE_OMP` is defined.
         */
        void operator*=(const std::vector<real> &R)
        {
            DNDS_assert(R.size() >= this->father->Size());
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index i = 0; i < this->father->Size(); i++)
                this->operator[](i) *= R[i];
        }
        /**
         * @brief Component-wise multiplication by a scalar-per-cell ArrayDOFV<1>.
         *
         * Each cell's vector is scaled by the single scalar stored in the
         * corresponding cell of @p R. Disabled via SFINAE when nVarsFixed == 1
         * (to avoid ambiguity with the self-multiplication overload).
         *
         * @tparam nVarsFixed_T SFINAE helper; defaults to @p nVarsFixed.
         * @param R Scalar-per-cell array (ArrayDOFV<1>).
         */
        template <int nVarsFixed_T = nVarsFixed>
        std::enable_if_t<!(nVarsFixed_T == 1)>
        operator*=(const ArrayDOFV<1> &R)
        {
            this->t_base::operator*=(R);
        }

        /**
         * @brief Add a constant vector to every cell: `this[i] += R` for all cells.
         * @param R Vector added uniformly to each cell's DOF.
         */
        void operator+=(const Eigen::Vector<real, nVarsFixed> &R)
        {
            this->t_base::operator+=(R);
        }

        /**
         * @brief Add a scalar to every component of every cell.
         * @param R Scalar value to add uniformly.
         */
        void operator+=(real R)
        {
            this->t_base::operator+=(R);
        }

        /**
         * @brief Component-wise multiplication by a constant vector.
         *
         * Multiplies each cell's DOF vector component-wise by @p R:
         * `this[i] = this[i].cwiseProduct(R)`.
         *
         * @param R Per-variable multiplier vector applied identically to every cell.
         */
        void operator*=(const Eigen::Vector<real, nVarsFixed> &R)
        {
            this->t_base::operator*=(R);
        }

        /**
         * @brief Element-wise (Hadamard) multiplication: `this[i] = this[i] .* R[i]`.
         * @param R Source array; each cell's vector is multiplied component-wise.
         */
        void operator*=(const t_self &R)
        {
            this->t_base::operator*=(R);
        }

        /**
         * @brief Element-wise division: `this[i] = this[i] ./ R[i]`.
         * @param R Divisor array; each cell's vector is divided component-wise.
         */
        void operator/=(const t_self &R)
        {
            this->t_base::operator/=(R);
        }

        /**
         * @brief Compute the global L2 norm across all MPI ranks.
         *
         * Calculates \f$\sqrt{\sum_i \|\mathbf{u}_i\|^2}\f$ over all locally-owned cells,
         * then performs an `MPI_Allreduce(SUM)` to obtain the global result.
         *
         * @return Global L2 norm (same value on all ranks).
         */
        real norm2()
        {
            return this->t_base::norm2();
        }

        /**
         * @brief Compute the per-component global L1 norm.
         *
         * Returns a vector where each component @c j contains
         * \f$\sum_i |u_{i,j}|\f$ summed over all locally-owned cells across all ranks
         * via `MPI_Allreduce(SUM)`.
         *
         * @return Vector of component-wise L1 norms (same on all ranks).
         */
        Eigen::Vector<real, nVarsFixed> componentWiseNorm1()
        {
            return this->t_base::componentWiseNorm1();
        }

        /**
         * @brief Compute the global component-wise minimum across all MPI ranks.
         *
         * For each conservative variable component, finds the minimum value across
         * all locally-owned cells on all ranks. Uses `MPI_Allreduce(MIN)`.
         *
         * Useful for checking physical validity (e.g. minimum density or pressure).
         *
         * @return Vector of per-component global minimums (same on all ranks).
         *
         * @note Uses OpenMP custom reduction (`EigenVecMin`) when OMP is enabled.
         * @note Only iterates over father (locally-owned) cells, not ghost cells.
         */
        Eigen::Vector<real, nVarsFixed> min()
        {
            Eigen::Vector<real, nVarsFixed> minLocal, min;
            minLocal.resize(this->RowSize());
            minLocal.setConstant(veryLargeReal);
            min = minLocal;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp declare reduction(EigenVecMin : Eigen::Vector<real, nVarsFixed> : omp_out = omp_out.array().min(omp_in.array())) initializer(omp_priv = omp_orig)
#    pragma omp parallel for schedule(static) reduction(EigenVecMin : minLocal)
#endif
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                minLocal = minLocal.array().min(this->operator[](i).array());
            MPI::Allreduce(minLocal.data(), min.data(), minLocal.size(), DNDS_MPI_REAL, MPI_MIN, this->father->getMPI().comm);
            return min;
        }

        /**
         * @brief Compute the global dot product `<this, R>` with MPI reduction.
         *
         * Sums \f$\sum_i \mathbf{u}_i \cdot \mathbf{R}_i\f$ over locally-owned cells,
         * then calls `MPI_Allreduce(SUM)`.
         *
         * @param R The other ArrayDOFV operand.
         * @return Global dot product (same value on all ranks).
         */
        real dot(const t_self &R)
        {
            return this->t_base::dot(R);
        }

        /**
         * @brief Compute a weighted global dot product with per-component multipliers.
         *
         * Calculates \f$\sum_i (u_i \circ m_L) \cdot (R_i \circ m_R)\f$ where
         * \f$\circ\f$ denotes component-wise (Hadamard) multiplication by the
         * multiplier arrays. Performs `MPI_Allreduce(SUM)` for the global result.
         *
         * This is useful for preconditioned inner products in iterative solvers
         * where different variable components require different scaling.
         *
         * @tparam TMultL Type of the left multiplier (typically Eigen array expression).
         * @tparam TMultR Type of the right multiplier (typically Eigen array expression).
         * @param R     The other ArrayDOFV operand.
         * @param mL    Per-component multiplier applied to `this` entries.
         * @param mR    Per-component multiplier applied to @p R entries.
         * @return Global weighted dot product (same value on all ranks).
         *
         * @note Only operates on host memory. Asserts if device storage is active.
         * @note OpenMP parallelized when `DNDS_DIST_MT_USE_OMP` is defined.
         */
        template <class TMultL, class TMultR>
        real dot(const t_self &R, TMultL &&mL, TMultR &&mR)
        {
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
            real sqrSum{0}, sqrSumAll;
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static) reduction(+ : sqrSum)
#endif
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                sqrSum += (this->operator[](i).array() * mL).matrix().dot((R.operator[](i).array() * mR).matrix());
            MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, this->father->getMPI().comm);
            return sqrSumAll;
        }

        /**
         * @brief Clamp all components from below: `this[i] = max(this[i], R)`.
         *
         * Applies a component-wise maximum with the given value @p R, ensuring no
         * component drops below @p R. Operates over all cells including ghosts.
         *
         * @tparam TR Type of the lower bound (scalar or array expression broadcastable
         *            by Eigen's `.max()`).
         * @param R   Lower bound value.
         *
         * @note Only operates on host memory. Asserts if device storage is active.
         * @note OpenMP parallelized when `DNDS_DIST_MT_USE_OMP` is defined.
         */
        template <class TR>
        void setMaxWith(TR R)
        {
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array() = this->operator[](i).array().max(R);
        }
    };

    /**
     * @brief MPI-parallel distributed array of per-cell reconstruction coefficient matrices.
     *
     * Wraps `CFV::tURec<nVarsFixed>` (which is `ArrayDof<DynamicSize, nVarsFixed>`), adding
     * element-wise arithmetic, norms, dot products, and MPI-collective reductions.
     *
     * Each cell stores a matrix of shape (nRecBases x nVars), where @c nRecBases is the
     * number of polynomial reconstruction basis functions (determined at runtime by the
     * reconstruction order) and @c nVars is the number of conservative variables. These
     * coefficients define the high-order polynomial representation of the solution within
     * each cell, as used by the Compact Finite Volume variational reconstruction scheme.
     *
     * @tparam nVarsFixed Compile-time number of conservative variables (matrix columns),
     *                    or `Eigen::Dynamic` for runtime-sized.
     *
     * @todo Add more arithmetic operators (see existing TODO in source).
     *
     * @see ArrayDOFV for per-cell DOF vectors.
     * @see ArrayGRADV for per-cell gradient matrices.
     */
    ///@todo://TODO add operators
    template <int nVarsFixed>
    class ArrayRECV : public CFV::tURec<nVarsFixed>
    {
    public:
        using t_self = ArrayRECV<nVarsFixed>;         ///< Self type alias.
        using t_base = CFV::tURec<nVarsFixed>;        ///< Base distributed reconstruction array type.

        /**
         * @brief Set all reconstruction coefficients to a uniform scalar value.
         * @param R Scalar value to fill every matrix entry with.
         */
        void setConstant(real R)
        {
            this->t_base::setConstant(R);
        }
        /**
         * @brief Set all cells' reconstruction matrices to copies of a given matrix.
         *
         * @tparam TR Matrix type compatible with the per-cell storage (must be assignable
         *            to each cell's matrix block).
         * @param R   Reference matrix; each cell's coefficient matrix is set to this value.
         *
         * @note Only operates on host memory. Asserts if device storage is active.
         * @note OpenMP parallelized when `DNDS_DIST_MT_USE_OMP` is defined.
         */
        template <class TR>
        void setConstant(const TR &R)
        {
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) = R;
        }
        /**
         * @brief Element-wise addition: `this[i] += R[i]` for every cell.
         * @param R Source array with the same distribution layout.
         */
        void operator+=(const t_self &R)
        {
            this->t_base::operator+=(R);
        }
        /**
         * @brief Element-wise subtraction: `this[i] -= R[i]` for every cell.
         * @param R Source array with the same distribution layout.
         */
        void operator-=(const t_self &R)
        {
            this->t_base::operator-=(R);
        }
        /**
         * @brief Uniform scalar multiplication: `this[i] *= R` for every cell.
         * @param R Scalar multiplier applied to all entries of all cells.
         */
        void operator*=(real R)
        {
            this->t_base::operator*=(R);
        }
        /**
         * @brief Per-cell scalar multiplication from a vector of cell-wise weights.
         *
         * Multiplies each cell's reconstruction matrix by the corresponding scalar in @p R:
         * `this[i] *= R[i]`. Only operates on father (locally-owned) cells.
         *
         * @param R Vector of per-cell scalar multipliers. Must have size >= number of
         *          locally-owned cells.
         *
         * @note Only operates on host memory. Asserts if device storage is active.
         * @note OpenMP parallelized when `DNDS_DIST_MT_USE_OMP` is defined.
         */
        void operator*=(const std::vector<real> &R)
        {
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
            DNDS_assert(R.size() >= this->father->Size());
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index i = 0; i < this->father->Size(); i++)
                this->operator[](i) *= R[i];
        }
        /**
         * @brief Component-wise multiplication by a scalar-per-cell ArrayDOFV<1>.
         *
         * Each cell's reconstruction matrix is scaled by the single scalar stored in
         * the corresponding cell of @p R.
         *
         * @param R Scalar-per-cell array (ArrayDOFV<1>).
         */
        void operator*=(const ArrayDOFV<1> &R)
        {
            this->t_base::operator*=(R);
        }
        /**
         * @brief Column-wise scaling by a per-variable multiplier row-vector.
         *
         * Each column (variable) of every cell's reconstruction matrix is multiplied by
         * the corresponding element of @p R: `this[i].col(j) *= R(j)`.
         * Useful for applying per-variable scaling (e.g. non-dimensionalization).
         *
         * @param R Row array of per-variable multipliers (1 x nVarsFixed).
         *
         * @note Only operates on host memory. Asserts if device storage is active.
         * @note OpenMP parallelized when `DNDS_DIST_MT_USE_OMP` is defined.
         */
        void operator*=(const Eigen::Array<real, 1, nVarsFixed> &R)
        {
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array().rowwise() *= R;
        }
        /**
         * @brief Deep copy assignment from another ArrayRECV.
         * @param R Source array to copy from. Must have the same distribution layout.
         */
        void operator=(const t_self &R)
        {
            this->t_base::operator=(R);
        }

        /**
         * @brief Column-wise scaled addition: `this[i] += R[i] .* r` (per-variable scaling).
         *
         * For each cell, adds @p R's reconstruction matrix with each column scaled by the
         * corresponding element of @p r: `this[i].col(j) += r(j) * R[i].col(j)`.
         *
         * @param R Source reconstruction array to add.
         * @param r Per-variable (column-wise) scaling factors (1 x nVarsFixed).
         *
         * @note Only operates on host memory. Asserts if device storage is active.
         * @note OpenMP parallelized when `DNDS_DIST_MT_USE_OMP` is defined.
         */
        void addTo(const t_self &R, const Eigen::Array<real, 1, nVarsFixed> &r)
        {
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array() += R.operator[](i).array().rowwise() * r;
        }

        /**
         * @brief Uniform scaled addition: `this[i] += r * R[i]` for every cell (axpy-style).
         * @param R Source reconstruction array to add.
         * @param r Uniform scalar multiplier applied to @p R before addition.
         */
        void addTo(const t_self &R, real r)
        {
            this->t_base::addTo(R, r);
        }

        /**
         * @brief Compute the global L2 norm of all reconstruction coefficients with MPI reduction.
         * @return Global Frobenius-like L2 norm (same value on all ranks).
         */
        real norm2()
        {
            return this->t_base::norm2();
        }

        /**
         * @brief Compute the global dot product `<this, R>` with MPI reduction.
         *
         * Sums the element-wise product of all matrix entries across locally-owned cells,
         * then performs `MPI_Allreduce(SUM)`.
         *
         * @param R The other ArrayRECV operand.
         * @return Global dot product (same value on all ranks).
         */
        real dot(const t_self &R)
        {
            return this->t_base::dot(R);
        }

        /**
         * @brief Compute a per-variable (column-wise) dot product with MPI reduction.
         *
         * Returns a row vector where each component @c j contains
         * \f$\sum_i \sum_k \text{this}[i](k,j) \cdot R[i](k,j)\f$
         * summed over all locally-owned cells and reconstruction bases,
         * then reduced across MPI ranks via `MPI_Allreduce(SUM)`.
         *
         * Useful for monitoring per-variable convergence of reconstruction coefficients.
         *
         * @param R The other ArrayRECV operand.
         * @return Row vector of per-variable dot products (1 x nVarsFixed, same on all ranks).
         *
         * @note Only operates on host memory. Asserts if device storage is active.
         * @note Uses OpenMP custom reduction (`EigenVecSum`) when OMP is enabled.
         * @note Only iterates over father (locally-owned) cells, not ghost cells.
         */
        auto dotV(const t_self &R)
        {
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
            Eigen::RowVector<real, nVarsFixed> sqrSum, sqrSumAll;
            sqrSum.resize(this->father->MatColSize());
            sqrSumAll.resizeLike(sqrSum);
            sqrSum.setZero();
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp declare reduction(EigenVecSum : Eigen::RowVector<real, nVarsFixed> : omp_out += omp_in) initializer(omp_priv = omp_orig)
#    pragma omp parallel for schedule(static) reduction(EigenVecSum : sqrSum)
#endif
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                sqrSum += (this->operator[](i).array() * R.operator[](i).array()).colwise().sum().matrix();
            MPI::Allreduce(sqrSum.data(), sqrSumAll.data(), sqrSum.size(), DNDS_MPI_REAL, MPI_SUM, this->father->getMPI().comm);
            return sqrSumAll;
        }
    };

    /**
     * @brief MPI-parallel distributed array of per-cell gradient matrices.
     *
     * Wraps `CFV::tUGrad<nVarsFixed, gDim>` (which is `ArrayDof<gDim, nVarsFixed>`), adding
     * element-wise arithmetic operators for gradient manipulation.
     *
     * Each cell stores a (gDim x nVars) matrix where row @c d contains the partial derivative
     * of all conservative variables with respect to spatial coordinate @c d:
     * \f[
     *   \mathbf{G}_i = \begin{bmatrix}
     *     \partial u_1/\partial x_1 & \cdots & \partial u_{nVars}/\partial x_1 \\
     *     \vdots & \ddots & \vdots \\
     *     \partial u_1/\partial x_{gDim} & \cdots & \partial u_{nVars}/\partial x_{gDim}
     *   \end{bmatrix}
     * \f]
     *
     * @tparam nVarsFixed Compile-time number of conservative variables (matrix columns),
     *                    or `Eigen::Dynamic` for runtime-sized.
     * @tparam gDim       Geometry (mesh) spatial dimension (2 or 3), determining the
     *                    number of gradient rows.
     *
     * @see ArrayDOFV for per-cell DOF vectors.
     * @see ArrayRECV for reconstruction coefficient storage.
     */
    template <int nVarsFixed, int gDim>
    class ArrayGRADV : public CFV::tUGrad<nVarsFixed, gDim>
    {
    public:
        using t_self = ArrayGRADV<nVarsFixed, gDim>;          ///< Self type alias.
        using t_base = CFV::tUGrad<nVarsFixed, gDim>;         ///< Base distributed gradient array type.

        /**
         * @brief Set all gradient entries to a uniform scalar value.
         * @param R Scalar value to fill every matrix entry with.
         */
        void setConstant(real R)
        {
            this->t_base::setConstant(R);
        }
        /**
         * @brief Set all cells' gradient matrices to copies of a given matrix.
         *
         * @tparam TR Matrix type compatible with per-cell gradient storage.
         * @param R   Reference matrix; each cell's gradient is set to this value.
         *
         * @note Only operates on host memory. Asserts if device storage is active.
         * @note OpenMP parallelized when `DNDS_DIST_MT_USE_OMP` is defined.
         */
        template <class TR>
        void setConstant(const TR &R)
        {
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) = R;
        }

        /**
         * @brief Element-wise addition: `this[i] += R[i]` for every cell.
         * @param R Source gradient array with the same distribution layout.
         */
        void operator+=(t_self &R)
        {
            this->t_base::operator+=(R);
        }
        /**
         * @brief Element-wise subtraction: `this[i] -= R[i]` for every cell.
         * @param R Source gradient array with the same distribution layout.
         */
        void operator-=(t_self &R)
        {
            this->t_base::operator-=(R);
        }
        /**
         * @brief Uniform scalar multiplication: `this[i] *= R` for every cell.
         * @param R Scalar multiplier applied to all entries of all cells.
         */
        void operator*=(real R)
        {
            this->t_base::operator*=(R);
        }
        /**
         * @brief Per-cell scalar multiplication from a vector of cell-wise weights.
         *
         * Multiplies each cell's gradient matrix by the corresponding scalar in @p R:
         * `this[i] *= R[i]`. Only operates on father (locally-owned) cells.
         *
         * @param R Vector of per-cell scalar multipliers. Must have size >= number of
         *          locally-owned cells.
         *
         * @note OpenMP parallelized when `DNDS_DIST_MT_USE_OMP` is defined.
         */
        void operator*=(std::vector<real> &R)
        {
            DNDS_assert(R.size() >= this->father->Size());
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index i = 0; i < this->father->Size(); i++)
                this->operator[](i) *= R[i];
        }
        /**
         * @brief Component-wise multiplication by a scalar-per-cell ArrayDOFV<1>.
         *
         * Each cell's gradient matrix is scaled by the single scalar stored in
         * the corresponding cell of @p R.
         *
         * @param R Scalar-per-cell array (ArrayDOFV<1>).
         */
        void operator*=(ArrayDOFV<1> &R)
        {
            this->t_base::operator*=(R);
        }
        /**
         * @brief Column-wise scaling by a per-variable multiplier row-vector.
         *
         * Each column (variable) of every cell's gradient matrix is multiplied by
         * the corresponding element of @p R: `this[i].col(j) *= R(j)`.
         *
         * @param R Row array of per-variable multipliers (1 x nVarsFixed).
         *
         * @note Only operates on host memory. Asserts if device storage is active.
         * @note OpenMP parallelized when `DNDS_DIST_MT_USE_OMP` is defined.
         */
        void operator*=(const Eigen::Array<real, 1, nVarsFixed> &R)
        {
            DNDS_assert(this->father->device() == DeviceBackend::Host ||
                        this->father->device() == DeviceBackend::Unknown);
#if defined(DNDS_DIST_MT_USE_OMP)
#    pragma omp parallel for schedule(static)
#endif
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array().rowwise() *= R;
        }
        /**
         * @brief Deep copy assignment from another ArrayGRADV.
         * @param R Source array to copy from. Must have the same distribution layout.
         */
        void operator=(t_self &R)
        {
            this->t_base::operator=(R);
        }
    };

    /**
     * @brief Placeholder container for implicit-solver Jacobian matrix storage.
     *
     * Supports three storage modes for the flux Jacobian \f$\partial \mathbf{R}/\partial \mathbf{U}\f$:
     * - **Diagonal:** One scalar per variable per cell (cheapest, first-order implicit).
     * - **DiagonalBlock:** One dense (nVars x nVars) block per cell (block-Jacobi preconditioner).
     * - **Full:** Sparse block structure following cell-to-cell adjacency (full implicit).
     *
     * @warning This class is currently a stub. The `Set*` and `InverseDiag` methods contain
     *          only allocation placeholders (marked with `// todo`). Actual Jacobian assembly
     *          and inversion are not yet implemented.
     *
     * @tparam nVarsFixed Compile-time number of conservative variables, determining the
     *                    block size for DiagonalBlock and Full modes.
     */
    template <int nVarsFixed>
    class JacobianValue
    {
    public:
        /**
         * @brief Jacobian storage mode selector.
         */
        enum Type
        {
            Diagonal = 0,       ///< One scalar per variable per cell (diagonal approximation).
            DiagonalBlock = 1,  ///< One (nVars x nVars) dense block per cell.
            Full = 2,           ///< Sparse block structure following cell adjacency graph.
        };
        ArrayDOFV<nVarsFixed> diag;                                 ///< Diagonal Jacobian values (one vector per cell).
        ArrayDOFV<nVarsFixed> diagInv;                              ///< Inverse of diagonal Jacobian values.
        ArrayEigenMatrix<nVarsFixed, nVarsFixed> diagBlock;         ///< Block-diagonal Jacobian (one nVars x nVars matrix per cell).
        ArrayEigenMatrix<nVarsFixed, nVarsFixed> diagBlockInv;      ///< Inverse of block-diagonal Jacobian.
        ArrayRECV<nVarsFixed> offDiagBlock;                         ///< Off-diagonal blocks for full Jacobian (sparse, adjacency-based).

        /**
         * @brief Initialize storage for diagonal (scalar-per-variable) Jacobian mode.
         * @param uDof Reference DOF array whose distribution layout is used for allocation.
         * @todo Implement actual allocation of diagonal storage.
         */
        void SetDiagonal(ArrayDOFV<nVarsFixed> &uDof)
        {
            type = Diagonal;
            // todo ! allocate square blocks!
        }

        /**
         * @brief Initialize storage for block-diagonal (dense block-per-cell) Jacobian mode.
         * @param uDof Reference DOF array whose distribution layout is used for allocation.
         * @todo Implement actual allocation of (nVars x nVars) block storage.
         */
        void SetDiagonalBlock(ArrayDOFV<nVarsFixed> &uDof)
        {
            type = DiagonalBlock;
            // todo ! allocate square blocks!
        }

        /**
         * @brief Initialize storage for the full sparse block Jacobian following cell adjacency.
         * @param uDof       Reference DOF array whose distribution layout is used for allocation.
         * @param cell2cell  Cell-to-cell adjacency pair defining the sparsity pattern.
         * @todo Implement actual allocation using the adjacency graph.
         */
        void SetFull(ArrayDOFV<nVarsFixed> &uDof, Geom::tAdjPair &cell2cell)
        {
            type = Full;
            // todo ! allocate with adjacency!
        }

        /**
         * @brief Compute the inverse of the diagonal (or block-diagonal) Jacobian.
         *
         * Stores the result in @ref diagInv (Diagonal mode) or @ref diagBlockInv
         * (DiagonalBlock mode) for use as a preconditioner in iterative solvers.
         *
         * @todo Implement actual inversion logic.
         */
        void InverseDiag()
        {
            // todo get inverse!
        }

    private:
        Type type = Diagonal; ///< Currently active Jacobian storage mode.
    };

    /**
     * @brief Enumerates the available compressible flow solver model configurations.
     *
     * Each enumerant selects a combination of:
     * - **Physics dimension** (`dim`): number of velocity components in the equations.
     *   All models use `dim=3` (three velocity components) except `NS_2D` which uses `dim=2`.
     * - **Geometry dimension** (`gDim`): mesh spatial dimension (2D or 3D).
     *   Models without `_3D` suffix use `gDim=2`; those with `_3D` use `gDim=3`.
     * - **Turbulence model**: plain NS (no turbulence transport), Spalart-Allmaras (`SA`, +1 var),
     *   two-equation RANS (`2EQ`, +2 vars), or extended/dynamic (`EX`, runtime-determined).
     *
     * The compile-time number of conservative variables (`nVarsFixed`) follows from the model:
     * | Model       | nVarsFixed | Variables                                            |
     * |-------------|------------|------------------------------------------------------|
     * | NS, NS_3D   | 5          | \f$\rho,\; \rho u,\; \rho v,\; \rho w,\; E\f$      |
     * | NS_2D       | 4          | \f$\rho,\; \rho u,\; \rho v,\; E\f$                 |
     * | NS_SA, NS_SA_3D | 6      | NS + \f$\rho\tilde{\nu}\f$                           |
     * | NS_2EQ, NS_2EQ_3D | 7    | NS + \f$\rho k,\; \rho\omega\f$                     |
     * | NS_EX, NS_EX_3D | Dynamic | Runtime-determined (Eigen::Dynamic)                 |
     *
     * @see getnVarsFixed() for the nVarsFixed mapping.
     * @see getDim_Fixed() for the physics dimension mapping.
     * @see getGeomDim_Fixed() for the geometry dimension mapping.
     * @see EulerModelTraits for a compile-time trait bundle.
     */
    enum EulerModel
    {
        NS = 0,         ///< Navier-Stokes, 2D geometry, 3D physics (5 vars).
        NS_SA = 1,      ///< NS + Spalart-Allmaras, 2D geometry (6 vars).
        NS_2D = 2,      ///< NS with 2D physics (2 velocity components, 4 vars). The only model with dim=2.
        NS_3D = 3,      ///< Navier-Stokes, 3D geometry, 3D physics (5 vars).
        NS_SA_3D = 4,   ///< NS + Spalart-Allmaras, 3D geometry (6 vars).
        NS_2EQ = 5,     ///< NS + 2-equation RANS, 2D geometry (7 vars).
        NS_2EQ_3D = 6,  ///< NS + 2-equation RANS, 3D geometry (7 vars).
        NS_EX = 101,    ///< Extended NS, 2D geometry, dynamic nVars (Eigen::Dynamic).
        NS_EX_3D = 102, ///< Extended NS, 3D geometry, dynamic nVars (Eigen::Dynamic).
    };

    /**
     * @brief Enumerates the available RANS turbulence closure models.
     *
     * Selects which turbulence transport equations are solved at runtime. This is orthogonal
     * to the compile-time @ref EulerModel selection: `EulerModel` determines the number of
     * transport variables, while `RANSModel` selects the specific closure (e.g. which
     * two-equation model to use when `EulerModel` is `NS_2EQ` or `NS_2EQ_3D`).
     *
     * JSON serialization is provided via @c DNDS_DEFINE_ENUM_JSON.
     */
    enum RANSModel
    {
        RANS_Unknown = 0,  ///< Sentinel / uninitialized value.
        RANS_None,         ///< No turbulence model (laminar or DNS).
        RANS_SA,           ///< Spalart-Allmaras one-equation model.
        RANS_KOWilcox,     ///< Wilcox k-omega two-equation model.
        RANS_KOSST,        ///< Menter k-omega SST two-equation model.
        RANS_RKE,          ///< Realizable k-epsilon two-equation model.
    };

    DNDS_DEFINE_ENUM_JSON(
        RANSModel,
        {
            {RANS_Unknown, nullptr},
            {RANS_None, "RANS_None"},
            {RANS_SA, "RANS_SA"},
            {RANS_KOWilcox, "RANS_KOWilcox"},
            {RANS_KOSST, "RANS_KOSST"},
            {RANS_RKE, "RANS_RKE"},
        })

    /**
     * @brief Returns the compile-time (fixed) number of conservative variables for a given model.
     *
     * This is a `constexpr` function suitable for use as a template argument. The mapping is:
     * - NS, NS_3D: 5 (\f$\rho, \rho u, \rho v, \rho w, E\f$)
     * - NS_SA, NS_SA_3D: 6 (+ \f$\rho\tilde{\nu}\f$)
     * - NS_2D: 4 (\f$\rho, \rho u, \rho v, E\f$)
     * - NS_2EQ, NS_2EQ_3D: 7 (+ \f$\rho k, \rho\omega\f$)
     * - NS_EX, NS_EX_3D: `Eigen::Dynamic` (determined at runtime)
     *
     * @param model The Euler model enumerant.
     * @return Compile-time variable count, or `Eigen::Dynamic` (-1) for extended models.
     *
     * @see getNVars() for a version that always returns a positive runtime value.
     */
    constexpr static inline int getnVarsFixed(const EulerModel model)
    {
        if (model == NS || model == NS_3D)
            return 5;
        else if (model == NS_SA)
            return 6;
        else if (model == NS_SA_3D)
            return 6;
        else if (model == NS_2D)
            return 4;
        else if (model == NS_2EQ || model == NS_2EQ_3D)
            return 7;
        return Eigen::Dynamic;
    }

    /**
     * @brief Returns the runtime number of conservative variables for a given model.
     *
     * Unlike @ref getnVarsFixed(), this function always returns a positive integer.
     * For fixed-size models it returns the same value as `getnVarsFixed()`. For
     * extended models (`NS_EX`, `NS_EX_3D`) where `getnVarsFixed()` returns
     * `Eigen::Dynamic`, this function falls through to a secondary lookup that
     * returns the base variable count (though extended models may require additional
     * runtime configuration to determine the true count).
     *
     * @param model The Euler model enumerant.
     * @return Positive number of conservative variables.
     *
     * @note For extended models, the returned value is a base count and may need
     *       to be augmented by runtime configuration (e.g. number of species).
     */
    constexpr static inline int getNVars(EulerModel model)
    {
        int nVars = getnVarsFixed(model);
        if (nVars < 0)
        {
            if (model == NS || model == NS_3D)
                return 5;
            else if (model == NS_SA)
                return 6;
            else if (model == NS_SA_3D)
                return 6;
            else if (model == NS_2D)
                return 4;
            else if (model == NS_2EQ || model == NS_2EQ_3D)
                return 7;
            // *** handle variable nVars
        }
        return nVars;
    }

    /**
     * @brief Returns the physics spatial dimension for a given model at compile time.
     *
     * The physics dimension determines how many velocity components appear in the
     * governing equations:
     * - `NS_2D`: returns 2 (only \f$u, v\f$ velocity components).
     * - All other models: returns 3 (full \f$u, v, w\f$ velocity components),
     *   even for 2D-geometry models where the mesh is two-dimensional.
     *
     * @param model The Euler model enumerant.
     * @return Physics dimension (2 or 3), or `Eigen::Dynamic` if unrecognized.
     *
     * @see getGeomDim_Fixed() for the mesh/geometry dimension.
     */
    constexpr static inline int getDim_Fixed(const EulerModel model)
    {
        if (model == NS || model == NS_3D)
            return 3;
        else if (model == NS_SA)
            return 3;
        else if (model == NS_SA_3D)
            return 3;
        else if (model == NS_2D)
            return 2;
        else if (model == NS_2EQ || model == NS_2EQ_3D)
            return 3;
        else if (model == NS_EX || model == NS_EX_3D)
            return 3;
        return Eigen::Dynamic;
    }

    /**
     * @brief Returns the geometry (mesh) spatial dimension for a given model at compile time.
     *
     * The geometry dimension determines whether the mesh is two-dimensional or
     * three-dimensional:
     * - 2D geometry (`gDim=2`): NS, NS_SA, NS_2D, NS_2EQ, NS_EX.
     * - 3D geometry (`gDim=3`): NS_3D, NS_SA_3D, NS_2EQ_3D, NS_EX_3D.
     *
     * Note that `gDim` differs from the physics dimension (`dim`). For example,
     * `NS` has `gDim=2` (2D mesh) but `dim=3` (3 velocity components in the equations).
     *
     * @param model The Euler model enumerant.
     * @return Geometry dimension (2 or 3), or `Eigen::Dynamic` if unrecognized.
     *
     * @see getDim_Fixed() for the physics dimension.
     */
    constexpr static inline int getGeomDim_Fixed(const EulerModel model)
    {
        if (model == NS)
            return 2;
        else if (model == NS_SA)
            return 2;
        else if (model == NS_2D)
            return 2;
        else if (model == NS_3D)
            return 3;
        else if (model == NS_SA_3D)
            return 3;
        else if (model == NS_2EQ)
            return 2;
        else if (model == NS_2EQ_3D)
            return 3;
        else if (model == NS_EX)
            return 2;
        else if (model == NS_EX_3D)
            return 3;
        return Eigen::Dynamic;
    }

    // constexpr static inline bool ifFixedNvars(EulerModel model)
    // {
    //     return (
    //         model == NS ||
    //         model == NS_SA);
    // } // use +/- is ok

    /**
     * @brief Compile-time traits for EulerModel variants.
     *
     * Bundles all model-dependent compile-time constants (variable count, dimensions,
     * feature flags) into a single struct, replacing scattered `if constexpr` chains
     * with clean named traits. All members are `static constexpr`.
     *
     * Usage example:
     * @code
     * using Traits = EulerModelTraits<NS_SA_3D>;
     * static_assert(Traits::nVarsFixed == 6);
     * static_assert(Traits::dim == 3);
     * static_assert(Traits::gDim == 3);
     * static_assert(Traits::hasSA);
     * static_assert(Traits::hasRANS);
     * static_assert(Traits::nRANSVars == 1);
     * @endcode
     *
     * Future models (reactive, multi-species) should add traits here rather than
     * adding new if-constexpr chains.
     *
     * @tparam model The EulerModel enumerant to query traits for.
     */
    template <EulerModel model>
    struct EulerModelTraits
    {
        /// Number of fixed conservative variables (Eigen::Dynamic for NS_EX).
        static constexpr int nVarsFixed = getnVarsFixed(model);
        /// Physics spatial dimension (2 for NS_2D, 3 for all others).
        static constexpr int dim = getDim_Fixed(model);
        /// Geometry (mesh) spatial dimension.
        static constexpr int gDim = getGeomDim_Fixed(model);

        /// True for Spalart-Allmaras models (NS_SA, NS_SA_3D).
        static constexpr bool hasSA = (model == NS_SA || model == NS_SA_3D);
        /// True for 2-equation RANS models (NS_2EQ, NS_2EQ_3D).
        static constexpr bool has2EQ = (model == NS_2EQ || model == NS_2EQ_3D);
        /// True for any RANS turbulence model (SA or 2-equation).
        static constexpr bool hasRANS = hasSA || has2EQ;
        /// Number of extra RANS transport variables (0, 1, or 2).
        static constexpr int nRANSVars = hasSA ? 1 : (has2EQ ? 2 : 0);

        /// True for extended/dynamic models (NS_EX, NS_EX_3D).
        static constexpr bool isExtended = (model == NS_EX || model == NS_EX_3D);
        /// True for plain NS without turbulence or extensions.
        static constexpr bool isPlainNS = !hasRANS && !isExtended;
        /// True for 2D geometry models.
        static constexpr bool isGeom2D = (gDim == 2);
        /// True for 3D geometry models.
        static constexpr bool isGeom3D = (gDim == 3);
    };

    /**
     * @brief Compile-time multiplication that propagates `Eigen::Dynamic`.
     *
     * Returns `nVarsFixed * mul` when @p nVarsFixed is a positive compile-time constant.
     * If @p nVarsFixed equals `Eigen::Dynamic` (i.e. -1), the result is `Eigen::Dynamic`,
     * preserving the dynamic-size sentinel through arithmetic. This is needed when
     * computing derived template sizes (e.g. Jacobian block dimensions) from `nVarsFixed`.
     *
     * @tparam nVarsFixed Base compile-time size, or `Eigen::Dynamic`.
     * @tparam mul        Positive integer multiplier.
     * @return `nVarsFixed * mul`, or `Eigen::Dynamic` if input is dynamic.
     */
    template <int nVarsFixed, int mul>
    constexpr static inline int nvarsFixedMultiply()
    {
        return nVarsFixed != Eigen::Dynamic ? nVarsFixed * mul : Eigen::Dynamic;
    }
}