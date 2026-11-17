#include "DNDS/Runtime/Execution/CUDAStream.hpp"
#include "DNDS/Algebra/BatchedSVD.hpp"
#include <climits>
#include <cstdlib>
#include <limits>
#include <mutex>
#ifdef DNDS_USE_CUDA
#    include <cuda_runtime_api.h>
#    include <cusolverDn.h>
#endif

namespace DNDS::Algebra
{
    namespace R = Runtime;
    namespace
    {
        template <class T>
        void CheckTable(R::ArrayView<T> view, R::Device device, index rows, index width)
        {
            view.ValidateDevice(device);
            DNDS_check_throw_info(view.kind == R::RowLayoutKind::Table && view.RowCount() == rows &&
                                      view.Width() == width && view.stride == width && !view.offsets && !view.sizes &&
                                      (!rows || view.data),
                                  "batched SVD requires compact contiguous tables");
        }
        void CheckViews(BatchedSVDViews views, R::Device device, index rows, int dimension)
        {
            CheckTable(views.matrix, device, rows, dimension * dimension);
            CheckTable(views.singularValues, device, rows, dimension);
            CheckTable(views.leftVectors, device, rows, dimension * dimension);
            CheckTable(views.rightVectors, device, rows, dimension * dimension);
            if (rows)
            {
                const real *addresses[]{views.matrix.data, views.singularValues.data, views.leftVectors.data, views.rightVectors.data};
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < i; ++j)
                        DNDS_check_throw_info(addresses[i] != addresses[j], "batched SVD operands alias");
            }
        }
#ifdef DNDS_USE_CUDA
        void Check(cusolverStatus_t status)
        {
            DNDS_check_throw_info(status == CUSOLVER_STATUS_SUCCESS, "cuSOLVER batched SVD host API failed: " + std::to_string(int(status)));
        }
        struct DeviceScope
        {
            int previous = -1;
            explicit DeviceScope(int ordinal)
            {
                DNDS_check_throw_info(cudaGetDevice(&previous) == cudaSuccess, "batched SVD cannot query CUDA device");
                DNDS_check_throw_info(cudaSetDevice(ordinal) == cudaSuccess, "batched SVD cannot select CUDA device");
            }
            ~DeviceScope()
            {
                if (cudaSetDevice(previous) != cudaSuccess)
                    std::abort();
            }
        };
#endif
    }
    struct BatchedSVD::Data
    {
        R::ExecutionContext context;
        int dimension = 0, rows = 0, workspace = 0;
        std::mutex submission;
#ifdef DNDS_USE_CUDA
        cusolverDnHandle_t handle = nullptr;
        gesvdjInfo_t policy = nullptr;
        ~Data()
        {
            if (!handle && !policy)
                return;
            int previous = -1;
            if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(context.device.ordinal) != cudaSuccess)
                std::abort();
            if (policy && cusolverDnDestroyGesvdjInfo(policy) != CUSOLVER_STATUS_SUCCESS)
                std::abort();
            if (handle && cusolverDnDestroy(handle) != CUSOLVER_STATUS_SUCCESS)
                std::abort();
            if (cudaSetDevice(previous) != cudaSuccess)
                std::abort();
        }
#endif
    };
    BatchedSVD::BatchedSVD(R::ExecutionContext context, int dimension, BatchedSVDViews prototype)
        : _data(std::make_unique<Data>())
    {
        DNDS_check_throw_info(context.device.space == R::MemorySpace::CUDA && context.device.IsValid() &&
                                  dimension > 0 && dimension <= 32 && prototype.matrix.RowCount() >= 0 &&
                                  prototype.matrix.RowCount() <= INT_MAX,
                              "invalid CUDA batched SVD dimensions/device");
        CheckViews(prototype, context.device, prototype.matrix.RowCount(), dimension);
        _data->context = context;
        _data->dimension = dimension;
        _data->rows = int(prototype.matrix.RowCount());
#ifdef DNDS_USE_CUDA
        if (!_data->rows)
            return;
        DeviceScope device(context.device.ordinal);
        Check(cusolverDnCreate(&_data->handle));
        Check(cusolverDnSetStream(_data->handle, ::DNDS::Runtime::CUDAStream(context)));
        Check(cusolverDnCreateGesvdjInfo(&_data->policy));
        Check(cusolverDnXgesvdjSetTolerance(_data->policy, std::numeric_limits<real>::epsilon()));
        Check(cusolverDnXgesvdjSetMaxSweeps(_data->policy, 100));
        Check(cusolverDnXgesvdjSetSortEig(_data->policy, 1));
        Check(cusolverDnDgesvdjBatched_bufferSize(_data->handle, CUSOLVER_EIG_MODE_VECTOR,
                                                  dimension, dimension, prototype.matrix.data, dimension, prototype.singularValues.data,
                                                  prototype.leftVectors.data, dimension, prototype.rightVectors.data, dimension,
                                                  &_data->workspace, _data->policy, _data->rows));
        DNDS_check_throw_info(_data->workspace >= 0, "cuSOLVER returned invalid workspace size");
#else
        DNDS_check_throw_info(false, "CUDA batched SVD was not compiled");
#endif
    }
    BatchedSVD::~BatchedSVD() = default;
    index BatchedSVD::WorkspaceElements() const { return _data->workspace; }
    void BatchedSVD::SubmitBorrowed(R::ExecutionContext context, BatchedSVDViews values,
                                    R::ArrayView<real> workspace, R::ArrayView<int> info)
    {
        DNDS_check_throw_info(context.device == _data->context.device && context.stream == _data->context.stream,
                              "batched SVD belongs to another execution stream");
        CheckViews(values, context.device, _data->rows, _data->dimension);
        CheckTable(workspace, context.device, _data->workspace, 1);
        CheckTable(info, context.device, _data->rows, 1);
        DNDS_check_throw_info(!workspace.data || (workspace.data != values.matrix.data && workspace.data != values.singularValues.data &&
                                                  workspace.data != values.leftVectors.data && workspace.data != values.rightVectors.data),
                              "batched SVD workspace aliases values");
#ifdef DNDS_USE_CUDA
        if (!_data->rows)
            return;
        std::lock_guard<std::mutex> lock(_data->submission);
        DeviceScope device(context.device.ordinal);
        const int n = _data->dimension;
        Check(cusolverDnDgesvdjBatched(_data->handle, CUSOLVER_EIG_MODE_VECTOR, n, n,
                                       values.matrix.data, n, values.singularValues.data, values.leftVectors.data, n,
                                       values.rightVectors.data, n, workspace.data, _data->workspace, info.data, _data->policy, _data->rows));
#else
        DNDS_check_throw_info(false, "CUDA batched SVD was not compiled");
#endif
    }
}
