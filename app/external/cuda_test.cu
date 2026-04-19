#include <cstdint>
#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <cuda_runtime.h>

namespace DNDS::cuda_test
{
    using index = int64_t;
    using real = double;
    void test(int arr_size, int v_size)
    {

        index self_father_d_size = arr_size * v_size * 3;
        thrust::device_vector<real> vec(self_father_d_size, 1.0);
        std::cout << vec.data() << std::endl;
        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                exit(1);
            }
        }
        real sqrSum = thrust::transform_reduce(
            thrust::device,
            vec.data(), vec.data() + self_father_d_size,
            thrust::square<real>(),
            0.0,
            thrust::plus<real>());
        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                exit(1);
            }
        }
        std::cout << std::sqrt(sqrSum) << std::endl;
        assert(std::abs(sqrSum - double(self_father_d_size)) < 1e-16 * self_father_d_size);
    }

    void test_1(int arr_size, int v_size)
    {
        index self_father_d_size = arr_size * v_size;
        // index self_father_d_size = 256 * 1024;
        thrust::device_vector<real> vec(self_father_d_size, 1.0);
        std::cout << vec.data() << std::endl;
        cudaError_t err = cudaGetLastError();
        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                exit(1);
            }
        }

        real sqrSum = thrust::reduce(vec.data(), vec.data() + self_father_d_size, 1e100, [] __device__ __host__(real a, real b)
                                     { return a < b ? a : b; });
        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
                exit(1);
            }
        }
        std::cout << sqrSum << std::endl;
        assert(std::abs(sqrSum - double(1)) < 1e-16 * self_father_d_size);
    }
}

int main(int argc, char *argv[])
{
    assert(argc > 2 && "need 2 args: arr_size v_size");

    DNDS::cuda_test::test(std::stoi(argv[1]),
                          std::stoi(argv[2]));
    DNDS::cuda_test::test_1(std::stoi(argv[1]),
                            std::stoi(argv[2]));
    return 0;
}