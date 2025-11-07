#include <cstdint>
#include <cassert>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>

namespace DNDS::cuda_test
{
    using index = int64_t;
    using real = double;
    void test(int arr_size, int v_size)
    {
        index self_father_d_size = arr_size * v_size * 3;
        thrust::device_vector<real> vec(self_father_d_size, 1.0);
        std::cout << vec.data() << std::endl;

        real sqrSum = thrust::transform_reduce(
            thrust::device,
            vec.data(), vec.data() + self_father_d_size,
            thrust::square<real>(),
            0.0,
            thrust::plus<real>());
        std::cout << std::sqrt(sqrSum) << std::endl;
    }
}

int main(int argc, char *argv[])
{
    assert(argc > 2 && "need 2 args: arr_size v_size");

    DNDS::cuda_test::test(std::stoi(argv[1]),
                          std::stoi(argv[2]));

    return 0;
}