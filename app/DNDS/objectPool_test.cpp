#include "DNDS/ObjectPool.hpp"

void testObjectPool()
{
    using namespace DNDS;
    ObjectPool<std::string> stringPool;
    stringPool.resizeInit(2, [](std::string &s)
                          { s = "init by FInit"; });
    std::cout << stringPool.size() << std::endl;
    auto s0 = stringPool.get();
    auto s1 = stringPool.get();
    auto s2 = stringPool.get();
    assert(!s2);
    s2 = stringPool.getAllocInit([](std::string &s)
                                 { s = "Alloc init by FInit"; });
    std::cout << stringPool.size() << std::endl;
    {
        auto s3 = stringPool.getAllocInit([](std::string &s)
                                          { s = "Alloc init by FInit"; });
        ObjectPool<std::string> stringPool1;

        s2 = stringPool1.getAllocInit([](std::string &s)
                                      { s = "Alloc init by FInit, from pool 1"; });
    }
    std::cout << stringPool.size() << std::endl;
    std::cout << *s2 << std::endl;
}

int main()
{
    testObjectPool();
}