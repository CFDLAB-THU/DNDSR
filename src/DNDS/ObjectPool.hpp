#pragma once

// #include <unordered_set>
#include "Defines.hpp"

namespace DNDS
{
    // DNDS_SWITCH_INTELLISENSE(
    //     template <class T = int>
    //     ,
    //     using T = int;)
    template <class T = int>
    class ObjectPool
    {
    public:
        using uPtrResource = std::unique_ptr<T>;
        struct Pool
        {
            std::vector<uPtrResource> _pool;
            void recycle(uPtrResource p)
            {
                _pool.emplace_back(std::move(p));
            }
        };
        std::shared_ptr<Pool> pPool;

    public:
        ObjectPool()
        {
            pPool = std::make_shared<Pool>();
        }

        size_t size()
        {
            return pPool->_pool.size();
        }

        class ObjectPoolAllocated
        {
            uPtrResource _ptr;
            std::weak_ptr<Pool> pool;

        public:
            ObjectPoolAllocated(uPtrResource n_ptr, std::shared_ptr<Pool> &pPool)
            {
                DNDS_assert_info(pPool, "the original pool is invalid!");
                pool = pPool;
                _ptr = std::move(n_ptr);
            }
            ObjectPoolAllocated(ObjectPoolAllocated &&R) noexcept
            {
                _ptr = std::move(R._ptr);
                pool = std::move(R.pool);
            }
            void operator=(ObjectPoolAllocated &&R) noexcept
            {
                _ptr = std::move(R._ptr);
                pool = std::move(R.pool);
            }
            T &operator*() { return *_ptr; }
            const T &operator*() const { return *_ptr; }
            operator bool() const { return bool(_ptr); }
            uPtrResource &operator->() { return _ptr; }
            ~ObjectPoolAllocated()
            {
                auto poolLocked = pool.lock();
                if (poolLocked && _ptr)
                    poolLocked->_pool.emplace_back(std::move(_ptr));
            }
        };

        template <class... _CtorArgs>
        void resize(size_t N, _CtorArgs &&...__ctorArgs)
        {
            pPool->_pool.clear();
            pPool->_pool.reserve(N);
            while (pPool->_pool.size() < N)
            {
                uPtrResource p = std::make_unique<T>(std::forward<_CtorArgs>(__ctorArgs)...);
                pPool->_pool.emplace_back(std::move(p));
            }
        }

        template <class TFInit, class... _CtorArgs>
        void resizeInit(size_t N, TFInit &&FInit, _CtorArgs &&...__ctorArgs)
        {
            pPool->_pool.clear();
            pPool->_pool.reserve(N);
            while (pPool->_pool.size() < N)
            {
                uPtrResource p = std::make_unique<T>(std::forward<_CtorArgs>(__ctorArgs)...);
                FInit(*p);
                pPool->_pool.emplace_back(std::move(p));
            }
        }

        ObjectPoolAllocated get()
        {
            if (pPool->_pool.size())
            {
                auto ret = ObjectPoolAllocated(std::move(pPool->_pool.back()), pPool);
                pPool->_pool.pop_back();
                return ret;
            }
            return ObjectPoolAllocated(uPtrResource(), pPool); // empty if no resource left
        }

        template <class... _CtorArgs>
        ObjectPoolAllocated getAlloc(_CtorArgs &&...__ctorArgs)
        {
            if (pPool->_pool.size())
            {
                auto ret = ObjectPoolAllocated(std::move(pPool->_pool.back()), pPool);
                pPool->_pool.pop_back();
                return ret;
            }
            uPtrResource p = std::make_unique<T>(std::forward<_CtorArgs>(__ctorArgs)...);
            return ObjectPoolAllocated(std::move(p), pPool);
        }

        template <class TFInit, class... _CtorArgs>
        ObjectPoolAllocated getAllocInit(TFInit &&FInit, _CtorArgs &&...__ctorArgs)
        {
            if (pPool->_pool.size())
            {
                auto ret = ObjectPoolAllocated(std::move(pPool->_pool.back()), pPool);
                pPool->_pool.pop_back();
                return ret;
            }
            uPtrResource p = std::make_unique<T>(std::forward<_CtorArgs>(__ctorArgs)...);
            FInit(*p);
            return ObjectPoolAllocated(std::move(p), pPool);
        }
    };
}