#pragma once
/// @file ObjectPool.hpp
/// @brief Pre-allocated object pool with RAII checkout/return semantics.

// #include <unordered_set>
#include "Defines.hpp"

namespace DNDS
{
    /**
     * @brief Generic object pool: caches `unique_ptr<T>` instances and hands them
     * out with RAII return-on-destruction semantics.
     *
     * @details Intended for expensive-to-construct objects used in hot loops
     * (e.g., Eigen solvers, pre-allocated scratch matrices). Typical pattern:
     * ```cpp
     * ObjectPool<SomeWorker> pool;
     * pool.resize(nThreads, ctorArgs...);
     * {
     *   auto handle = pool.get();     // returns from pool, or empty if exhausted
     *   handle->doWork();
     * } // handle goes out of scope -> worker returned to pool
     * ```
     *
     * The pool shares ownership with every outstanding #ObjectPoolAllocated via
     * a `shared_ptr<Pool>`; workers correctly skip the return step if the
     * pool itself has been destroyed meanwhile.
     *
     * @tparam T  Pooled object type.
     */
    template <class T = int>
    class ObjectPool
    {
    public:
        using uPtrResource = std::unique_ptr<T>;
        /// @brief Internal storage shared among all handles.
        struct Pool
        {
            std::vector<uPtrResource> _pool;
            /// @brief Push an object back onto the free-list.
            void recycle(uPtrResource p)
            {
                _pool.emplace_back(std::move(p));
            }
        };
        /// @brief Shared pointer to the underlying storage.
        std::shared_ptr<Pool> pPool;

    public:
        /// @brief Construct an empty pool; populate with #resize.
        ObjectPool()
        {
            pPool = std::make_shared<Pool>();
        }

        /// @brief Number of objects currently available (not checked out).
        size_t size()
        {
            return pPool->_pool.size();
        }

        /**
         * @brief RAII handle returned by #ObjectPool::get (and friends).
         *
         * @details Behaves like a `unique_ptr<T>`; in its destructor it
         * returns the managed object to the pool (or drops it if the pool
         * is already gone).
         */
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
            /// @brief `true` if the handle holds an object.
            operator bool() const { return bool(_ptr); }
            /// @brief Arrow-access to the underlying `unique_ptr` (lets callers
            /// reach through it to `T`).
            uPtrResource &operator->() { return _ptr; }
            ~ObjectPoolAllocated()
            {
                auto poolLocked = pool.lock();
                if (poolLocked && _ptr)
                    poolLocked->_pool.emplace_back(std::move(_ptr));
            }
        };

        /// @brief Pre-allocate `N` objects, forwarding `__ctorArgs` to each ctor.
        /// Clears any previously pooled instances.
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

        /// @brief Like #resize but calls `FInit(obj)` on each newly-created object.
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

        /// @brief Take an object out of the pool. Returns an empty handle if exhausted.
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

        /// @brief Like #get, but allocates a brand-new object if the pool is empty.
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

        /// @brief Like #getAlloc, but additionally runs `FInit(obj)` on newly-allocated objects.
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