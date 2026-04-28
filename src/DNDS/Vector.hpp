#pragma once
/// @file Vector.hpp
/// @brief Host-device vector types with optional GPU storage and device-side views.

#include "DNDS/Errors.hpp"
#include "Device/DeviceStorage.hpp"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

namespace DNDS
{

    /**
     * @brief Abstract single-allocation owning byte buffer.
     *
     * @details A cross-backend "uniquely owned chunk of memory" interface:
     * either a plain `std::vector<uint8_t>` on the host, or a
     * backend-specific @ref DNDS::DeviceStorage "DeviceStorage" on a device. Used as the storage
     * primitive for #host_device_vector_r1.
     */
    class DeviceHostSingleAllocationBase
    {
    public:
        DeviceHostSingleAllocationBase() = default;
        virtual ~DeviceHostSingleAllocationBase();

        // Polymorphic base: delete copy / move to prevent slicing. Concrete
        // subclasses (`DeviceHostSingleAllocationDirect`) own a device buffer;
        // callers manipulate them via `unique_ptr<DeviceHostSingleAllocationBase>`.
        DeviceHostSingleAllocationBase(const DeviceHostSingleAllocationBase &) = delete;
        DeviceHostSingleAllocationBase &operator=(const DeviceHostSingleAllocationBase &) = delete;
        DeviceHostSingleAllocationBase(DeviceHostSingleAllocationBase &&) = delete;
        DeviceHostSingleAllocationBase &operator=(DeviceHostSingleAllocationBase &&) = delete;

        /// @brief Allocate `bytes` on backend `B` (or on the host when @ref Unknown).
        virtual void allocate(size_t bytes, DeviceBackend B = DeviceBackend::Unknown) = 0;
        /// @brief Release the allocation.
        virtual void free() = 0;
        /// @brief Typed byte pointer to the current allocation.
        virtual uint8_t *get() = 0;
        /// @brief Allocation size in bytes.
        [[nodiscard]] virtual size_t bytes() const = 0;
        /// @brief Which backend currently owns the allocation.
        virtual DeviceBackend device() = 0;
        /// @brief Copy `n` bytes from `host_src` into this allocation.
        virtual void copy_from_host(uint8_t *host_src, size_t n) = 0;
        /// @brief Copy `n` bytes from this allocation into `host_dst`.
        virtual void copy_to_host(uint8_t *host_dst, size_t n) = 0;
        /// @brief Deep copy; returns a new allocation containing the same bytes.
        virtual std::unique_ptr<DeviceHostSingleAllocationBase> clone() = 0;
    };

    /**
     * @brief Concrete @ref DNDS::DeviceHostSingleAllocationBase "DeviceHostSingleAllocationBase" using `std::vector<uint8_t>`
     * for host memory and @ref DNDS::DeviceStorage "DeviceStorage" for device memory.
     */
    class DeviceHostSingleAllocationDirect : public DeviceHostSingleAllocationBase
    {
        t_supDeviceStorageBase device_storage;
        std::vector<uint8_t> host_data;
        DeviceBackend B_ = DeviceBackend::Unknown;
        size_t bytes_ = 0;

        using t_self = DeviceHostSingleAllocationDirect;

    public:
        DeviceHostSingleAllocationDirect() = default;

        ~DeviceHostSingleAllocationDirect() override {}

        // Inherited copy / move are `= delete` (polymorphic base owns a
        // device buffer). Re-declare explicitly for the rule-of-five.
        DeviceHostSingleAllocationDirect(const DeviceHostSingleAllocationDirect &) = delete;
        DeviceHostSingleAllocationDirect &operator=(const DeviceHostSingleAllocationDirect &) = delete;
        DeviceHostSingleAllocationDirect(DeviceHostSingleAllocationDirect &&) = delete;
        DeviceHostSingleAllocationDirect &operator=(DeviceHostSingleAllocationDirect &&) = delete;

        void allocate(size_t bytes, DeviceBackend B = DeviceBackend::Unknown) override
        {
            if (bytes_)
                DNDS_check_throw_info(false, "single allocation already allocated");
            bytes_ = bytes;
            B_ = B;
            if (B_ == DeviceBackend::Unknown)
                host_data.resize(bytes_);
            else
                device_storage = device_storage_create(B, bytes_);
        }
        void free() override
        {
            B_ = DeviceBackend::Unknown;
            bytes_ = 0;
            device_storage = nullptr;
            host_data.clear();
        }
        [[nodiscard]] size_t bytes() const override { return bytes_; }
        uint8_t *get() override
        {
            if (B_ == DeviceBackend::Unknown)
                return host_data.data();
            else
            {
                DNDS_check_throw_info(device_storage, "device storage not initialized");
                return device_storage->raw_ptr();
            }
        }
        DeviceBackend device() override { return B_; }

        void copy_to_host(uint8_t *host_dst, size_t n) override
        {
            DNDS_check_throw(n <= bytes());
            if (B_ == DeviceBackend::Unknown)
                std::copy(host_data.begin(), host_data.begin() + n, host_dst);
            else
            {
                DNDS_check_throw_info(device_storage, "device storage not initialized");
                device_storage->copy_device_to_host(host_dst, n);
            }
        }

        void copy_from_host(uint8_t *host_src, size_t n) override
        {
            DNDS_check_throw(n <= bytes());
            if (B_ == DeviceBackend::Unknown)
                std::copy(host_src, host_src + n, host_data.begin());
            else
            {
                DNDS_check_throw_info(device_storage, "device storage not initialized");
                device_storage->copy_host_to_device(host_src, n);
            }
        }

        std::unique_ptr<DeviceHostSingleAllocationBase> clone() override
        {
            auto ret = std::make_unique<t_self>();
            if (!bytes_)
                return std::move(ret);
            ret->allocate(bytes_, B_);
            // if B_ == Host here, actually no allocation
            if (B_ == DeviceBackend::Unknown)
                std::copy(host_data.begin(), host_data.end(), ret->host_data.begin());
            else
            {
                DNDS_check_throw_info(device_storage, "device storage not initialized");
                device_storage->copy_to_device(ret->get(), bytes_);
                // if B_ == Host here, actually no copy
            }
            return std::move(ret);
        }
    };

    /**
     * @brief Non-owning device-callable view `{pointer, size}` over a typed array.
     *
     * @details Analogue of `std::span<T>` that compiles inside `__device__` code.
     * Constant copy semantics (trivially copyable); must not outlive its backing storage.
     */
    template <DeviceBackend B, typename T, typename TSize = int64_t>
    class vector_DeviceView
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_default_constructible_v<T>,
                      "host_device_vector elements must be trivially_copyable and default_constructible");

        T *_data = nullptr;
        TSize _size = 0;

    public:
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(vector_DeviceView, vector_DeviceView)
        DNDS_DEVICE_CALLABLE vector_DeviceView(T *n_data, TSize n_size) : _data(n_data), _size(n_size) {}

        DNDS_DEVICE_CALLABLE T operator[](TSize i) const
        {
            DNDS_HD_assert(i >= 0 && i < _size);
            return _data[i];
        }
        DNDS_DEVICE_CALLABLE T &operator[](TSize i)
        {
            DNDS_HD_assert(i >= 0 && i < _size);
            return _data[i];
        }
        DNDS_DEVICE_CALLABLE TSize size() const { return _size; }
    };

    /**
     * @brief CRTP base offering `operator[]` / `at` on top of a derived's
     * `data()` and `size()` accessors. Used by both #host_device_vector_r1 and
     * (implicitly) #host_device_vector_r0.
     */
    template <class T, class Derived>
    class data_vector_base
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_default_constructible_v<T>,
                      "data_vector_base elements must be trivially_copyable and default_constructible");

    public:
        T &operator[](size_t i) { return static_cast<Derived *>(this)->data()[i]; }

        const T &operator[](size_t i) const { return static_cast<const Derived *>(this)->data()[i]; }

        [[nodiscard]] const T &at(size_t i) const
        {
            auto *dThis = static_cast<const Derived *>(this);
            DNDS_check_throw_info(dThis->size() > i, std::to_string(i) + " --- " + std::to_string(dThis->size()));
            return this->operator[](i);
        }

        T &at(size_t i)
        {
            auto *dThis = static_cast<Derived *>(this);
            DNDS_check_throw_info(dThis->size() > i, std::to_string(i) + " --- " + std::to_string(dThis->size()));
            return this->operator[](i);
        }
    };

    /**
     * @brief Host + optional device vector of trivially copyable `T`.
     *
     * @details Primary storage type used inside @ref DNDS::Array "Array". Always maintains a
     * host copy; on demand, a device mirror can be created via #to_device.
     * Many "vector-like" `std::vector` operations (`resize`, `assign`,
     * `operator[]`) have been reimplemented so the class can be used as a
     * drop-in replacement in DNDSR code paths that need device-awareness.
     *
     * Use the variant #host_device_vector (a thin alias below) to pick up
     * the appropriate specialisation based on element type.
     */
    template <typename T>
    class host_device_vector_r1 : public data_vector_base<T, host_device_vector_r1<T>>
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_default_constructible_v<T>,
                      "host_device_vector elements must be trivially_copyable and default_constructible");
        using t_self = host_device_vector_r1<T>;
        using t_base = data_vector_base<T, host_device_vector_r1<T>>;

        std::unique_ptr<DeviceHostSingleAllocationBase> host_data = std::make_unique<DeviceHostSingleAllocationDirect>();
        std::unique_ptr<DeviceHostSingleAllocationBase> device_data = std::make_unique<DeviceHostSingleAllocationDirect>();
        T *host_ptr = reinterpret_cast<T *>(host_data->get());
        T *device_ptr = reinterpret_cast<T *>(device_data->get());
        size_t size_ = 0;

        void sync_device_ptr()
        {
            device_ptr = reinterpret_cast<T *>(device_data->get());
        }

        void sync_host_ptr()
        {
            host_ptr = reinterpret_cast<T *>(host_data->get());
        }

    public:
        DNDS_HOST host_device_vector_r1() = default;

        DNDS_HOST host_device_vector_r1(size_t n)
        {
            this->resize(n);
        }

        template <class TFill>
        DNDS_HOST host_device_vector_r1(size_t n, TFill &&val)
        {
            this->resize(n, std::forward<TFill>(val));
        }

        // TODO: OPTIMIZE: support RVALUE reference of std::vector<T> HOW?
        DNDS_HOST t_self &operator=(const std::vector<T> &v)
        {
            this->resize(v.size());
            std::copy(v.begin(), v.end(), host_ptr);
            return *this;
        }

        DNDS_HOST host_device_vector_r1(const std::vector<T> &v)
        {
            this->operator=(v);
        }

        DNDS_HOST [[nodiscard]] size_t size() const { return size_; }

        DNDS_HOST void resize(size_t new_size)
        {
            size_ = new_size;
            host_data->free();
            host_data->allocate(size_ * sizeof(T), DeviceBackend::Unknown);
            sync_host_ptr();
        }

        template <class TFill>
        DNDS_HOST void resize(size_t new_size, TFill &&fill)
        {
            this->resize(new_size);
            std::fill(this->begin(), this->end(), std::forward<TFill>(fill));
        }

        DNDS_HOST void create_device_data(DeviceBackend B)
        {
            device_data->free();
            device_data->allocate(size_ * sizeof(T), B);
            sync_device_ptr();
        }

        DNDS_HOST T *data() { return host_ptr; }
        DNDS_HOST [[nodiscard]] const T *data() const { return host_ptr; }

        DNDS_HOST T *dataDevice() { return device_ptr; }
        DNDS_HOST [[nodiscard]] const T *dataDevice() const { return device_ptr; }

        DNDS_HOST auto begin() { return host_ptr; }
        DNDS_HOST auto end() { return host_ptr + size_; }
        DNDS_HOST [[nodiscard]] auto begin() const { return host_ptr; }
        DNDS_HOST [[nodiscard]] auto end() const { return host_ptr + size_; }

        DNDS_HOST [[nodiscard]] auto cbegin() const { return host_ptr; }
        DNDS_HOST [[nodiscard]] auto cend() const { return host_ptr + size_; }

        DNDS_HOST explicit operator std::vector<T>() const
        {
            std::vector<T> ret;
            ret.resize(this->size());
            std::copy(this->begin(), this->end(), ret.begin());
            return ret;
        }

        DNDS_HOST void to_device(DeviceBackend backend = DeviceBackend::Host)
        {
            DNDS_check_throw_info(DeviceBackend::Unknown != backend, "cannot to_device to Unknown");
            DNDS_check_throw_info(device_data, "device_data not initialized");
            if (
                device_data->bytes() != this->size() * sizeof(T) || // size change
                device_data->device() != backend)                   // backend change
                create_device_data(backend);

            device_data->copy_from_host(reinterpret_cast<uint8_t *>(this->data()), this->size() * sizeof(T));
        }

        DNDS_HOST void clear_device()
        {
            DNDS_check_throw_info(device_data, "device_data not initialized");
            device_data->free();
            sync_device_ptr();
        }

        DNDS_HOST void clear()
        {
            clear_device();
            host_data->free();
            sync_host_ptr();
            size_ = 0;
        }

        DNDS_HOST void to_host()
        {
            if (this->device() != DeviceBackend::Unknown)
            {
                DNDS_assert(device_data && host_data && device_data->bytes() == host_data->bytes());
                device_data->copy_to_host(reinterpret_cast<uint8_t *>(this->data()), this->size() * sizeof(T));
            }
        }

        template <DeviceBackend B, typename TSize = int64_t>
        using t_deviceView = vector_DeviceView<B, T, TSize>;
        template <DeviceBackend B, typename TSize = int64_t>
        t_deviceView<B, TSize> deviceView()
        {
            DNDS_assert_info(this->device() == B || (B == DeviceBackend::Host || B == DeviceBackend::Unknown),
                             "not on this device: " + std::string(device_backend_name(B)));
            if constexpr (B == DeviceBackend::Host || B == DeviceBackend::Unknown)
                return t_deviceView<B, TSize>(this->data(), this->size());
            else
                return t_deviceView<B, TSize>(this->dataDevice(), this->size());
        }

        t_self &operator=(const t_self &R)
        {
            if (this == &R)
                return *this;
            this->size_ = R.size();
            this->host_data = R.host_data->clone();
            this->sync_host_ptr();
            this->device_data = R.device_data->clone();
            //! the cloned host "device" has no idea where new data reference is
            //! use to_device to sync it
            if (this->device_data->device() == DeviceBackend::Host)
                this->to_device(DeviceBackend::Host);
            this->sync_device_ptr();
            return *this;
        }

        host_device_vector_r1(const t_self &R)
        {
            this->size_ = R.size();
            this->host_data = R.host_data->clone();
            this->sync_host_ptr();
            this->device_data = R.device_data->clone();
            //! the cloned host "device" has no idea where new data reference is
            //! use to_device to sync it
            if (this->device_data->device() == DeviceBackend::Host)
                this->to_device(DeviceBackend::Host);
            this->sync_device_ptr();
        }

        // Rule-of-five closure. Members are `unique_ptr` + raw-cache
        // pointers + size_t; the default move transfers the unique_ptrs
        // and copies the cache pointers, which is correct because the
        // moved-from object is reset to empty.
        host_device_vector_r1(t_self &&) noexcept = default;
        t_self &operator=(t_self &&) noexcept = default;
        ~host_device_vector_r1() = default;

        DeviceBackend device()
        {
            return device_data ? device_data->device() : DeviceBackend::Unknown;
        }

        DNDS_HOST void swap(t_self &R) noexcept
        {
            R.device_data.swap(device_data);
            std::swap(R.device_ptr, device_ptr);
            R.host_data.swap(host_data);
            std::swap(R.host_ptr, host_ptr);
            std::swap(R.size_, size_);
        }
    };

    /**
     * @brief Legacy `std::vector<T>` subclass with an optional device mirror.
     *
     * @details Simpler predecessor of #host_device_vector_r1, kept for
     * third-party interop that expected a true `std::vector<T>`. Prefer
     * #host_device_vector (aliased to `_r1`) for new code.
     */
    template <typename T>
    struct host_device_vector_r0 : public std::vector<T>
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_default_constructible_v<T>,
                      "host_device_vector elements must be trivially_copyable and default_constructible");
        using t_base = std::vector<T>;
        using t_base::t_base;
        using t_self = host_device_vector_r0<T>;

        t_supDeviceStorageBase deviceStorage = null_supDeviceStorageBase();

        DNDS_HOST host_device_vector_r0(const std::vector<T> &v) : t_base(v) {}

        DNDS_HOST t_self &operator=(const std::vector<T> &v)
        {
            this->t_base::operator=(v);
            return *this;
        }

        // DNDS_HOST explicit operator std::vector<T>() const
        // {
        //     std::vector<T> ret;
        //     ret.resize(this->size());
        //     std::copy(this->begin(), this->end(), ret.begin());
        //     return ret;
        // }

        void to_device(DeviceBackend backend = DeviceBackend::Host)
        {
            DNDS_assert_info(DeviceBackend::Unknown != backend, "cannot to_device to Unknown");
            if (!deviceStorage ||
                deviceStorage->bytes() != this->size() * sizeof(T) || // size change
                deviceStorage->backend() != backend)                  // backend change
                deviceStorage = device_storage_create(backend, this->size() * sizeof(T));
            deviceStorage->copy_host_to_device(reinterpret_cast<uint8_t *>(this->data()), this->size() * sizeof(T));
        }

        void to_host()
        {
            if (this->device() != DeviceBackend::Unknown)
            {
                DNDS_assert(deviceStorage);
                deviceStorage->copy_device_to_host(reinterpret_cast<uint8_t *>(this->data()), this->size() * sizeof(T));
            }
        }

        T *dataDevice()
        {
            return deviceStorage ? reinterpret_cast<T *>(deviceStorage->raw_ptr()) : nullptr;
        }

        template <DeviceBackend B, typename TSize = int64_t>
        using t_deviceView = vector_DeviceView<B, T, TSize>;
        template <DeviceBackend B, typename TSize = int64_t>
        t_deviceView<B, TSize> deviceView()
        {
            DNDS_assert_info(this->device() == B || (B == DeviceBackend::Host || B == DeviceBackend::Unknown),
                             "not on this device: " + std::string(device_backend_name(B)));
            if constexpr (B == DeviceBackend::Host || B == DeviceBackend::Unknown)
                return t_deviceView<B, TSize>(this->data(), this->size());
            else
                return t_deviceView<B, TSize>(this->dataDevice(), this->size());
        }

        const T *dataDevice() const
        {
            return deviceStorage ? reinterpret_cast<const T *>(deviceStorage->raw_ptr()) : nullptr;
        }

        t_self &operator=(const t_self &R)
        {
            if (this == &R)
                return *this;
            this->t_base::operator=(R);
            this->deviceStorage = R.deviceStorage ? device_storage_create(R.deviceStorage->backend(), R.deviceStorage->bytes()) : null_supDeviceStorageBase();
            if (deviceStorage)
            {
                R.deviceStorage->copy_to_device(this->deviceStorage->raw_ptr(), this->deviceStorage->bytes());
                if (deviceStorage->backend() == DeviceBackend::Host)
                    this->to_device(DeviceBackend::Host);
            }
            return *this;
        }

        host_device_vector_r0(const t_self &R) : t_base(R)
        {
            this->deviceStorage = R.deviceStorage ? device_storage_create(R.deviceStorage->backend(), R.deviceStorage->bytes()) : null_supDeviceStorageBase();
            if (deviceStorage)
            {
                R.deviceStorage->copy_to_device(this->deviceStorage->raw_ptr(), this->deviceStorage->bytes());
                if (deviceStorage->backend() == DeviceBackend::Host)
                    this->to_device(DeviceBackend::Host);
            }
        }

        // Rule-of-five closure. `deviceStorage` is a `unique_ptr`; default
        // move transfers it correctly. Dtor is trivial.
        host_device_vector_r0(t_self &&) noexcept = default;
        t_self &operator=(t_self &&) noexcept = default;
        ~host_device_vector_r0() = default;

        DeviceBackend device()
        {
            return deviceStorage ? deviceStorage->backend() : DeviceBackend::Unknown;
        }

        void clear_device()
        {
            deviceStorage.reset();
        }

        void swap(t_self &R) noexcept
        {
            R.swap(*this);
            R.deviceStorage.swap(this->deviceStorage);
        }
    };

    /// @brief Primary public alias: `host_device_vector<T>` = #host_device_vector_r1<T>.
    /// Prefer this name throughout the code base.
    template <class T>
    using host_device_vector = host_device_vector_r1<T>;
}