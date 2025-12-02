#pragma once

#include "DNDS/Errors.hpp"
#include "DeviceStorage.hpp"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

namespace DNDS
{

    class DeviceHostSingleAllocationBase
    {
    public:
        DeviceHostSingleAllocationBase() = default;
        virtual ~DeviceHostSingleAllocationBase();

        virtual void allocate(size_t bytes, DeviceBackend B = DeviceBackend::Unknown) = 0;
        virtual void free() = 0;
        virtual uint8_t *get() = 0;
        virtual size_t bytes() const = 0;
        virtual DeviceBackend device() = 0;
        virtual void copy_from_host(uint8_t *host_src, size_t n) = 0;
        virtual void copy_to_host(uint8_t *host_dst, size_t n) = 0;
        virtual std::unique_ptr<DeviceHostSingleAllocationBase> clone() = 0;
    };

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
        size_t bytes() const override { return bytes_; }
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

    template <class T, class Derived>
    class data_vector_base
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_default_constructible_v<T>,
                      "data_vector_base elements must be trivially_copyable and default_constructible");

    public:
        T &operator[](size_t i) { return static_cast<Derived *>(this)->data()[i]; }

        const T &operator[](size_t i) const { return static_cast<const Derived *>(this)->data()[i]; }

        const T &at(size_t i) const
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

    template <typename T>
    class host_device_vector : public data_vector_base<T, host_device_vector<T>>
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_default_constructible_v<T>,
                      "host_device_vector elements must be trivially_copyable and default_constructible");

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
        DNDS_HOST host_device_vector() = default;

        DNDS_HOST host_device_vector(size_t n)
        {
            this->resize(n);
        }

        template <class TFill>
        DNDS_HOST host_device_vector(size_t n, TFill &&val)
        {
            this->resize(n, std::forward<TFill>(val));
        }

        // TODO: OPTIMIZE: support RVALUE reference of std::vector<T> HOW?
        DNDS_HOST host_device_vector<T> &operator=(const std::vector<T> &v)
        {
            this->resize(v.size());
            std::copy(v.begin(), v.end(), host_ptr);
            return *this;
        }

        DNDS_HOST host_device_vector(const std::vector<T> &v)
        {
            this->operator=(v);
        }

        DNDS_HOST size_t size() const { return size_; }

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
        DNDS_HOST const T *data() const { return host_ptr; }

        DNDS_HOST T *dataDevice() { return device_ptr; }
        DNDS_HOST const T *dataDevice() const { return device_ptr; }

        DNDS_HOST auto begin() { return host_ptr; }
        DNDS_HOST auto end() { return host_ptr + size_; }
        DNDS_HOST auto begin() const { return host_ptr; }
        DNDS_HOST auto end() const { return host_ptr + size_; }

        DNDS_HOST auto cbegin() const { return host_ptr; }
        DNDS_HOST auto cend() const { return host_ptr + size_; }

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

        host_device_vector &operator=(const host_device_vector<T> &R)
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

        host_device_vector(const host_device_vector<T> &R)
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

        DeviceBackend device()
        {
            return device_data ? device_data->device() : DeviceBackend::Unknown;
        }

        DNDS_HOST void swap(host_device_vector<T> &R) noexcept
        {
            R.device_data.swap(device_data);
            std::swap(R.device_ptr, device_ptr);
            R.host_data.swap(host_data);
            std::swap(R.host_ptr, host_ptr);
            std::swap(R.size_, size_);
        }
    };
}