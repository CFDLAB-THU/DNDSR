#pragma once

#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Runtime/Storage/ElementSchema.hpp"
#include "DNDS/Runtime/Array/RowBuilder.hpp"
#include <pybind11/numpy.h>
#include <functional>
#include <typeindex>
#include <cstring>
#include <optional>

namespace DNDS::Runtime
{
    inline std::string PythonDtypeIdentity(const pybind11::dtype &dtype)
    {
        const auto kind = pybind11::str(dtype.attr("kind")).cast<std::string>();
        if (kind == "i" || kind == "u" || kind == "f")
            return pybind11::str(dtype.attr("name")).cast<std::string>();
        const auto canonical = dtype.attr("newbyteorder")("<");
        return "numpy.pod.v1:" + pybind11::str(canonical.attr("descr")).cast<std::string>() + ":" + std::to_string(dtype.itemsize());
    }
    inline ElementSchema PythonDtypeSchema(const pybind11::dtype &dtype)
    {
        namespace py = pybind11;
        DNDS_check_throw_info(dtype.itemsize() > 0 && !py::cast<bool>(dtype.attr("hasobject")) &&
                                  py::cast<bool>(dtype.attr("isnative")),
                              "element schema requires a native POD dtype");
#define DNDS_DTYPE_SCHEMA(T)             \
    if (dtype.equal(py::dtype::of<T>())) \
        return ElementSchema::For<T>();
        DNDS_DTYPE_SCHEMA(double)
        DNDS_DTYPE_SCHEMA(float)
        DNDS_DTYPE_SCHEMA(int64_t)
        DNDS_DTYPE_SCHEMA(uint64_t)
        DNDS_DTYPE_SCHEMA(int32_t)
        DNDS_DTYPE_SCHEMA(uint32_t)
        DNDS_DTYPE_SCHEMA(int16_t)
        DNDS_DTYPE_SCHEMA(uint16_t)
        DNDS_DTYPE_SCHEMA(int8_t)
        DNDS_DTYPE_SCHEMA(uint8_t)
        DNDS_DTYPE_SCHEMA(bool)
#undef DNDS_DTYPE_SCHEMA
        std::vector<ElementField> fields;
        const auto alignment = std::max<size_t>(1, py::cast<size_t>(dtype.attr("alignment")));
        const auto escapeName = [](const std::string &name)
        {
            std::string result;
            for (char character : name)
            {
                if (character == '\\' || character == '.' || character == '[' || character == ']')
                    result += '\\';
                result += character;
            }
            return result;
        };
        std::function<void(const py::dtype &, size_t, const std::string &)> visit;
        visit = [&](const py::dtype &type, size_t offset, const std::string &name)
        {
            auto sub = type.attr("subdtype");
            if (!sub.is_none())
            {
                const auto tuple = sub.cast<py::tuple>();
                const auto base = py::dtype::from_args(tuple[0]);
                size_t count = 1;
                for (auto dimension : tuple[1].cast<py::tuple>())
                    count = CheckedSize::Multiply(count, py::cast<size_t>(dimension));
                for (size_t i = 0; i < count; ++i)
                    visit(base, CheckedSize::Add(offset, CheckedSize::Multiply(i, size_t(base.itemsize()))), name + "[" + std::to_string(i) + "]");
                return;
            }
            auto members = type.attr("fields");
            if (!members.is_none())
            {
                for (auto field : type.attr("names").cast<py::tuple>())
                {
                    const auto item = members[field].cast<py::tuple>();
                    visit(py::dtype::from_args(item[0]), CheckedSize::Add(offset, item[1].cast<size_t>()),
                          name.empty() ? escapeName(py::str(field).cast<std::string>()) : name + "." + escapeName(py::str(field).cast<std::string>()));
                }
                return;
            }
            const auto leaf = PythonDtypeSchema(type);
            if (leaf.Fields().empty())
                fields.push_back({name, offset, ElementScalar::Byte, size_t(type.itemsize())});
            else
                for (const auto &word : leaf.Fields())
                    fields.push_back({name, CheckedSize::Add(offset, word.offset), word.scalar, word.count});
        };
        if (!dtype.attr("fields").is_none() || !dtype.attr("subdtype").is_none())
            visit(dtype, 0, "");
        else
        {
            const auto kind = py::str(dtype.attr("kind")).cast<std::string>();
            DNDS_check_throw_info(kind == "V", "NumPy dtype has no supported scalar-word codec");
            fields.push_back({"bytes", 0, ElementScalar::Byte, size_t(dtype.itemsize())});
        }
        const auto identity = PythonDtypeIdentity(dtype);
        auto described = ElementSchema::Record(identity, dtype.itemsize(), std::move(fields), {}, alignment).WithBinding(identity);
        if (auto registered = ElementSchema::Find(identity))
        {
            // A registered native dtype can report alignment1 in NumPy while
            // its C++ object binding requires a stronger alignment.
            const auto nativeShape = ElementSchema::Record(identity, described.Extent(), described.Fields(), {}, registered->Alignment());
            DNDS_check_throw_info(nativeShape.Definition() == registered->Definition(), "NumPy dtype differs from its registered element schema");
            return *registered;
        }
        return described;
    }
    inline pybind11::dtype PythonSchemaDtype(const ElementSchema &schema)
    {
        namespace py = pybind11;
        py::list names, formats, offsets;
        for (const auto &field : schema.Fields())
        {
            names.append(field.name);
            auto dtype = py::dtype(field.scalar == ElementScalar::Byte ? "V1" : ElementSchema::ScalarName(field.scalar));
            formats.append(field.count == 1 ? py::object(dtype) : py::object(py::make_tuple(dtype, py::make_tuple(field.count))));
            offsets.append(field.offset);
        }
        py::dict description;
        description["names"] = names;
        description["formats"] = formats;
        description["offsets"] = offsets;
        description["itemsize"] = schema.Extent();
        return py::dtype::from_args(description);
    }
    inline bool PythonSchemaMatchesDtype(const ElementSchema &schema, const pybind11::dtype &dtype)
    {
        const auto described = PythonDtypeSchema(dtype);
        if (described.Extent() != schema.Extent())
            return false;
        using Word = std::pair<size_t, ElementScalar>;
        auto words = [](const ElementSchema &value)
        {
            std::vector<Word> result;
            for (const auto &field : value.Fields())
                for (size_t i = 0; i < field.count; ++i)
                    result.emplace_back(field.offset + i * ElementSchema::ScalarBytes(field.scalar), field.scalar);
            std::sort(result.begin(), result.end());
            return result;
        };
        return words(schema) == words(described);
    }
    /// Binding-side POD codec. The runtime allocation contains no Python objects.
    struct PythonElement
    {
        std::type_index type;
        pybind11::dtype dtype;
        ElementSchema schema;
        size_t alignment;
        std::function<pybind11::object(const void *)> load;
        std::function<void(void *, pybind11::handle)> store;
    };
    using PythonElementShared = std::shared_ptr<const PythonElement>;

    inline PythonElementShared GetPythonElement(pybind11::handle type)
    {
        if (!pybind11::hasattr(type, "_dnds_array_element"))
            return {};
        auto capsule = pybind11::getattr(type, "_dnds_array_element").cast<pybind11::capsule>();
        auto *value = PyCapsule_GetPointer(capsule.ptr(), "DNDS.Runtime.PythonElement.v1");
        if (!value)
            throw pybind11::error_already_set();
        return *static_cast<PythonElementShared *>(value);
    }

    /// Register once per bound POD class, independently of Array widths/layouts.
    /// A NumPy dtype is optional; otherwise it is derived from the registered
    /// scalar/opaque-byte fields. A supplied dtype must describe those same words.
    template <class T, class... Options>
    void RegisterArrayElement(pybind11::class_<T, Options...> &type, ElementSchema schema,
                              pybind11::object numpyDtype = pybind11::none())
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);
        DNDS_check_throw_info(!pybind11::hasattr(type, "_dnds_array_element"), "duplicate POD Array registration");
        schema = ElementSchema::Register<T>(schema);
        auto dtype = numpyDtype.is_none() ? PythonSchemaDtype(schema) : pybind11::dtype::from_args(numpyDtype);
        DNDS_check_throw_info(size_t(dtype.itemsize()) == sizeof(T) && !pybind11::cast<bool>(dtype.attr("hasobject")) &&
                                  pybind11::cast<bool>(dtype.attr("isnative")),
                              "POD dtype size/representation mismatch");
        DNDS_check_throw_info(PythonSchemaMatchesDtype(schema, dtype), "POD dtype fields differ from the shared element schema");
        auto element = std::make_shared<PythonElement>(PythonElement{
            typeid(T), dtype, std::move(schema), alignof(T),
            [](const void *source) -> pybind11::object
            { T value; std::memcpy(&value, source, sizeof(T)); return pybind11::cast(std::move(value)); },
            [](void *destination, pybind11::handle object)
            { auto value = pybind11::cast<T>(object); std::memcpy(destination, &value, sizeof(T)); }});
        // The capsule belongs to the Python class; the codec does not retain
        // that class, so class -> capsule -> codec has no ownership cycle.
        type.attr("_dnds_array_element") = pybind11::capsule(new PythonElementShared(std::move(element)),
                                                             "DNDS.Runtime.PythonElement.v1", [](PyObject *capsule)
                                                             { delete static_cast<PythonElementShared *>(PyCapsule_GetPointer(capsule, "DNDS.Runtime.PythonElement.v1")); });
    }
    template <class T, class... Options>
    void RegisterArrayElement(pybind11::class_<T, Options...> &type, const std::string &identity,
                              pybind11::object numpyDtype = pybind11::none())
    {
        auto schema = ElementSchema::For<T>();
        DNDS_check_throw_info(schema.Identity() == identity, "Python POD identity differs from its registered native schema");
        RegisterArrayElement(type, std::move(schema), std::move(numpyDtype));
    }

    /// Python's dtype facade over the exact native allocation/layout record.
    /// Include this header in a binding TU; no width-specific Python classes.
    class PythonArray
    {
        ArrayStorage::Shared _storage;
        pybind11::dtype _dtype;
        PythonElementShared _element;
        std::optional<ElementSchema> _schema;
        pybind11::object _elementType = pybind11::none();

        static void ValidateDtype(const pybind11::dtype &dtype)
        {
            DNDS_check_throw_info(dtype.itemsize() > 0 && !pybind11::cast<bool>(dtype.attr("hasobject")) &&
                                      pybind11::cast<bool>(dtype.attr("isnative")),
                                  "Array requires a native POD dtype");
        }

    public:
        PythonArray(const PythonArray &other)
            : _storage(ArrayStorage::Share(other._storage)), _dtype(other._dtype), _element(other._element), _schema(other._schema), _elementType(other._elementType) {}
        PythonArray &operator=(const PythonArray &other)
        {
            if (this != &other)
            {
                auto shared = ArrayStorage::Share(other._storage);
                _storage = std::move(shared);
                _dtype = other._dtype;
                _element = other._element;
                _schema = other._schema;
                _elementType = other._elementType;
            }
            return *this;
        }
        PythonArray(PythonArray &&) noexcept = default;
        PythonArray &operator=(PythonArray &&) noexcept = default;
        PythonArray(RowLayout layout, pybind11::object dtypeOrClass, Device device = Device::Host(), AllocationOptions options = {})
        {
            _element = GetPythonElement(dtypeOrClass);
            _dtype = _element ? _element->dtype : pybind11::dtype::from_args(dtypeOrClass);
            if (_element)
                _elementType = std::move(dtypeOrClass);
            ValidateDtype(_dtype);
            _schema = _element ? _element->schema : PythonDtypeSchema(_dtype);
            DNDS_check_throw_info(!options.alignmentBytes || !(options.alignmentBytes & (options.alignmentBytes - 1)),
                                  "allocation alignment must be a power of two");
            options.alignmentBytes = std::max(options.alignmentBytes, _schema->Alignment());
            _storage = ArrayStorage::Allocate(std::move(layout), _dtype.itemsize(), device, options);
        }
        PythonArray(RowLayout layout, pybind11::dtype dtype, Device device = Device::Host(), AllocationOptions options = {}) : _dtype(std::move(dtype))
        {
            ValidateDtype(_dtype);
            _schema = PythonDtypeSchema(_dtype);
            DNDS_check_throw_info(!options.alignmentBytes || !(options.alignmentBytes & (options.alignmentBytes - 1)),
                                  "allocation alignment must be a power of two");
            options.alignmentBytes = std::max(options.alignmentBytes, _schema->Alignment());
            _storage = ArrayStorage::Allocate(std::move(layout), _dtype.itemsize(), device, options);
        }
        PythonArray(ArrayStorage::Shared storage, pybind11::dtype dtype)
            : _storage(std::move(storage)), _dtype(std::move(dtype))
        {
            ValidateDtype(_dtype);
            _schema = PythonDtypeSchema(_dtype);
            DNDS_check_throw_info(_storage && _storage->ElementBytes() == size_t(_dtype.itemsize()), "Array dtype size mismatch");
            const auto alignment = std::max<size_t>(1, pybind11::cast<size_t>(_dtype.attr("alignment")));
            DNDS_check_throw_info(Storage().IsAligned(alignment),
                                  "Array dtype alignment mismatch");
        }
        template <class T>
        static PythonArray Share(const Array<T> &array)
        {
            if constexpr (!std::is_arithmetic_v<T>)
            {
                auto info = pybind11::detail::get_type_info(typeid(T), false);
                if (info)
                {
                    auto type = pybind11::reinterpret_borrow<pybind11::object>(reinterpret_cast<PyObject *>(info->type));
                    if (auto element = GetPythonElement(type))
                    {
                        PythonArray result(array.SharedStorage(), element->dtype);
                        result._element = std::move(element);
                        result._schema = result._element->schema;
                        result._elementType = std::move(type);
                        return result;
                    }
                }
            }
            PythonArray result(array.SharedStorage(), pybind11::dtype::of<T>());
            result._schema = ElementSchema::For<T>();
            return result;
        }
        template <class T>
        bool MatchesType() const
        {
            if (_element)
                return _element->type == std::type_index(typeid(T));
            if constexpr (!std::is_arithmetic_v<T>)
            {
                // A class-only POD codec does not accept untyped opaque bytes.
                auto info = pybind11::detail::get_type_info(typeid(T), false);
                if (info && GetPythonElement(pybind11::handle(reinterpret_cast<PyObject *>(info->type))))
                    return false;
            }
            return _dtype.equal(pybind11::dtype::of<T>());
        }
        template <class T, class L = RowLayout>
        Array<T, L> AsArray() const
        {
            if (!MatchesType<T>())
                throw pybind11::type_error("native Array dtype mismatch (no implicit conversion)");
            return Array<T, L>::FromSharedStorage(_storage);
        }
        ArrayStorage::Shared SharedStorage() const { return ArrayStorage::Share(_storage); }
        ExclusiveRows EditRows() { return ExclusiveRows(_storage); }
        bool IsValid() const { return _storage->IsValid(); }
        uint64_t ShapeVersion() const { return _storage->ShapeVersion(); }
        RowLayout Layout() const { return _storage->Layout(); }
        const pybind11::dtype &Dtype() const { return _dtype; }
        const PythonElementShared &Element() const { return _element; }
        const ElementSchema &Schema() const { return *_schema; }
        const pybind11::object &ElementType() const { return _elementType; }
        bool SameElement(const PythonArray &other) const
        {
            return _dtype.equal(other._dtype) && bool(_element) == bool(other._element) &&
                   Schema().Identity() == other.Schema().Identity() && Schema().Definition() == other.Schema().Definition() &&
                   (!_element || _element->type == other._element->type);
        }
        PythonArray New(RowLayout layout, Device device) const
        {
            // Replacement must not alias the old allocation: reset is also
            // the recovery operation for an abandoned exclusive shape edit.
            return WithStorage(ArrayStorage::Allocate(std::move(layout), _dtype.itemsize(), device, AllocationOptions{_schema->Alignment()}));
        }
        PythonArray WithStorage(ArrayStorage::Shared storage) const
        {
            PythonArray result(std::move(storage), _dtype);
            result._element = _element;
            result._schema = _schema;
            result._elementType = _elementType;
            if (_element)
                DNDS_check_throw_info(result.Storage().IsAligned(_element->alignment),
                                      "Array POD alignment mismatch");
            return result;
        }
        Buffer Storage() const { return _storage->Values(); }
        ExecutionContext Context(std::shared_ptr<StreamResource> stream = {}) const { return {Storage().GetDevice(), std::move(stream)}; }
        ReadAccess Read(ExecutionContext context) const { return _storage->Read(context); }
        WriteAccess Write(ExecutionContext context) { return _storage->Write(context); }
        CopyOperation CopyFrom(const PythonArray &source, ExecutionContext context)
        {
            DNDS_check_throw_info(SameElement(source), "Array copy dtype mismatch");
            return _storage->CopyFrom(*source._storage, context);
        }
    };
}

namespace pybind11::detail
{
    /// Converts shallow handles, never values or row metadata. Array Reset on a
    /// converted argument rebinds that handle only, just like a C++ Array copy.
    /// Pass Arrays by value for element updates; use explicit setters to replace
    /// a solver-owned Array. The shared allocation outlives either language owner.
    template <class T>
    struct type_caster<DNDS::Runtime::ArrayHandle<T>>
    {
        using Native = DNDS::Runtime::ArrayHandle<T>;
        PYBIND11_TYPE_CASTER(Native, const_name("DNDSR.DNDS.Runtime.Array"));

        bool load(handle source, bool)
        {
            using Facade = DNDS::Runtime::PythonArray;
            if (!isinstance<Facade>(source))
                return false;
            const auto &array = pybind11::cast<const Facade &>(source);
            if (!array.template MatchesType<T>())
                return false;
            value = array.template AsArray<T>();
            return true;
        }
        static handle cast(const Native &array, return_value_policy, handle)
        {
            return pybind11::cast(DNDS::Runtime::PythonArray::Share(array)).release();
        }
    };
}
