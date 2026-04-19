/// @file SerializerBase.cpp
/// @brief Out-of-line definition of the virtual destructor of #SerializerBase;
/// separated so the vtable has a single translation-unit home.

#include "SerializerBase.hpp"

namespace DNDS::Serializer
{
    SerializerBase::~SerializerBase() = default;
}