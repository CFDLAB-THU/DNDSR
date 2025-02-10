#include "DNDS/Array.hpp"

#include "DNDS/SerializerBase.hpp"
#include "DNDS/SerializerJSON.hpp"

int main()
{
    using namespace DNDS;
    Array<real, 3> a1;                           // staticFixed
    Array<real, DynamicSize> b1;                 // fixed
    Array<real, NonUniformSize, 3> c1;           // staticMax
    Array<real, NonUniformSize, DynamicSize> d1; // Max
    Array<real, NonUniformSize> e1;

    a1.Resize(4);

    b1.Resize(4, 3);

    assert(c1.RowSizeMax() == 3);
    c1.Resize(4);
    c1.ResizeRow(0, 3);
    c1.ResizeRow(1, 3);
    c1.ResizeRow(2, 3);
    c1.ResizeRow(3, 3);

    d1.Resize(4, 3);
    assert(d1.RowSizeMax() == 3);
    d1.ResizeRow(0, 3);
    d1.ResizeRow(1, 3);
    d1.ResizeRow(2, 3);
    d1.ResizeRow(3, 3);

    e1.Resize(4);
    e1.ResizeRow(0, 3);
    e1.ResizeRow(1, 3);
    e1.ResizeRow(2, 3);
    e1.ResizeRow(3, 3);
    e1.Compress();

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
        {
            a1(i, j) = i + j;
            b1(i, j) = i + j;
            c1(i, j) = i + j;
            d1(i, j) = i + j;
            e1(i, j) = i + j;
        }

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
        {
            a1(i, j) = i + j;
            b1(i, j) = i + j;
            c1(i, j) = i + j;
            d1(i, j) = i + j;
            e1(i, j) = i + j;
        }

    std::cout << a1 << std::endl;
    std::cout << b1 << std::endl;
    std::cout << c1 << std::endl;
    std::cout << d1 << std::endl;
    std::cout << e1 << std::endl;

    // TEST: e1
    e1.Resize(3, [](DNDS::index ir)
              { return ir + 1; });

    e1(0, 0) = 3;
    e1(1, 0) = e1(1, 1) = 1;
    e1(2, 0) = e1(2, 1) = e1(2, 2) = 2;
    std::cout << e1 << std::endl;

    e1.Decompress();
    e1.ResizeRow(1, 4);
    e1(1, 0) = e1(1, 1) = e1(1, 2) = e1(1, 3) = 0.1;
    std::cout << e1 << std::endl;

    e1.Compress();
    std::cout << e1 << std::endl;

    e1[0][0] = 1;
    e1[1][0] = e1[1][1] = 3;
    e1[2][0] = e1[2][1] = e1[2][2] = 4;
    std::cout << e1 << std::endl;

    // TODO: more tests on different types and function overloads

    // serializerP test:
    DNDS::Serializer::SerializerBaseSSP serializerP = std::make_shared<DNDS::Serializer::SerializerJSON>();
    std::dynamic_pointer_cast<DNDS::Serializer::SerializerJSON>(serializerP)->SetUseCodecOnUint8(true);

    serializerP->OpenFile("test_arraysOut.json", false);
    serializerP->CreatePath("/main");
    serializerP->GoToPath("/main");
    serializerP->CreatePath("array1");
    e1.WriteSerializer(serializerP, "e1", Serializer::ArrayGlobalOffset_Parts);
    b1.WriteSerializer(serializerP, "b1", Serializer::ArrayGlobalOffset_Parts);
    serializerP->CloseFile();

    serializerP->OpenFile("test_arraysOut1.json", true);
    serializerP->GoToPath("/main");
    Serializer::ArrayGlobalOffset offsetV = Serializer::ArrayGlobalOffset_Unknown;
    e1.ReadSerializer(serializerP, "e1", offsetV);
    b1.ReadSerializer(serializerP, "b1", offsetV);
    serializerP->CloseFile();

    std::cout << "read data:" << std::endl;
    std::cout << e1 << std::endl;
    std::cout << b1 << std::endl;

    return 0;
}