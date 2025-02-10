#include "DNDS/ArrayDerived/ArrayAdjacency.hpp"
#include "DNDS/ArrayDerived/ArrayEigenVector.hpp"
#include "DNDS/ArrayDerived/ArrayEigenMatirx.hpp"
#include "DNDS/ArrayDerived/ArrayEigenUniMatrixBatch.hpp"
#include "DNDS/ArrayDerived/ArrayEigenMatrixBatch.hpp"
#include "DNDS/SerializerJSON.hpp"

#include <cstdlib>
using namespace DNDS;

std::vector<double> argD;

/*
running:
valgrind --log-file=log_valgrind.log test/arrayDerived_test.exe
*/

void test_ADJ()
{
    int nRow = 1000;
    if (argD.size() == 1)
    {
        nRow = int(argD[0]);
    }

    // ArrayAdjacency<3> A_adj;
    // ArrayAdjacency<DynamicSize> A_adj;
    // ArrayAdjacency<NonUniformSize, DynamicSize> A_adj;
    // ArrayAdjacency<NonUniformSize, 3> A_adj;
    ArrayAdjacency<NonUniformSize, NonUniformSize> A_adj;

    A_adj.Resize(nRow, 3);

    if (A_adj._dataLayout == CSR || isTABLE_Max(A_adj._dataLayout))
    {
        for (DNDS::index i = 0; i < A_adj.Size(); i++)
            A_adj.ResizeRow(i, i % 3 + 1);
    }
    for (DNDS::index i = 0; i < A_adj.Size(); i++)
    {
        for (rowsize j = 0; j < A_adj.RowSize(i); j++)
            A_adj[i][j] = i;
    }
    A_adj.Compress();

    /**************************/
    // serializing
    DNDS::Serializer::SerializerBaseSSP serializerP = std::make_shared<DNDS::Serializer::SerializerJSON>();
    std::dynamic_pointer_cast<DNDS::Serializer::SerializerJSON>(serializerP)->SetUseCodecOnUint8(true);

    serializerP->OpenFile("test_ArrayAdjacency.json", false);
    A_adj.WriteSerializer(serializerP, "arrayAdj", Serializer::ArrayGlobalOffset_Parts);
    serializerP->CloseFile();

    serializerP->OpenFile("test_ArrayAdjacency.json", true);
    Serializer::ArrayGlobalOffset offsetV = Serializer::ArrayGlobalOffset_Unknown;
    A_adj.ReadSerializer(serializerP, "arrayAdj", offsetV);
    serializerP->CloseFile();
    /**************************/

    std::cout << A_adj << std::endl;
}

void test_Mat()
{
    int nRow = 1000;
    if (argD.size() == 1)
    {
        nRow = int(argD[0]);
    }

    //? these are incomplete anyway
    // ArrayEigenMatrix<3, 3> A_v;
    // ArrayEigenMatrix<DynamicSize, DynamicSize> A_v;
    // ArrayEigenMatrix<3, DynamicSize> A_v;
    // ArrayEigenMatrix<DynamicSize, 3> A_v;
    // ArrayEigenMatrix<NonUniformSize, 3, 3, 3> A_v;
    // ArrayEigenMatrix<NonUniformSize, NonUniformSize, 3, 3> A_v;
    ArrayEigenMatrix<NonUniformSize, NonUniformSize> A_v;

    A_v.Resize(nRow, 3, 3);

    if (A_v._dataLayout == CSR || isTABLE_Max(A_v._dataLayout))
    {
        for (DNDS::index i = 0; i < A_v.Size(); i++)
            A_v.ResizeMat(i, i % 3 + 1, 3);
    }
    for (DNDS::index i = 0; i < A_v.Size(); i++)
    {
        A_v[i].setIdentity();
    }
    A_v.Compress();

    /**************************/
    // serializing
    DNDS::Serializer::SerializerBaseSSP serializerP = std::make_shared<DNDS::Serializer::SerializerJSON>();
    std::dynamic_pointer_cast<DNDS::Serializer::SerializerJSON>(serializerP)->SetUseCodecOnUint8(true);

    serializerP->OpenFile("test_ArrayMatrix.json", false);
    A_v.WriteSerializer(serializerP, "arrayMat", Serializer::ArrayGlobalOffset_Parts);
    serializerP->CloseFile();

    serializerP->OpenFile("test_ArrayMatrix.json", true);
    Serializer::ArrayGlobalOffset offsetV = Serializer::ArrayGlobalOffset_Unknown;
    A_v.ReadSerializer(serializerP, "arrayMat", offsetV);
    serializerP->CloseFile();
    /**************************/

    for (DNDS::index i = 0; i < A_v.Size(); i++)
    {
        std::cout << A_v[i].transpose() << std::endl;
    }
}

void test_Vec()
{
    int nRow = 1000;
    if (argD.size() == 1)
    {
        nRow = int(argD[0]);
    }

    // ArrayEigenVector<3> A_v;
    // ArrayEigenVector<DynamicSize> A_v;
    // ArrayEigenVector<NonUniformSize, 3> A_v;
    // ArrayEigenVector<NonUniformSize, DynamicSize> A_v;
    ArrayEigenVector<NonUniformSize> A_v;

    A_v.Resize(nRow, 3);

    if (A_v._dataLayout == CSR || isTABLE_Max(A_v._dataLayout))
    {
        for (DNDS::index i = 0; i < A_v.Size(); i++)
            A_v.ResizeRow(i, i % 3 + 1);
    }
    for (DNDS::index i = 0; i < A_v.Size(); i++)
    {
        A_v[i].setLinSpaced(i, i + 0.5);
    }
    A_v.Compress();

    /**************************/
    // serializing
    DNDS::Serializer::SerializerBaseSSP serializerP = std::make_shared<DNDS::Serializer::SerializerJSON>();
    std::dynamic_pointer_cast<DNDS::Serializer::SerializerJSON>(serializerP)->SetUseCodecOnUint8(true);

    serializerP->OpenFile("test_ArrayVector.json", false);
    A_v.WriteSerializer(serializerP, "arrayVec", Serializer::ArrayGlobalOffset_Parts);
    serializerP->CloseFile();

    serializerP->OpenFile("test_ArrayVector.json", true);
    Serializer::ArrayGlobalOffset offsetV = Serializer::ArrayGlobalOffset_Unknown;
    A_v.ReadSerializer(serializerP, "arrayVec", offsetV);
    serializerP->CloseFile();
    /**************************/

    for (DNDS::index i = 0; i < A_v.Size(); i++)
    {
        std::cout << A_v[i].transpose() << std::endl;
    }
}

void test_UniMatBatch()
{
    int nRow = 1000;
    if (argD.size() == 1)
    {
        nRow = int(argD[0]);
    }

    ArrayEigenUniMatrixBatch<Eigen::Dynamic, Eigen::Dynamic> A_UMB;
    // ArrayEigenUniMatrixBatch<3, Eigen::Dynamic> A_UMB;
    // ArrayEigenUniMatrixBatch<Eigen::Dynamic, 4> A_UMB;
    // ArrayEigenUniMatrixBatch<3, 4> A_UMB;

    A_UMB.ResizeMatrix(3, 4);
    A_UMB.Resize(nRow, -1, -1);

    for (DNDS::index i = 0; i < A_UMB.Size(); i++)
    {
        A_UMB.ResizeBatch(i, i % 3 + 1);
        for (rowsize j = 0; j < A_UMB.BatchSize(i); j++)
            A_UMB(i, j).setIdentity(), A_UMB(i, j) *= i;
    }
    A_UMB.Compress();

    /**************************/
    // serializing
    DNDS::Serializer::SerializerBaseSSP serializerP = std::make_shared<DNDS::Serializer::SerializerJSON>();
    std::dynamic_pointer_cast<DNDS::Serializer::SerializerJSON>(serializerP)->SetUseCodecOnUint8(true);

    serializerP->OpenFile("test_ArrayUMB.json", false);
    A_UMB.WriteSerializer(serializerP, "arrayUMB", Serializer::ArrayGlobalOffset_Parts);
    serializerP->CloseFile();

    serializerP->OpenFile("test_ArrayUMB.json", true);
    Serializer::ArrayGlobalOffset offsetV = Serializer::ArrayGlobalOffset_Unknown;
    A_UMB.ReadSerializer(serializerP, "arrayUMB", offsetV);
    serializerP->CloseFile();
    /**************************/

    for (DNDS::index i = 0; i < A_UMB.Size(); i++)
    {
        std::cout << i << ": \n";
        auto Batch = A_UMB[i];
        for (auto &j : Batch)
            std::cout << j << std::endl;
    }
}

void test_MatBatch()
{
    MatrixBatch::UInt32PairIn64 two_ints;
    two_ints.setM(123321u);
    two_ints.setN(321123u);
    std::cout << two_ints.getM() << " " << two_ints.getN() << std::endl;
    MatrixBatch::UInt16QuadIn64 four_shorts;
    four_shorts.setA(1234);
    four_shorts.setB(4321);
    four_shorts.setC(4444);
    four_shorts.setD(1111);
    std::cout
        << four_shorts.getA() << " "
        << four_shorts.getB() << " "
        << four_shorts.getC() << " "
        << four_shorts.getD() << " " << std::endl;

    int nRow = 1000;
    if (argD.size() == 1)
    {
        nRow = int(argD[0]);
    }

    ArrayEigenMatrixBatch A_MB;

    A_MB.Resize(nRow);

    for (DNDS::index i = 0; i < A_MB.Size(); i++)
    {
        std::vector<Eigen::MatrixXd> mats;
        for (int j = 0; j < i % 3 + 1; j++)
            mats.emplace_back(), mats.back().setIdentity(j, j);
        A_MB.InitializeWriteRow(i, mats);
        // std::cout << MatrixBatch::getBufSize(mats) << std::endl;
    }

    A_MB.Compress();

    /**************************/
    // serializing
    DNDS::Serializer::SerializerBaseSSP serializerP = std::make_shared<DNDS::Serializer::SerializerJSON>();
    std::dynamic_pointer_cast<DNDS::Serializer::SerializerJSON>(serializerP)->SetUseCodecOnUint8(true);

    serializerP->OpenFile("test_ArrayMB.json", false);
    A_MB.WriteSerializer(serializerP, "arrayMB", Serializer::ArrayGlobalOffset_Parts);
    serializerP->CloseFile();

    serializerP->OpenFile("test_ArrayMB.json", true);
    Serializer::ArrayGlobalOffset offsetV = Serializer::ArrayGlobalOffset_Unknown;
    A_MB.ReadSerializer(serializerP, "arrayMB", offsetV);
    serializerP->CloseFile();
    /**************************/

    for (DNDS::index i = 0; i < A_MB.Size(); i++)
    {
        std::cout << i << ": \n";
        auto Batch = A_MB[i];
        // std::cout << A_MB.RowSize(i) << " " << Batch.Size() << std::endl;
        for (int j = 0; j < Batch.Size(); j++)
            std::cout << Batch[j] << std::endl;
    }
}

int main(int argc, char *argv[])
{
    // ! Disable MPI call to help serial mem check
    // MPI_Init(&argc, &argv);

    for (int i = 1; i < argc; i++)
    {
        double v = std::atof(argv[i]);
        argD.push_back(v);
    }

    // test_ADJ();
    // test_Vec();
    test_Mat();
    // test_UniMatBatch();
    // test_MatBatch();

    // MPI_Finalize();

    return 0;
}