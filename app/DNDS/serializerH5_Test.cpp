#include "DNDS/SerializerH5.hpp"
#include <iostream>

void testSerializerH5(const DNDS::MPIInfo &mpiInfo)
{
    using namespace DNDS;
    using namespace DNDS::Serializer;
    std::vector<real> rVec = {1. + mpiInfo.rank * 100.0, 2, 3.3, 4, 5.4};
    std::vector<real> rVec2 = {1, 2, 3.3, 4, 5.4};
    std::vector<rowsize> rsVec0 = {1, 0, 1, 0, 1};

    ssp<std::vector<DNDS::index>> p_iVec0;
    DNDS_MAKE_SSP(p_iVec0, std::vector<DNDS::index>{1 + mpiInfo.rank * 100, 2, 3, 4, 5, 6, 5, 4, 3, 2});
    ssp<std::vector<DNDS::index>> p_iVec1 = p_iVec0;

    SerializerH5 serializer(mpiInfo);
    serializer.SetChunkAndDeflate(1024, 1);

    serializer.OpenFile("testOut.dnds.h5", false);
    serializer.CreatePath("main");
    serializer.GoToPath("main");
    serializer.CreatePath("very/deep/path");
    serializer.GoToPath("very/deep/path");
    serializer.WriteString("array_sig", "TABLE_StaticFixed__8_6_6_-1");
    serializer.GoToPath("/main");
    serializer.WriteRealVector("aRealVec", rVec, ArrayGlobalOffset_Parts);
    serializer.WriteRealVector("aRealVec_dist", rVec, ArrayGlobalOffset{DNDS::index(rVec.size()) * mpiInfo.size, DNDS::index(rVec.size()) * mpiInfo.rank});

    serializer.WriteRowsizeVector("aRSVec", rsVec0, ArrayGlobalOffset_Parts);
    serializer.WriteSharedIndexVector("aIndexVec_shared", p_iVec0, ArrayGlobalOffset_Parts);
    serializer.WriteSharedIndexVector("aIndexVec_shared_1", p_iVec1, ArrayGlobalOffset_Parts);
    serializer.WriteUint8Array("aRealVecWithoutType", (uint8_t *)rVec2.data(), DNDS::index(rVec2.size()) * sizeof(real), ArrayGlobalOffset_Parts);
    serializer.CloseFile();

    serializer.OpenFile("testOut.dnds.h5", true);
    serializer.GoToPath("main");
    ArrayGlobalOffset offsetV{0, 0};
    offsetV = ArrayGlobalOffset_Unknown;
    serializer.ReadRealVector("aRealVec", rVec, offsetV);
    offsetV = ArrayGlobalOffset_Unknown;
    serializer.ReadRealVector("aRealVec_dist", rVec, offsetV);
    offsetV = ArrayGlobalOffset_Unknown;
    serializer.ReadRowsizeVector("aRSVec", rsVec0, offsetV);
    offsetV = ArrayGlobalOffset_Unknown;
    serializer.ReadSharedIndexVector("aIndexVec_shared", p_iVec0, offsetV);
    offsetV = ArrayGlobalOffset_Unknown;
    serializer.ReadSharedIndexVector("aIndexVec_shared_1", p_iVec1, offsetV);
    offsetV = ArrayGlobalOffset_Unknown;
    DNDS::index sizeRead;
    serializer.ReadUint8Array("aRealVecWithoutType", nullptr, sizeRead, offsetV);
    DNDS_assert(sizeRead == rVec2.size() * sizeof(real));
    serializer.ReadUint8Array("aRealVecWithoutType", (uint8_t *)rVec2.data(), sizeRead, offsetV);
    serializer.CloseFile();

    // auto a = new double[1000 * 1024];
    // for (int i = 0; i < 1000 * 1024; i++)
    //     a[i] = 0.99 + i;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    auto mpi = DNDS::MPIInfo();
    mpi.setWorld();

    testSerializerH5(mpi);

    MPI_Finalize();
    return 0;
}

// int main()
// {
//     using base64 = cppcodec::base64_rfc4648;

//     std::vector<uint8_t> decoded = base64::decode("AAAAAAAA8D8AAAAAAAAAQGZmZmZmZgpAAAAAAAAAEECamZmZmZkVQA==");
//     return 0;
// }

// #include <mpi.h>
// #include <hdf5.h>
// #include <iostream>
// #include <vector>
// #include <sstream>

// int main(int argc, char **argv)
// {
//     MPI_Init(&argc, &argv);

//     // Get MPI communicator
//     MPI_Comm comm = MPI_COMM_WORLD;
//     MPI_Info info = MPI_INFO_NULL;
//     int rank;
//     MPI_Comm_rank(comm, &rank);

//     // Create an HDF5 property list for MPI-IO
//     hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
//     H5Pset_fapl_mpio(plist_id, comm, info);

//     // Open or create the HDF5 file with parallel access
//     hid_t file_id = H5Fcreate("parallel_array_example.h5", H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
//     if (file_id < 0)
//     {
//         std::cerr << "Error creating file\n";
//         MPI_Finalize();
//         return 1;
//     }

//     // Create an array to write (e.g., 10 elements per rank)
//     std::vector<int> data(10, rank); // Each rank writes its rank value in the array

//     // Create a unique dataset name for each rank
//     std::stringstream dataset_name;
//     dataset_name << "rank_" << rank;

//     // Write the array to the file
//     hsize_t dims[1] = {data.size()};
//     hid_t dataspace_id = H5Screate_simple(1, dims, nullptr);
//     hid_t dataset_id = H5Dcreate(file_id, dataset_name.str().c_str(), H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

//     H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());

//     // Close resources
//     H5Dclose(dataset_id);
//     H5Sclose(dataspace_id);
//     H5Fclose(file_id);
//     H5Pclose(plist_id);

//     MPI_Finalize();
//     return 0;
// }