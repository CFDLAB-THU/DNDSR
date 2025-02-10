#include <mpi.h>
#include <hdf5.h>
#include <iostream>
#include <vector>
#include <sstream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // Get MPI communicator
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Info info = MPI_INFO_NULL;
    int rank;
    MPI_Comm_rank(comm, &rank);
    int size;
    MPI_Comm_size(comm, &size);

    // Create an HDF5 property list for MPI-IO
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, info);

    // Open the HDF5 file (collectively) with parallel access
    hid_t file_id = H5Fopen("parallel_array_example.h5", H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    if (file_id < 0) {
        std::cerr << "Error creating or opening file\n";
        MPI_Finalize();
        return 1;
    }

    // Create an array to write (e.g., 10 elements per rank)
    std::vector<int> data(10, rank);  // Each rank writes its rank value in the array

    // Create a unique dataset name for each rank
    std::stringstream dataset_name;
    dataset_name << "rank_" << rank;

    // Create the dataspace for each rank's dataset
    hsize_t dims[1] = { data.size() };
    hid_t dataspace_id = H5Screate_simple(1, dims, nullptr);

    // Write the array to the file
    hid_t dataset_id = H5Dcreate(file_id, dataset_name.str().c_str(), H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());

    // Close resources
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
    H5Pclose(plist_id);

    MPI_Finalize();
    return 0;
}