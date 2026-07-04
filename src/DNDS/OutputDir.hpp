#pragma once

#include <filesystem>

#include "MPI.hpp"

namespace DNDS
{

    enum class OutputDirMode
    {
        Safe, ///< Collective: allreduce existence, master-only create, barrier, recheck.
        Fast  ///< Per-rank: check existence, create if missing (no collective communication).
    };

    namespace detail
    {
        /**
         * @brief Core implementation: ensures @p dirPath exists.
         *
         * Does nothing (returns true) if @p dirPath is empty or is "." (current directory).
         */
        inline bool createDirImpl(const std::filesystem::path &dirPath,
                                  const MPIInfo &mpi,
                                  OutputDirMode mode)
        {
            if (dirPath.empty() || dirPath == "." || dirPath == "./")
                return true;

            bool exists = std::filesystem::exists(dirPath);

            if (mode == OutputDirMode::Safe)
            {
                int all_exists = exists ? 1 : 0;
                MPI_Allreduce(MPI_IN_PLACE, &all_exists, 1, MPI_INT, MPI_LAND, mpi.comm);

                if (all_exists == 0)
                {
                    if (mpi.rank == 0)
                        std::filesystem::create_directories(dirPath);
                    MPI_Barrier(mpi.comm);
                }
            }
            else
            {
                if (!exists)
                    std::filesystem::create_directories(dirPath);
            }

            return std::filesystem::exists(dirPath);
        }

        inline bool createDirImpl(const std::filesystem::path &dirPath)
        {
            if (dirPath.empty() || dirPath == "." || dirPath == "./")
                return true;

            if (!std::filesystem::exists(dirPath))
                std::filesystem::create_directories(dirPath);

            return std::filesystem::exists(dirPath);
        }
    } // namespace detail

    /**
     * @brief MPI-collective output directory creation for a given file path.
     *
     * Extracts the parent directory from @p filePath and ensures it exists.
     * If the file has no directory component (bare filename e.g. "output.plt"),
     * the function returns true immediately — the current working directory
     * is assumed to always exist.
     *
     * @b Safe mode: all ranks check existence of the parent; an all-reduce
     * (MPI_LAND) decides whether the directory is absent on any rank. If it
     * is, the master rank creates it and a barrier follows, then all ranks
     * verify existence.
     *
     * @b Fast mode: each rank checks and creates independently — suitable
     * for hot code paths where no collective synchronization is desired.
     *
     * @param filePath  Full path to the output FILE (parent is extracted internally).
     * @param mpi       MPI communicator info (used only in Safe mode).
     * @param mode      Creation mode.
     * @return true     The parent directory existed or was successfully created.
     */
    inline bool createOutputDir(const std::filesystem::path &filePath,
                                const MPIInfo &mpi,
                                OutputDirMode mode)
    {
        return detail::createDirImpl(filePath.parent_path(), mpi, mode);
    }

    /**
     * @brief Fast-mode output directory creation without MPI dependency.
     *
     * Convenience overload for code paths that do not have an @ref MPIInfo
     * (e.g. CsvLog, SerializerFactory). Equivalent to createOutputDir(filePath,
     * mpi, OutputDirMode::Fast) with an unused MPI context.
     *
     * @param filePath  Full path to the output FILE (parent is extracted internally).
     * @return true     The parent directory existed or was successfully created.
     */
    inline bool createOutputDir(const std::filesystem::path &filePath)
    {
        return detail::createDirImpl(filePath.parent_path());
    }

    /**
     * @brief Create a directory directly (not the parent of a file).
     *
     * Use this when you have a directory path (e.g. a .dir directory for
     * per-rank serializers) that you want to ensure exists.
     *
     * @param dirPath   Directory path to create.
     * @param mpi       MPI communicator info.
     * @param mode      Creation mode.
     * @return true     The directory existed or was successfully created.
     */
    inline bool createOutputDirAsDir(const std::filesystem::path &dirPath,
                                     const MPIInfo &mpi,
                                     OutputDirMode mode)
    {
        return detail::createDirImpl(dirPath, mpi, mode);
    }

    /**
     * @brief Create a directory directly without MPI dependency.
     *
     * @param dirPath   Directory path to create.
     * @return true     The directory existed or was successfully created.
     */
    inline bool createOutputDirAsDir(const std::filesystem::path &dirPath)
    {
        return detail::createDirImpl(dirPath);
    }

} // namespace DNDS
