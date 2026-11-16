// Standalone diagnostic: fixed input block-CSR actions, CUDA-event timing.
// Link against the active checkout's cauchy/cfv/geom/dnds targets. No MPI or
// numerical solver policy is exercised; this isolates the public row launcher.
#include "Cauchy/Implicit/CUDA/LinearBlockKernels.hpp"
#include "Cauchy/Implicit/LinearPattern.hpp"
#include "DNDS/Runtime/Array/ArrayCopy.hpp"
#include "DNDS/Serializer/JsonUtil.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

namespace R = DNDS::Runtime;
namespace C = DNDS::Cauchy;
namespace A = DNDS::Algebra;
using I = DNDS::index;
using DNDS::real;

namespace
{
    void CUDA(cudaError_t error)
    {
        DNDS_check_throw_info(error == cudaSuccess, cudaGetErrorString(error));
    }
    template <class T, class F>
    R::Array<T> Table(I rows, I width, F fill)
    {
        R::Array<T> result(R::RowLayout::Dense(rows, width));
        auto write = result.Write();
        const auto values = write.template View<R::TableRows>();
        for (I iRow = 0; iRow < rows; ++iRow)
            for (I j = 0; j < width; ++j)
                values(iRow, j) = fill(iRow, j);
        return result;
    }
    real Block(I row, I neighbor, int i, int j)
    {
        return (i == j ? -.01 : .00013 * (1 + i - j)) * (1 + .01 * ((row + neighbor) % 7));
    }
    real Inverse(I row, int i, int j)
    {
        return i == j ? .4 + .001 * (row % 5) : .0003 * (1 + i + j);
    }
    real Value(I row, int j) { return .2 + std::sin(.013 * row + .11 * j); }
    real RHS(I row, int j) { return .8 + std::cos(.009 * row - .07 * j); }
    real Expected(I row, int component, int n, bool zero, const std::vector<std::vector<I>> &columns)
    {
        real value[7];
        for (int i = 0; i < n; ++i)
        {
            value[i] = RHS(row, i);
            if (!zero)
                for (I k = 0; k < I(columns[row].size()); ++k)
                {
                    real product = 0;
                    for (int j = 0; j < n; ++j)
                        product += Block(row, k, i, j) * Value(columns[row][k], j);
                    value[i] -= product;
                }
        }
        real result = 0;
        for (int j = 0; j < n; ++j)
            result += Inverse(row, component, j) * value[j];
        return result;
    }
}

int main(int argc, char **argv)
{
    I side = 64;
    int components = 5, repeats = 16, device = 0, selectedMapping = -1;
    bool irregular = false;
    std::string matrixLayout = "aos", inverseLayout = "aos", indices = "int64", blockOrder = "columnMajor";
    I matrixPitchBytes = 0, inversePitchBytes = 0;
    size_t matrixAlignmentBytes = 0, inverseAlignmentBytes = 0;
    const auto options = [](const std::string &name)
    {
        DNDS_check_throw_info(name == "aos" || name == "soa" || name == "aosoa32", "invalid matrix layout");
        return A::MatrixTableOptions{name == "aos" ? A::MatrixTableLayout::AoS : name == "soa" ? A::MatrixTableLayout::SoA
                                                                                               : A::MatrixTableLayout::AoSoA32,
                                     32};
    };
    for (int i = 1; i < argc; ++i)
    {
        const std::string key = argv[i];
        DNDS_check_throw_info(i + 1 < argc, "benchmark option needs a value");
        const auto value = std::string(argv[++i]);
        if (key == "--side")
            side = std::stoll(value);
        else if (key == "--components")
            components = std::stoi(value);
        else if (key == "--repeats")
            repeats = std::stoi(value);
        else if (key == "--device")
            device = std::stoi(value);
        else if (key == "--mapping")
            selectedMapping = value == "thread" ? 0 : value == "componentGroup" ? 1
                                                                                : -1;
        else if (key == "--irregular")
            irregular = value == "true";
        else if (key == "--matrix-layout")
            matrixLayout = value;
        else if (key == "--inverse-layout")
            inverseLayout = value;
        else if (key == "--indices")
            indices = value;
        else if (key == "--block-order")
            blockOrder = value;
        else if (key == "--matrix-pitch-bytes")
            matrixPitchBytes = std::stoll(value);
        else if (key == "--inverse-pitch-bytes")
            inversePitchBytes = std::stoll(value);
        else if (key == "--matrix-alignment-bytes")
            matrixAlignmentBytes = std::stoull(value);
        else if (key == "--inverse-alignment-bytes")
            inverseAlignmentBytes = std::stoull(value);
        else
            DNDS_check_throw_info(false, "unknown benchmark option: " + key);
    }
    DNDS_check_throw_info(side >= 4 && side <= 256 && components >= 5 && components <= 7 && repeats > 0,
                          "invalid benchmark dimensions");
    CUDA(cudaSetDevice(device));
    cudaDeviceProp properties;
    CUDA(cudaGetDeviceProperties(&properties, device));
    cudaStream_t native;
    CUDA(cudaStreamCreateWithFlags(&native, cudaStreamNonBlocking));
    const auto context = R::ExecutionContext::CUDA(device, native);
    {
        DNDS_check_throw_info(indices == "int64" || indices == "int32", "invalid symbolic index width");
        DNDS_check_throw_info(blockOrder == "columnMajor" || blockOrder == "rowMajor", "invalid dense block order");
        const bool rowMajorBlocks = blockOrder == "rowMajor";
        const I rows = side * side * side;
        std::vector<std::vector<I>> columns(rows);
        std::vector<I> order(rows), position(rows), counts(rows), empty;
        std::iota(order.begin(), order.end(), I(0));
        position = order;
        const auto cell = [side](I x, I y, I z)
        { return ((z + side) % side * side + (y + side) % side) * side + (x + side) % side; };
        for (I iCell = 0; iCell < rows; ++iCell)
        {
            const I x = iCell % side, y = iCell / side % side, z = iCell / (side * side);
            columns[iCell] = {cell(x - 1, y, z), cell(x + 1, y, z), cell(x, y - 1, z), cell(x, y + 1, z), cell(x, y, z - 1), cell(x, y, z + 1)};
            if (irregular && iCell % 5 == 0)
                columns[iCell].resize(3 + iCell % 3);
            counts[iCell] = I(columns[iCell].size());
        }
        const auto blocks = std::accumulate(counts.begin(), counts.end(), I(0));
        const C::LinearPatternStorage hostPattern(columns, order, position, order, empty, indices == "int32" ? C::LinearIndexStorage::Int32 : C::LinearIndexStorage::Int64);
        const auto pattern = hostPattern.Placed(context.device, context);
        DNDS_check_throw_info(matrixPitchBytes >= 0 && inversePitchBytes >= 0 &&
                                  matrixPitchBytes % sizeof(real) == 0 && inversePitchBytes % sizeof(real) == 0,
                              "matrix pitches must be nonnegative multiples of sizeof(real)");
        auto matrixOptions = options(matrixLayout), inverseOptions = options(inverseLayout);
        matrixOptions.rowStridePaddingMultiple = matrixPitchBytes ? matrixPitchBytes / sizeof(real) : 1;
        inverseOptions.rowStridePaddingMultiple = inversePitchBytes ? inversePitchBytes / sizeof(real) : 1;
        matrixOptions.allocationAlignmentBytes = matrixAlignmentBytes;
        inverseOptions.allocationAlignmentBytes = inverseAlignmentBytes;
        A::BlockMatrixStorage matrix(counts, components, R::Device::Host(), matrixOptions, rowMajorBlocks);
        {
            auto write = matrix.Write();
            const auto view = write.ArrayView();
            for (I row = 0; row < rows; ++row)
                for (I k = 0; k < counts[row]; ++k)
                    for (int j = 0; j < components; ++j)
                        for (int i = 0; i < components; ++i)
                            view.Element(row, k, i, j) = Block(row, k, i, j);
        }
        const auto placed = matrix.Placed(context.device, context);
        const auto hostInverse = Table<real>(rows, components * components, [&](I row, I j)
                                             { return Inverse(row, int(rowMajorBlocks ? j / components : j % components), int(rowMajorBlocks ? j % components : j / components)); });
        const auto shapedInverse = R::ConvertLayout(hostInverse, inverseOptions.MakeLayout(rows, components * components));
        R::Array<real> inverse(shapedInverse.Layout(), context.device, inverseOptions.Allocation());
        inverse.CopyFrom(shapedInverse, context).Wait();
        const auto pitchCandidates = [&](const A::MatrixTableOptions &base, I logicalRows)
        {
            nlohmann::ordered_json candidates = nlohmann::ordered_json::array();
            std::vector<I> seen;
            for (const I bytes : {I(0), I(128), I(256)})
            {
                auto choice = base;
                choice.rowStridePaddingMultiple = bytes ? bytes / sizeof(real) : 1;
                const I stride = choice.MakeLayout(logicalRows, components * components).Stride();
                if (std::find(seen.begin(), seen.end(), stride) == seen.end())
                {
                    seen.push_back(stride);
                    candidates.push_back({{"representative_pitch_bytes", bytes}, {"physical_stride_elements", stride}});
                }
            }
            return candidates;
        };
        const auto input = Table<real>(rows, components, [](I row, I j)
                                       { return Value(row, int(j)); })
                               .To(context.device, context);
        const auto rhs = Table<real>(rows, components, [](I row, I j)
                                     { return RHS(row, int(j)); })
                             .To(context.device, context);
        R::Array<real> output(R::RowLayout::Dense(rows, components), context.device);
        const auto p = pattern.Read(context);
        const auto b = placed.Read(context);
        const auto d = inverse.Read(context), x = input.Read(context), r = rhs.Read(context);
        nlohmann::ordered_json records = nlohmann::ordered_json::array();
        for (bool zero : {false, true})
            for (bool groups : {false, true})
            {
                if (selectedMapping >= 0 && int(groups) != selectedMapping)
                    continue;
                std::optional<R::Array<real>::WriteLease> out(output.Write(context));
                C::CUDA::LinearBlockInput request{zero ? A::BlockMatrixArrayView<const real>{} : b.ArrayView(),
                                                  zero ? decltype(p.View().columns){} : p.View().columns,
                                                  p.View().order, d.View<R::UniformTableRows>(), rows, components, true, rowMajorBlocks};
                const auto launch = [&]
                {
                    C::CUDA::JacobiLinearRows(context, request, r.View<R::UniformTableRows>(),
                                              zero ? C::CUDA::VectorReadRows{} : x.View<R::UniformTableRows>(), {},
                                              out->View<R::TableRows>(), zero, groups);
                };
                for (int i = 0; i < 3; ++i)
                    launch();
                CUDA(cudaStreamSynchronize(native));
                std::vector<real> timings;
                cudaEvent_t begin, end;
                CUDA(cudaEventCreate(&begin));
                CUDA(cudaEventCreate(&end));
                for (int trial = 0; trial < 3; ++trial)
                {
                    CUDA(cudaEventRecord(begin, native));
                    for (int i = 0; i < repeats; ++i)
                        launch();
                    CUDA(cudaEventRecord(end, native));
                    CUDA(cudaEventSynchronize(end));
                    float milliseconds;
                    CUDA(cudaEventElapsedTime(&milliseconds, begin, end));
                    timings.push_back(milliseconds / repeats);
                }
                CUDA(cudaEventDestroy(begin));
                CUDA(cudaEventDestroy(end));
                out.reset();
                const auto host = output.To(R::Device::Host());
                const auto answer = host.Read();
                real error = 0;
                for (I row : {I(0), I(1), rows / 2, rows - 1})
                    for (int i = 0; i < components; ++i)
                        error = std::max(error, std::abs(answer.View()(row, i) - Expected(row, i, components, zero, columns)));
                DNDS_check_throw_info(error < 2e-12, "Jacobi diagnostic disagrees with independent CPU recurrence");
                std::sort(timings.begin(), timings.end());
                const I matrixBytes = zero ? rows * components * components * sizeof(real) : (blocks + rows) * components * components * sizeof(real);
                const I vectorBytes = (zero ? 2 : 3) * rows * components * sizeof(real);
                const I indexWidth = indices == "int32" ? sizeof(int) : sizeof(I);
                // Logical CSR row offsets counted once in useful payload. The
                // numeric block-view's separate int64 offsets are disclosed below.
                const I indexBytes = rows * sizeof(I) + (zero ? 0 : (blocks + rows + 1) * indexWidth);
                records.push_back({{"mapping", groups ? "componentGroup" : "thread"}, {"zero_start", zero}, {"median_ms", timings[1]}, {"minimum_ms", timings[0]}, {"maximum_ms", timings[2]}, {"model_unique_matrix_bytes", matrixBytes}, {"model_unique_vector_bytes", vectorBytes}, {"model_unique_index_bytes", indexBytes}, {"model_effective_GBs", (matrixBytes + vectorBytes + indexBytes) / (timings[1] * 1e6)}, {"cpu_sample_max_abs_error", error}});
            }
        const auto baseAlignment = [](const R::Array<real> &array)
        {
            for (size_t bytes = 4096; bytes >= sizeof(real); bytes /= 2)
                if (array.Storage().IsAligned(bytes))
                    return bytes;
            return size_t(1);
        };
        nlohmann::ordered_json storage{
            {"matrix_stride_elements", placed.Values().Layout().Stride()}, {"inverse_stride_elements", inverse.Layout().Stride()}, {"matrix_pitch_request_bytes", matrixPitchBytes}, {"inverse_pitch_request_bytes", inversePitchBytes}, {"matrix_alignment_request_bytes", matrixAlignmentBytes}, {"inverse_alignment_request_bytes", inverseAlignmentBytes}, {"matrix_base_alignment_bytes_at_least", baseAlignment(placed.Values())}, {"inverse_base_alignment_bytes_at_least", baseAlignment(inverse)}, {"matrix_padding_bytes", (placed.Values().Layout().Elements() - blocks * components * components) * sizeof(real)}, {"inverse_padding_bytes", (inverse.Layout().Elements() - rows * components * components) * sizeof(real)}, {"distinct_matrix_pitch_candidates", pitchCandidates(matrixOptions, blocks)}, {"distinct_inverse_pitch_candidates", pitchCandidates(inverseOptions, rows)}};
        std::cout << nlohmann::ordered_json{{"storage", storage}, {"gpu", properties.name}, {"device", device}, {"side", side}, {"rows", rows}, {"blocks", blocks}, {"components", components}, {"irregular", irregular}, {"block_order", blockOrder}, {"matrix_layout", matrixLayout}, {"inverse_layout", inverseLayout}, {"indices", indices}, {"matrix_allocated_bytes", placed.Values().Layout().Elements() * sizeof(real)}, {"inverse_allocated_bytes", inverse.Layout().Elements() * sizeof(real)}, {"numeric_block_offsets_bytes", (rows + 1) * sizeof(I)}, {"useful_byte_model", "unique logical matrix/vector/index payload; excludes duplicate numeric offsets and physical padding"}, {"repeats", repeats}, {"cache", "warmed fixed-input replay; payload exceeds L2 at default shape"}, {"timings", records}}.dump(2) << '\n';
    }
    CUDA(cudaStreamDestroy(native));
}
