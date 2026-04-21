/**
 * @file test_Elements.cpp
 * @brief Comprehensive doctest-based unit tests for element traits and shape functions.
 *
 * Tests cover:
 *   A. ElementTraits data integrity:
 *      1. Basic identification fields (elemType, dim, order, etc.)
 *      2. Standard coordinates consistency
 *      3. Face definitions (GetFaceType, faceNodes)
 *      4. Order elevation (elevatedType, elevSpans, elevNodeSpanTypes)
 *      5. Bisection refinement (numBisect, bisectElements)
 *      6. VTK compatibility (vtkCellType, vtkNodeOrder)
 *
 *   B. Shape function correctness:
 *      1. Partition of unity: sum_j N_j(p) == 1
 *      2. Nodal interpolation: N_i(node_j) == delta_ij
 *      3. First derivative consistency (FD vs analytic)
 *      4. Second derivative consistency (FD of D1)
 *
 * This is a serial test (no MPI required).
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "Geom/Elements.hpp"
#include "Geom/Quadrature.hpp"

#include <cmath>
#include <random>
#include <array>
#include <iostream>
#include <set>

using namespace DNDS::Geom;
using namespace DNDS::Geom::Elem;

// Convenience: 3D point from 1–3 coords, rest zeroed.
static tPoint MakePoint(t_real x, t_real y = 0, t_real z = 0)
{
    tPoint p;
    p << x, y, z;
    return p;
}

// Get reference node coordinates (columns of a 3×N matrix).
static Eigen::Matrix<t_real, 3, Eigen::Dynamic> NodeCoords(ElemType t)
{
    return GetStandardCoord(t);
}

// Random interior point for each param-space type.
// Uses a fixed seed for reproducibility.
static tPoint RandomInteriorPoint(ElemType t, std::mt19937 &rng)
{
    std::uniform_real_distribution<t_real> u01(0.05, 0.95);
    std::uniform_real_distribution<t_real> um11(-0.8, 0.8);

    Element e{t};
    auto ps = e.GetParamSpace();

    switch (ps)
    {
    case LineSpace:
        return MakePoint(um11(rng));
    case TriSpace:
    {
        // Ensure xi + et < 1
        t_real xi = u01(rng) * 0.5;
        t_real et = u01(rng) * (1.0 - xi) * 0.8;
        return MakePoint(xi, et);
    }
    case QuadSpace:
        return MakePoint(um11(rng), um11(rng));
    case TetSpace:
    {
        t_real xi = u01(rng) * 0.3;
        t_real et = u01(rng) * (1.0 - xi) * 0.3;
        t_real zt = u01(rng) * (1.0 - xi - et) * 0.8;
        return MakePoint(xi, et, zt);
    }
    case HexSpace:
        return MakePoint(um11(rng), um11(rng), um11(rng));
    case PrismSpace:
    {
        t_real xi = u01(rng) * 0.5;
        t_real et = u01(rng) * (1.0 - xi) * 0.8;
        t_real zt = um11(rng);
        return MakePoint(xi, et, zt);
    }
    case PyramidSpace:
    {
        // zt in (0, 1), base in (-1,1)^2 scaled by (1-zt)
        t_real zt = u01(rng) * 0.8;
        t_real scale = (1.0 - zt) * 0.8;
        t_real xi = um11(rng) * scale;
        t_real et = um11(rng) * scale;
        return MakePoint(xi, et, zt);
    }
    default:
        return MakePoint(0);
    }
}

// All valid element types.
static constexpr ElemType AllTypes[] = {
    Line2, Line3, Tri3, Tri6, Quad4, Quad9,
    Tet4, Tet10, Hex8, Hex27, Prism6, Prism18,
    Pyramid5, Pyramid14};

// 2D element types that have faces
static constexpr ElemType Elem2DWithFaces[] = {
    Tri3, Tri6, Quad4, Quad9};

// 3D element types that have faces
static constexpr ElemType Elem3DWithFaces[] = {
    Tet4, Tet10, Hex8, Hex27, Prism6, Prism18, Pyramid5, Pyramid14};

// O1 element types
static constexpr ElemType O1Types[] = {
    Line2, Tri3, Quad4, Tet4, Hex8, Prism6, Pyramid5};

// O2 element types
static constexpr ElemType O2Types[] = {
    Line3, Tri6, Quad9, Tet10, Hex27, Prism18, Pyramid14};

// ===================================================================
// Part A: ElementTraits Data Integrity Tests
// ===================================================================

TEST_CASE("ElementTraits: basic identification fields are consistent")
{
    for (auto t : AllTypes)
    {
        Element e{t};
        CAPTURE(t);
        
        // Basic sanity checks
        CHECK(e.GetDim() >= 1);
        CHECK(e.GetDim() <= 3);
        CHECK(e.GetNumNodes() >= e.GetNumVertices());
        CHECK(e.GetNumVertices() >= 2);  // At least a line
        
        // Order should be 1 or 2 for supported elements
        CHECK((e.GetOrder() == 1 || e.GetOrder() == 2));
        
        // ParamSpace volume should be positive (accessed via helper)
        CHECK(ParamSpaceVolume(e.GetParamSpace()) > 0);
    }
}

TEST_CASE("ElementTraits: standard coordinates have correct dimensions")
{
    for (auto t : AllTypes)
    {
        auto coords = NodeCoords(t);
        Element e{t};
        CAPTURE(t);
        
        // Should have 3 rows (x, y, z)
        CHECK(coords.rows() == 3);
        
        // Should have numNodes columns
        CHECK(coords.cols() == e.GetNumNodes());
        
        // For 1D elements, y and z should be zero
        if (e.GetDim() == 1)
        {
            for (DNDS::index i = 0; i < e.GetNumNodes(); i++)
            {
                CHECK(coords(1, i) == doctest::Approx(0.0));
                CHECK(coords(2, i) == doctest::Approx(0.0));
            }
        }
        
        // For 2D elements, z should be zero
        if (e.GetDim() == 2)
        {
            for (DNDS::index i = 0; i < e.GetNumNodes(); i++)
            {
                CHECK(coords(2, i) == doctest::Approx(0.0));
            }
        }
    }
}

TEST_CASE("ElementTraits: face definitions are consistent for 2D elements")
{
    for (auto t : Elem2DWithFaces)
    {
        Element e{t};
        CAPTURE(t);
        
        int numFaces = e.GetNumFaces();
        CHECK(numFaces > 0);
        
        for (int iFace = 0; iFace < numFaces; iFace++)
        {
            Element faceElem = e.ObtainFace(iFace);
            
            // Face should have consistent dimensionality
            CHECK(faceElem.GetDim() == e.GetDim() - 1);
            
            // Face should be a line element
            CHECK(faceElem.GetDim() == 1);
        }
    }
}

TEST_CASE("ElementTraits: face definitions are consistent for 3D elements")
{
    for (auto t : Elem3DWithFaces)
    {
        Element e{t};
        CAPTURE(t);
        
        int numFaces = e.GetNumFaces();
        CHECK(numFaces > 0);
        
        for (int iFace = 0; iFace < numFaces; iFace++)
        {
            Element faceElem = e.ObtainFace(iFace);
            
            // Face should have consistent dimensionality
            CHECK(faceElem.GetDim() == e.GetDim() - 1);
            
            // Face should be a 2D element
            CHECK(faceElem.GetDim() == 2);
        }
    }
}

TEST_CASE("ElementTraits: ExtractFaceNodes works correctly")
{
    // Test Tri3 face extraction
    {
        Element e{Tri3};
        std::vector<DNDS::index> nodes = {0, 1, 2};  // 3 nodes
        std::array<DNDS::index, 2> faceNodes;
        
        // Edge 0: should extract nodes 0 and 1
        e.ExtractFaceNodes(0, nodes, faceNodes);
        CHECK(faceNodes[0] == 0);
        CHECK(faceNodes[1] == 1);
        
        // Edge 1: should extract nodes 1 and 2
        e.ExtractFaceNodes(1, nodes, faceNodes);
        CHECK(faceNodes[0] == 1);
        CHECK(faceNodes[1] == 2);
        
        // Edge 2: should extract nodes 2 and 0
        e.ExtractFaceNodes(2, nodes, faceNodes);
        CHECK(faceNodes[0] == 2);
        CHECK(faceNodes[1] == 0);
    }
    
    // Test Quad4 face extraction
    {
        Element e{Quad4};
        std::vector<DNDS::index> nodes = {0, 1, 2, 3};  // 4 nodes
        std::array<DNDS::index, 2> faceNodes;
        
        // Each face should have 2 nodes
        for (int i = 0; i < 4; i++)
        {
            e.ExtractFaceNodes(i, nodes, faceNodes);
            // Just verify indices are valid
            CHECK(faceNodes[0] < 4);
            CHECK(faceNodes[1] < 4);
        }
    }
}

TEST_CASE("ElementTraits: order elevation data is consistent for O1 elements")
{
    // O1 elements should elevate to O2
    for (auto t : O1Types)
    {
        Element e{t};
        CAPTURE(t);
        
        Element elevated = e.ObtainElevatedElem();
        
        // Elevated element should exist
        CHECK(elevated.type != UnknownElem);
        
        // Elevated element should have more nodes
        CHECK(elevated.GetNumNodes() > e.GetNumNodes());
        
        // Elevated element should have same dimension
        CHECK(elevated.GetDim() == e.GetDim());
        
        // Should have elevation nodes defined
        CHECK(e.GetNumElev_O1O2() > 0);
    }
}

TEST_CASE("ElementTraits: order elevation data for specific elements")
{
    // Line2 -> Line3
    {
        Element e{Line2};
        CHECK(e.ObtainElevatedElem().type == Line3);
        CHECK(e.GetNumElev_O1O2() == 1);  // One edge midpoint
        
        // Check elevation span type
        Element spanElem = e.ObtainElevNodeSpan(0);
        CHECK(spanElem.type == Line2);  // Edge span
    }
    
    // Tri3 -> Tri6
    {
        Element e{Tri3};
        CHECK(e.ObtainElevatedElem().type == Tri6);
        CHECK(e.GetNumElev_O1O2() == 3);  // Three edge midpoints
    }
    
    // Quad4 -> Quad9
    {
        Element e{Quad4};
        CHECK(e.ObtainElevatedElem().type == Quad9);
        CHECK(e.GetNumElev_O1O2() == 5);  // 4 edges + 1 face center
    }
    
    // Hex8 -> Hex27
    {
        Element e{Hex8};
        CHECK(e.ObtainElevatedElem().type == Hex27);
        CHECK(e.GetNumElev_O1O2() == 19);  // 12 edges + 6 faces + 1 center
    }
}

TEST_CASE("ElementTraits: O2 elements do not have further elevation")
{
    // O2 elements should not have elevation defined
    for (auto t : O2Types)
    {
        Element e{t};
        CAPTURE(t);
        
        Element elevated = e.ObtainElevatedElem();
        CHECK(elevated.type == UnknownElem);
        CHECK(e.GetNumElev_O1O2() == 0);
    }
}

TEST_CASE("ElementTraits: ExtractElevNodeSpanNodes works correctly")
{
    // Test Tri3 elevation spans
    {
        Element e{Tri3};
        std::vector<DNDS::index> nodes = {0, 1, 2};  // Parent nodes
        std::array<DNDS::index, 2> spanNodes;
        
        // Each elevation span should connect 2 parent nodes
        for (int i = 0; i < e.GetNumElev_O1O2(); i++)
        {
            e.ExtractElevNodeSpanNodes(i, nodes, spanNodes);
            CHECK(spanNodes[0] < 3);  // References valid parent node
            CHECK(spanNodes[1] < 3);
        }
    }
}

TEST_CASE("ElementTraits: bisection data is valid for O2 elements")
{
    // Line3 bisection
    {
        Element e{Line3};
        CHECK(e.GetO2NumBisect() == 2);
        
        // Each sub-element should be Line2
        for (int i = 0; i < e.GetO2NumBisect(); i++)
        {
            CHECK(e.ObtainO2BisectElem(i).type == Line2);
        }
    }
    
    // Tri6 bisection
    {
        Element e{Tri6};
        CHECK(e.GetO2NumBisect() == 4);  // 4 sub-triangles
        for (int i = 0; i < e.GetO2NumBisect(); i++)
        {
            CHECK(e.ObtainO2BisectElem(i).type == Tri3);
        }
    }
    
    // Tet10 bisection
    {
        Element e{Tet10};
        CHECK(e.GetO2NumBisect() == 8);  // 8 sub-tets
        for (int i = 0; i < e.GetO2NumBisect(); i++)
        {
            CHECK(e.ObtainO2BisectElem(i).type == Tet4);
        }
    }
    
    // Hex27 bisection
    {
        Element e{Hex27};
        CHECK(e.GetO2NumBisect() == 8);  // 8 sub-hexes
        for (int i = 0; i < e.GetO2NumBisect(); i++)
        {
            CHECK(e.ObtainO2BisectElem(i).type == Hex8);
        }
    }
    
    // Prism18 bisection
    {
        Element e{Prism18};
        CHECK(e.GetO2NumBisect() == 8);  // 8 sub-prisms
        for (int i = 0; i < e.GetO2NumBisect(); i++)
        {
            CHECK(e.ObtainO2BisectElem(i).type == Prism6);
        }
    }
}

TEST_CASE("ElementTraits: VTK conversion works correctly")
{
    // Test VTK conversion for each element type
    for (auto t : AllTypes)
    {
        Element e{t};
        CAPTURE(t);
        
        // Create dummy node data
        std::vector<t_real> nodes(e.GetNumNodes());
        for (int i = 0; i < e.GetNumNodes(); i++)
            nodes[i] = static_cast<t_real>(i);
        
        // Convert to VTK
        auto [vtkCellType, vtkNodes] = ToVTKVertsAndData(e, nodes);
        
        // VTK cell type should be valid
        CHECK(vtkCellType > 0);
        
        // VTK nodes should have correct size
        CHECK(vtkNodes.size() <= e.GetNumNodes());
        CHECK(vtkNodes.size() > 0);
    }
}

TEST_CASE("ElementTraits: VTK node order is a valid permutation for simple elements")
{
    // Test VTK node ordering produces valid permutations
    DispatchElementType(Line2, [](auto traits) {
        std::set<int> seen;
        for (size_t i = 0; i < 2; i++)
            seen.insert(traits.vtkNodeOrder[i]);
        CHECK(seen.size() == 2);  // All unique
        CHECK(*seen.begin() == 0);
        CHECK(*seen.rbegin() == 1);
    });
    
    DispatchElementType(Line3, [](auto traits) {
        std::set<int> seen;
        for (size_t i = 0; i < 3; i++)
            seen.insert(traits.vtkNodeOrder[i]);
        CHECK(seen.size() == 3);  // All unique
        // VTK uses different ordering: 0, 2, 1 (midpoint last)
        CHECK(traits.vtkNodeOrder[0] == 0);
        CHECK(traits.vtkNodeOrder[1] == 2);
        CHECK(traits.vtkNodeOrder[2] == 1);
    });
    
    DispatchElementType(Tri6, [](auto traits) {
        std::set<int> seen;
        for (size_t i = 0; i < 6; i++)
            seen.insert(traits.vtkNodeOrder[i]);
        CHECK(seen.size() == 6);  // All unique
    });
    
    DispatchElementType(Hex8, [](auto traits) {
        std::set<int> seen;
        for (size_t i = 0; i < 8; i++)
            seen.insert(traits.vtkNodeOrder[i]);
        CHECK(seen.size() == 8);  // All unique
        CHECK(traits.vtkCellType == 12);  // VTK_HEXAHEDRON
    });
}

// ===================================================================
// Part B: Shape Function Tests
// ===================================================================

TEST_CASE("Shape functions: partition of unity")
{
    constexpr int nTrials = 10;
    std::mt19937 rng(42);

    for (auto t : AllTypes)
    {
        Element e{t};
        CAPTURE(t);
        for (int trial = 0; trial < nTrials; trial++)
        {
            auto p = RandomInteriorPoint(t, rng);
            tNj Nj;
            e.GetNj(p, Nj);

            t_real sum = Nj.sum();
            CHECK(sum == doctest::Approx(1.0).epsilon(1e-12));
        }
    }
}

TEST_CASE("Shape functions: nodal interpolation (Kronecker delta)")
{
    for (auto t : AllTypes)
    {
        Element e{t};
        auto coords = NodeCoords(t);
        DNDS::index nn = e.GetNumNodes();
        CAPTURE(t);

        for (DNDS::index j = 0; j < nn; j++)
        {
            tPoint p = coords.col(j);
            tNj Nj;
            e.GetNj(p, Nj);

            for (DNDS::index i = 0; i < nn; i++)
            {
                t_real expected = (i == j) ? 1.0 : 0.0;
                CAPTURE(i);
                CAPTURE(j);
                CHECK(Nj(0, i) == doctest::Approx(expected).epsilon(1e-10));
            }
        }
    }
}

TEST_CASE("Shape functions: D1 derivatives vs finite difference")
{
    constexpr int nTrials = 5;
    constexpr t_real h = 1e-7;
    constexpr t_real tol = 1e-5;
    std::mt19937 rng(123);

    for (auto t : AllTypes)
    {
        Element e{t};
        DNDS::index nn = e.GetNumNodes();
        int dim = e.GetDim();
        CAPTURE(t);

        for (int trial = 0; trial < nTrials; trial++)
        {
            auto p = RandomInteriorPoint(t, rng);

            tD1Nj D1Nj;
            e.GetD1Nj(p, D1Nj);

            // Finite difference for each parametric direction
            for (int d = 0; d < dim; d++)
            {
                tPoint pp = p, pm = p;
                pp[d] += h;
                pm[d] -= h;

                tNj Np, Nm;
                e.GetNj(pp, Np);
                e.GetNj(pm, Nm);

                for (DNDS::index j = 0; j < nn; j++)
                {
                    t_real fd = (Np(0, j) - Nm(0, j)) / (2 * h);
                    t_real an = D1Nj(d, j);
                    CAPTURE(d);
                    CAPTURE(j);
                    CHECK(an == doctest::Approx(fd).epsilon(tol));
                }
            }
        }
    }
}

TEST_CASE("Shape functions: D2 derivatives vs finite difference of D1")
{
    constexpr t_real h = 1e-6;
    constexpr t_real tol = 1e-3;
    std::mt19937 rng(456);

    for (auto t : AllTypes)
    {
        Element e{t};
        DNDS::index nn = e.GetNumNodes();
        int dim = e.GetDim();
        CAPTURE(t);

        auto p = RandomInteriorPoint(t, rng);

        tDiNj DiNj;
        e.GetDiNj(p, DiNj, 2);

        // Test pure second derivatives (d^2/dxi_d^2)
        for (int d = 0; d < dim; d++)
        {
            tPoint pp = p, pm = p;
            pp[d] += h;
            pm[d] -= h;

            tD1Nj D1p, D1m;
            e.GetD1Nj(pp, D1p);
            e.GetD1Nj(pm, D1m);

            for (DNDS::index j = 0; j < nn; j++)
            {
                t_real fd_d2 = (D1p(d, j) - D1m(d, j)) / (2 * h);

                // Row index for d^2/dxi_d^2 in the DiNj layout
                DNDS::index row;
                if (dim == 1)
                    row = 2; // d^2/dxi^2
                else if (dim == 2)
                    row = 3 + (d == 0 ? 0 : 2); // row 3=d2/dxi2, row 5=d2/det2
                else
                    row = 4 + d; // rows 4,5,6 for d2/dxi2, d2/det2, d2/dzt2

                t_real an = DiNj(row, j);
                CAPTURE(d);
                CAPTURE(j);
                CHECK(an == doctest::Approx(fd_d2).epsilon(tol));
            }
        }
    }
}

TEST_CASE("Shape functions: all derivatives are finite")
{
    // Verify shape function derivatives don't produce NaN or Inf
    std::mt19937 rng(789);
    
    for (auto t : AllTypes)
    {
        Element e{t};
        auto p = RandomInteriorPoint(t, rng);
        CAPTURE(t);
        
        tDiNj DiNj;
        e.GetDiNj(p, DiNj, 3);
        
        // Check all entries are finite
        for (DNDS::index i = 0; i < DiNj.rows(); i++)
        {
            for (DNDS::index j = 0; j < DiNj.cols(); j++)
            {
                CHECK(std::isfinite(DiNj(i, j)));
            }
        }
    }
}

// ===================================================================
// Part C: Face Normal Orientation Tests
// ===================================================================
//
// On the standard reference element, all faces are flat (even for O2
// elements, since mid-edge nodes lie at exact geometric midpoints).
// This means the face normal is constant across the face.
//
// For each face, we compute the expected unit normal analytically from
// the face vertex positions, then verify that the Jacobian-derived
// normal at every quadrature point matches it (direction and magnitude).
//
// 2D faces (edges in the xy-plane):
//   tangent = v1 - v0
//   expected outward normal direction = (tangent.y, -tangent.x, 0)
//
// 3D faces (triangles/quads):
//   expected outward normal direction = (v1 - v0) × (v2 - v0)

/// Compute the centroid of the reference element by averaging all vertex
/// coordinates (not all nodes — just the first numVertices).
static tPoint CellCentroid(ElemType t)
{
    auto coords = NodeCoords(t);
    Element e{t};
    tPoint c = tPoint::Zero();
    for (int i = 0; i < e.GetNumVertices(); i++)
        c += coords.col(i);
    c /= e.GetNumVertices();
    return c;
}

/// Extract face node coordinates as a 3×N_face_nodes matrix.
static Eigen::Matrix<t_real, 3, Eigen::Dynamic> FaceCoords(
    ElemType cellType, int iFace)
{
    auto allCoords = NodeCoords(cellType);
    Element cell{cellType};
    Element face = cell.ObtainFace(iFace);
    int nFaceNodes = face.GetNumNodes();

    std::array<DNDS::index, 10> faceNodeIdx;
    Eigen::Array<DNDS::index, Eigen::Dynamic, 1> localIdx(cell.GetNumNodes());
    for (int i = 0; i < cell.GetNumNodes(); i++)
        localIdx(i) = i;

    cell.ExtractFaceNodes(iFace, localIdx, faceNodeIdx);

    Eigen::Matrix<t_real, 3, Eigen::Dynamic> fc(3, nFaceNodes);
    for (int i = 0; i < nFaceNodes; i++)
        fc.col(i) = allCoords.col(faceNodeIdx[i]);
    return fc;
}

/// Compute the expected unit outward normal for a 2D element's face (edge).
/// The face is a line segment from vertex v0 to v1 in the xy-plane.
/// Outward normal = (dy, -dx, 0) / length, verified against cell centroid.
static tPoint ExpectedNormal2D(
    const Eigen::Matrix<t_real, 3, Eigen::Dynamic> &fc,
    const tPoint &centroid)
{
    tPoint v0 = fc.col(0);
    tPoint v1 = fc.col(1);
    tPoint tangent = v1 - v0;
    tPoint n;
    n << tangent(1), -tangent(0), 0.0;

    // Ensure outward direction (away from centroid)
    tPoint faceMid = (v0 + v1) * 0.5;
    if (n.dot(faceMid - centroid) < 0)
        n = -n;

    return n.normalized();
}

/// Compute the expected unit outward normal for a 3D element's face.
/// Uses the first 3 face vertices: normal = (v1-v0) × (v2-v0).
/// Sign is verified against the cell centroid.
static tPoint ExpectedNormal3D(
    const Eigen::Matrix<t_real, 3, Eigen::Dynamic> &fc,
    const tPoint &centroid)
{
    tPoint v0 = fc.col(0);
    tPoint v1 = fc.col(1);
    tPoint v2 = fc.col(2);
    tPoint n = (v1 - v0).cross(v2 - v0);

    // Ensure outward direction (away from centroid)
    // Use the average of the first 3 vertices as face center
    tPoint faceMid = (v0 + v1 + v2) / 3.0;
    if (n.dot(faceMid - centroid) < 0)
        n = -n;

    return n.normalized();
}

TEST_CASE("Face normals: 2D elements match expected unit normal at quad points")
{
    const t_real tol = 1e-12;

    for (auto t : Elem2DWithFaces)
    {
        Element cell{t};
        tPoint centroid = CellCentroid(t);
        CAPTURE(t);

        for (int iFace = 0; iFace < cell.GetNumFaces(); iFace++)
        {
            Element face = cell.ObtainFace(iFace);
            auto fc = FaceCoords(t, iFace);
            tPoint expectedN = ExpectedNormal2D(fc, centroid);
            CAPTURE(iFace);

            Quadrature quad(face, 3);

            for (int iG = 0; iG < quad.GetNumPoints(); iG++)
            {
                auto [pParam, w] = quad.GetQuadraturePointInfo(iG);

                tD01Nj D01Nj;
                face.GetD01Nj(pParam, D01Nj);

                tJacobi J = ShapeJacobianCoordD01Nj(fc, D01Nj);

                // Computed normal from tangent
                tPoint tangent = J.col(0);
                tPoint computedN;
                computedN << tangent(1), -tangent(0), 0.0;
                tPoint computedUnitN = computedN.normalized();

                CAPTURE(iG);
                // Direction should match expected
                CHECK(computedUnitN(0) == doctest::Approx(expectedN(0)).epsilon(tol));
                CHECK(computedUnitN(1) == doctest::Approx(expectedN(1)).epsilon(tol));
                CHECK(computedUnitN(2) == doctest::Approx(expectedN(2)).epsilon(tol));

                // Normal should point outward
                tPoint pPhys = PPhysicsCoordD01Nj(fc, D01Nj);
                t_real outwardDot = computedN.dot(pPhys - centroid);
                CHECK(outwardDot > 0.0);

                // Face Jacobian determinant = |tangent| should be positive
                t_real jdet = JacobiDetFace<2>(J);
                CHECK(jdet > 0.0);

                // For flat faces on standard element, |tangent| should be
                // half the edge length (parametric range is [-1,1], length 2)
                tPoint edgeVec = fc.col(1) - fc.col(0);
                t_real expectedJdet = edgeVec.norm() / 2.0; // dphys/dparam scaling
                CHECK(jdet == doctest::Approx(expectedJdet).epsilon(tol));
            }
        }
    }
}

TEST_CASE("Face normals: 3D elements match expected unit normal at quad points")
{
    const t_real tol = 1e-10;

    for (auto t : Elem3DWithFaces)
    {
        Element cell{t};
        tPoint centroid = CellCentroid(t);
        CAPTURE(t);

        for (int iFace = 0; iFace < cell.GetNumFaces(); iFace++)
        {
            Element face = cell.ObtainFace(iFace);
            auto fc = FaceCoords(t, iFace);
            tPoint expectedN = ExpectedNormal3D(fc, centroid);
            CAPTURE(iFace);

            Quadrature quad(face, 3);

            for (int iG = 0; iG < quad.GetNumPoints(); iG++)
            {
                auto [pParam, w] = quad.GetQuadraturePointInfo(iG);

                tD01Nj D01Nj;
                face.GetD01Nj(pParam, D01Nj);

                tJacobi J = ShapeJacobianCoordD01Nj(fc, D01Nj);
                tPoint computedN = J.col(0).cross(J.col(1));
                t_real computedMag = computedN.norm();
                tPoint computedUnitN = computedN / computedMag;

                CAPTURE(iG);

                // Direction should match expected unit normal
                CHECK(computedUnitN(0) == doctest::Approx(expectedN(0)).epsilon(tol));
                CHECK(computedUnitN(1) == doctest::Approx(expectedN(1)).epsilon(tol));
                CHECK(computedUnitN(2) == doctest::Approx(expectedN(2)).epsilon(tol));

                // Normal should point outward (away from centroid)
                tPoint pPhys = PPhysicsCoordD01Nj(fc, D01Nj);
                t_real outwardDot = computedN.dot(pPhys - centroid);
                CHECK(outwardDot > 0.0);

                // Face Jacobian determinant should be positive
                t_real jdet = JacobiDetFace<3>(J);
                CHECK(jdet > 0.0);
                CHECK(jdet == doctest::Approx(computedMag).epsilon(tol));
            }
        }
    }
}

TEST_CASE("Face normals: CGNS outward convention (no centroid correction needed)")
{
    // Stronger test: the cross product / rotation normal should ALREADY
    // point outward WITHOUT needing to flip sign. This verifies that
    // the faceNodes ordering follows CGNS outward-normal convention
    // directly (right-hand rule for 3D, counterclockwise for 2D).
    //
    // This test does NOT use ExpectedNormal (which auto-flips).
    // Instead it directly checks: dot(raw_normal, face_center - centroid) > 0.

    for (auto t : Elem2DWithFaces)
    {
        Element cell{t};
        tPoint centroid = CellCentroid(t);
        CAPTURE(t);

        for (int iFace = 0; iFace < cell.GetNumFaces(); iFace++)
        {
            auto fc = FaceCoords(t, iFace);
            tPoint v0 = fc.col(0), v1 = fc.col(1);
            tPoint tangent = v1 - v0;
            tPoint rawNormal;
            rawNormal << tangent(1), -tangent(0), 0.0;

            tPoint faceMid = (v0 + v1) * 0.5;
            t_real dot = rawNormal.dot(faceMid - centroid);
            CAPTURE(iFace);
            CHECK(dot > 0.0);
        }
    }

    for (auto t : Elem3DWithFaces)
    {
        Element cell{t};
        tPoint centroid = CellCentroid(t);
        CAPTURE(t);

        for (int iFace = 0; iFace < cell.GetNumFaces(); iFace++)
        {
            auto fc = FaceCoords(t, iFace);
            tPoint v0 = fc.col(0), v1 = fc.col(1), v2 = fc.col(2);
            tPoint rawNormal = (v1 - v0).cross(v2 - v0);

            // Face center from vertices (use first 3 for simplicity)
            tPoint faceMid = (v0 + v1 + v2) / 3.0;
            t_real dot = rawNormal.dot(faceMid - centroid);
            CAPTURE(iFace);
            CHECK(dot > 0.0);
        }
    }
}

TEST_CASE("Edge topology: 3D elements have correct edge count and types")
{
    for (auto t : Elem3DWithFaces)
    {
        Element cell{t};
        CAPTURE(t);

        int numEdges = cell.GetNumEdges();
        CHECK(numEdges > 0);

        for (int iEdge = 0; iEdge < numEdges; iEdge++)
        {
            Element edge = cell.ObtainEdge(iEdge);
            CAPTURE(iEdge);

            // Edge should be a 1D element (Line2 or Line3)
            CHECK(edge.GetDim() == 1);
            CHECK(edge.GetNumNodes() >= 2);
            CHECK(edge.GetNumNodes() <= 3);
        }
    }
}

TEST_CASE("Edge topology: extracted edge nodes are valid cell nodes")
{
    for (auto t : Elem3DWithFaces)
    {
        Element cell{t};
        auto allCoords = NodeCoords(t);
        CAPTURE(t);

        // Build identity local node index array
        std::vector<DNDS::index> localIdx(cell.GetNumNodes());
        for (int i = 0; i < cell.GetNumNodes(); i++)
            localIdx[i] = i;

        for (int iEdge = 0; iEdge < cell.GetNumEdges(); iEdge++)
        {
            Element edge = cell.ObtainEdge(iEdge);
            std::array<DNDS::index, 3> edgeNodeIdx = {-1, -1, -1};
            cell.ExtractEdgeNodes(iEdge, localIdx, edgeNodeIdx);
            CAPTURE(iEdge);

            for (int i = 0; i < edge.GetNumNodes(); i++)
            {
                CAPTURE(i);
                CHECK(edgeNodeIdx[i] >= 0);
                CHECK(edgeNodeIdx[i] < cell.GetNumNodes());
            }

            // First two nodes (vertices) should be distinct
            CHECK(edgeNodeIdx[0] != edgeNodeIdx[1]);

            // Edge endpoints should have positive distance
            tPoint p0 = allCoords.col(edgeNodeIdx[0]);
            tPoint p1 = allCoords.col(edgeNodeIdx[1]);
            CHECK((p1 - p0).norm() > 0.0);

            // For quadratic edges, the midpoint should be between the endpoints
            if (edge.GetNumNodes() == 3)
            {
                tPoint pMid = allCoords.col(edgeNodeIdx[2]);
                tPoint expected = (p0 + p1) * 0.5;
                CHECK((pMid - expected).norm() < 1e-12);
            }
        }
    }
}

TEST_CASE("Edge topology: edges are unique (no duplicate edges per cell)")
{
    for (auto t : Elem3DWithFaces)
    {
        Element cell{t};
        CAPTURE(t);

        std::vector<DNDS::index> localIdx(cell.GetNumNodes());
        for (int i = 0; i < cell.GetNumNodes(); i++)
            localIdx[i] = i;

        // Collect edge vertex pairs as sorted pairs
        std::set<std::pair<DNDS::index, DNDS::index>> edgeSet;
        for (int iEdge = 0; iEdge < cell.GetNumEdges(); iEdge++)
        {
            std::array<DNDS::index, 3> edgeNodeIdx = {-1, -1, -1};
            cell.ExtractEdgeNodes(iEdge, localIdx, edgeNodeIdx);

            auto v0 = std::min(edgeNodeIdx[0], edgeNodeIdx[1]);
            auto v1 = std::max(edgeNodeIdx[0], edgeNodeIdx[1]);
            auto result = edgeSet.insert({v0, v1});
            CAPTURE(iEdge);
            CHECK(result.second); // No duplicate
        }
        CHECK(static_cast<int>(edgeSet.size()) == cell.GetNumEdges());
    }
}
