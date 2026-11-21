#pragma once

#include "Geom/Unstructured/UnstructuredMeshHelpers.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Polygon_2.h>
#include <random>

namespace CauchyTest
{
    enum class VortexPointDistribution
    {
        JitteredGrid,
        PoissonDisk
    };

    /// Deterministic Bridson sampling at approximately the requested uniform
    /// spacing. This is serial mesh-generation scratch on the reader only;
    /// it is not used by distributed closure or ghost building. The bin grid
    /// makes separation queries local instead of comparing every point pair.
    inline std::vector<std::array<double, 2>> VortexPoissonPoints(int subdivisions, bool disk)
    {
        using I = DNDS::index;
        constexpr double length = 10, center = 5, radius = 1.5;
        const double spacing = length / subdivisions, separation = .8 * spacing;
        const double binWidth = separation / std::sqrt(2.);
        const int binsPerAxis = int(std::ceil(length / binWidth));
        std::vector<I> bins(size_t(binsPerAxis) * binsPerAxis, -1), active;
        std::vector<std::array<double, 2>> points;
        std::mt19937_64 random(disk ? 202611052 : 202611051);
        const auto uniform = [&]
        { return double(random() >> 11) * 0x1.0p-53; };
        const auto legal = [&](double x, double y)
        {
            if (x <= .35 * spacing || x >= length - .35 * spacing || y <= .35 * spacing || y >= length - .35 * spacing)
                return false;
            const double distance = std::hypot(x - center, y - center) - radius;
            return disk ? distance < -.35 * spacing : distance > .35 * spacing;
        };
        const auto append = [&](double x, double y)
        {
            const I iPoint = I(points.size());
            points.push_back({x, y});
            active.push_back(iPoint);
            bins[I(y / binWidth) * binsPerAxis + I(x / binWidth)] = iPoint;
        };
        append(disk ? center : .5 * spacing, disk ? center : .5 * spacing);
        while (!active.empty())
        {
            const I slot = I(random() % uint64_t(active.size()));
            const auto origin = points[active[slot]];
            bool inserted = false;
            for (int attempt = 0; attempt < 30 && !inserted; ++attempt)
            {
                const double angle = 2 * std::acos(-1.) * uniform();
                const double distance = separation * std::sqrt(1 + 3 * uniform());
                const double x = origin[0] + distance * std::cos(angle), y = origin[1] + distance * std::sin(angle);
                if (!legal(x, y))
                    continue;
                const int bx = int(x / binWidth), by = int(y / binWidth);
                bool separated = true;
                for (int iy = std::max(0, by - 2); iy <= std::min(binsPerAxis - 1, by + 2) && separated; ++iy)
                    for (int ix = std::max(0, bx - 2); ix <= std::min(binsPerAxis - 1, bx + 2); ++ix)
                    {
                        const I neighbor = bins[I(iy) * binsPerAxis + ix];
                        if (neighbor >= 0 && std::hypot(points[neighbor][0] - x, points[neighbor][1] - y) < separation)
                        {
                            separated = false;
                            break;
                        }
                    }
                if (separated)
                {
                    append(x, y);
                    inserted = true;
                }
            }
            if (!inserted)
            {
                active[slot] = active.back();
                active.pop_back();
            }
        }
        return points;
    }

    /// Reproducible P1 constrained-Delaunay disk/outer periodic box. Global
    /// construction scratch exists only on the reader, before native sparse
    /// distribution/partition and final topology assembly. No replicated mesh
    /// is used by ghost building or interface matching.
    inline std::shared_ptr<DNDS::Geom::UnstructuredMesh> VortexPatch(
        const DNDS::MPIInfo &mpi, bool disk, int subdivisions = 16, int circleSegments = 48,
        VortexPointDistribution distribution = VortexPointDistribution::JitteredGrid, bool periodicBox = true)
    {
        namespace G = DNDS::Geom;
        namespace R = DNDS::Runtime;
        using I = DNDS::index;
        using K = CGAL::Exact_predicates_inexact_constructions_kernel;
        using Vertex = CGAL::Triangulation_vertex_base_with_info_2<I, K>;
        using Face = CGAL::Constrained_triangulation_face_base_2<K>;
        using Data = CGAL::Triangulation_data_structure_2<Vertex, Face>;
        using Triangulation = CGAL::Constrained_Delaunay_triangulation_2<K, Data, CGAL::Exact_predicates_tag>;
        constexpr double length = 10, center = 5, radius = 1.5;
        DNDS_check_throw_info(subdivisions >= 8 && subdivisions <= 256 && circleSegments >= 16 && circleSegments <= 512,
                              "vortex fixture resolution outside bounds");
        G::NativeSerialMeshData serial;
        DNDS::MPI::LocalStage(mpi, [&]
                              {
            if(mpi.rank != 0) return;
            Triangulation mesh;
            std::vector<Triangulation::Vertex_handle> circle;
            CGAL::Polygon_2<K> polygon;
            const double pi = std::acos(-1.);
            for(int i=0;i<circleSegments;++i)
            {
                const double angle=2*pi*i/circleSegments;
                const K::Point_2 point(center+radius*std::cos(angle),center+radius*std::sin(angle));
                circle.push_back(mesh.insert(point));
                polygon.push_back(point);
            }
            for(int i=0;i<circleSegments;++i) mesh.insert_constraint(circle[i],circle[(i+1)%circleSegments]);
            struct Boundary
            {
                Triangulation::Vertex_handle a,b;
                int zone;
            };
            std::vector<Boundary> boundary;
            for(int i=0;i<circleSegments;++i) boundary.push_back({circle[i],circle[(i+1)%circleSegments],101});
            const double spacing=length/subdivisions;
            if(!disk)
                for(int side=0;side<4;++side)
                {
                    std::vector<Triangulation::Vertex_handle> edge;
                    for(int i=0;i<=subdivisions;++i)
                    {
                        const double t=i*spacing;
                        edge.push_back(mesh.insert(side==0 ? K::Point_2(0,t) : side==1 ? K::Point_2(length,t)
                                                               : side==2 ? K::Point_2(t,0) : K::Point_2(t,length)));
                    }
                    const int zone=!periodicBox ? 102 : side==0 ? G::BC_ID_PERIODIC_1 : side==1 ? G::BC_ID_PERIODIC_1_DONOR
                                        : side==2 ? G::BC_ID_PERIODIC_2 : G::BC_ID_PERIODIC_2_DONOR;
                    for(int i=0;i<subdivisions;++i)
                    {
                        mesh.insert_constraint(edge[i],edge[i+1]);
                        boundary.push_back({edge[i],edge[i+1],zone});
                    }
                }
            if (distribution == VortexPointDistribution::PoissonDisk)
            {
                for (const auto &point : VortexPoissonPoints(subdivisions, disk)) mesh.insert(K::Point_2(point[0], point[1]));
            }
            else for(int y=1;y<subdivisions;++y)
                for(int x=1;x<subdivisions;++x)
                {
                    // Fixed analytic jitter avoids a grid-aligned triangulation
                    // without involving a random generator/version/seed.
                    const double px=x*spacing+.025*spacing*std::sin(13*x+7*y);
                    const double py=y*spacing+.025*spacing*std::sin(5*x+11*y);
                    const K::Point_2 point(px,py);
                    const bool inside=polygon.bounded_side(point)==CGAL::ON_BOUNDED_SIDE;
                    if(inside==disk && std::abs(std::hypot(px-center,py-center)-radius)>.1*spacing) mesh.insert(point);
                }
            std::vector<Triangulation::Face_handle> triangles;
            std::vector<Triangulation::Vertex_handle> vertices;
            for(auto face=mesh.finite_faces_begin();face!=mesh.finite_faces_end();++face)
            {
                const auto a=face->vertex(0)->point(),b=face->vertex(1)->point(),c=face->vertex(2)->point();
                const K::Point_2 centroid((a.x()+b.x()+c.x())/3,(a.y()+b.y()+c.y())/3);
                if((polygon.bounded_side(centroid)==CGAL::ON_BOUNDED_SIDE)==disk)
                {
                    triangles.push_back(face);
                    for(int j=0;j<3;++j) vertices.push_back(face->vertex(j));
                }
            }
            std::sort(vertices.begin(),vertices.end(),[](auto a,auto b){return a->point()<b->point();});
            vertices.erase(std::unique(vertices.begin(),vertices.end()),vertices.end());
            for(I iNode=0;iNode<I(vertices.size());++iNode) vertices[iNode]->info()=iNode;
            serial.coordinates=R::Array<double>(R::RowLayout::Dense(vertices.size(),3));
            serial.cellNodes=R::Array<I>(R::RowLayout::Dense(triangles.size(),3));
            serial.boundaryNodes=R::Array<I>(R::RowLayout::Dense(boundary.size(),2));
            serial.cellInfo=R::Array<G::ElemInfo>(R::RowLayout::Dense(triangles.size(),1));
            serial.boundaryInfo=R::Array<G::ElemInfo>(R::RowLayout::Dense(boundary.size(),1));
            serial.cellIds=R::Array<I>(R::RowLayout::Dense(triangles.size(),1));
            serial.nodeIds=R::Array<I>(R::RowLayout::Dense(vertices.size(),1));
            serial.boundaryIds=R::Array<I>(R::RowLayout::Dense(boundary.size(),1));
            auto coordinates=serial.coordinates.Write();
            auto cellNodes=serial.cellNodes.Write(),boundaryNodes=serial.boundaryNodes.Write();
            auto cellInfo=serial.cellInfo.Write(),boundaryInfo=serial.boundaryInfo.Write();
            auto cellIds=serial.cellIds.Write(),nodeIds=serial.nodeIds.Write(),boundaryIds=serial.boundaryIds.Write();
            for(I iNode=0;iNode<I(vertices.size());++iNode)
            {
                coordinates.View()(iNode,0)=CGAL::to_double(vertices[iNode]->point().x());
                coordinates.View()(iNode,1)=CGAL::to_double(vertices[iNode]->point().y());
                coordinates.View()(iNode,2)=0;
                nodeIds.View()(iNode,0)=iNode;
            }
            for(I iCell=0;iCell<I(triangles.size());++iCell)
            {
                for(int j=0;j<3;++j) cellNodes.View()(iCell,j)=triangles[iCell]->vertex(j)->info();
                cellInfo.View()(iCell,0)={G::Elem::Tri3,G::INTERNAL_ZONE};
                cellIds.View()(iCell,0)=iCell;
            }
            for(I iBnd=0;iBnd<I(boundary.size());++iBnd)
            {
                boundaryNodes.View()(iBnd,0)=boundary[iBnd].a->info();
                boundaryNodes.View()(iBnd,1)=boundary[iBnd].b->info();
                boundaryInfo.View()(iBnd,0)={G::Elem::Line2,boundary[iBnd].zone};
                boundaryIds.View()(iBnd,0)=iBnd;
            } });
        std::optional<G::NativePeriodicImportOptions> periodic;
        if (!disk && periodicBox)
        {
            periodic.emplace();
            periodic->frames.translation[1].map() = G::tPoint{length, 0, 0};
            periodic->frames.translation[2].map() = G::tPoint{0, length, 0};
        }
        auto owned = G::DistributeNativeSerialMeshOwned(mpi, std::move(serial), 2, 0, {}, periodic);
        owned = G::DeriveAndPartitionNativeMeshOwned(std::move(owned), mpi, {}, 0, 0);
        G::PrepareMeshOptions prepare;
        prepare.buildSerialOut = false;
        prepare.reorderCells = 1;
        return G::UnstructuredMesh::FromNative(mpi, G::PrepareNativeMeshForSolverOwned(std::move(owned), mpi, prepare));
    }
}
