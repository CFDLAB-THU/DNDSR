# Paradigm in designing DNDS {#paradigm}


DNDS is designed to be a set of commonly used infrastructure that can be used in CFD-like code. When organizing data and algorithms in CFD code, the programmer has to cope with geometry and field data, which correspond to mesh/grid related code and field related code. When the CFD scheme involves unstructured mesh and high order discretization, the grid and field code could become rather complex. Therefore, it is a natural thing to put some levels of abstraction here, and cover up the raw data types using C++ features.

## Basic Data Structure

There has been countless C++ involved computational applications in the field of computer graphics and CG designing (like [blender](https://github.com/blender/blender)), CAD (like [FreeCAD](https://github.com/FreeCAD/FreeCAD)), CAE mesh generation (like [gmsh](https://gitlab.onelab.info/gmsh/gmsh)) that involve very complex unstructured and polymorphic geometry data. And massive computational applications including deep learning architectures (like [PyTorch](https://github.com/pytorch/pytorch)) use high levels of abstraction directly on fully structured and homogeneously organized data arrays.

Unstructured CFD applications are different from both types of computational models, where both complex geometry and massive homogeneous numeric operations are required but easier to cover. Unstructured CFD code only involve limited types of geometry elements and connection type, which can be nearly hard-coded; while while global high-rank structured arrays are mostly not needed, only rank 2 to 5 arrays with potentially non-uniform sizes could be utilized.

So, how do we design the interface used in implementing CFD (By CFD, I mean math formulae of discrete schemes)? Here we inspect some references of famous open cfd code chunks:

It seems concerning basic data arrangement, the OpenFOAM and SU2 both require the data to be able to be accessed with random accessors (random_iterator, pointer, subscript or similar):

[OpenFOAM's gradient calculation](https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/finiteVolume/finiteVolume/gradSchemes/LeastSquaresGrad/LeastSquaresGrad.C):

```cpp
forAll(vtf, celli)
    {
        flatVtf[celli] = vtf[celli];
    }
```

[SU2's gradient calculation](https://github.com/su2code/SU2/blob/master/SU2_CFD/include/gradients/computeGradientsGreenGauss.hpp):

```cpp
for (size_t iVertex = 0; iVertex < geometry.GetnVertex(iMarker); ++iVertex)
      {
        //... code
        if (!nodes->GetDomain(iPoint)) continue;
        //... code
      }
```

And their directly operating data objects seem to be defined on a whole (zone of) mesh:

[OpenFOAM's gradient calculation](https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/finiteVolume/finiteVolume/gradSchemes/LeastSquaresGrad/LeastSquaresGrad.C):

```cpp
const List<List<label>>& stencilAddr = stencil.stencil();
const List<List<vector>>& lsvs = lsv.vectors();
```

[SU2's gradient calculation](https://github.com/su2code/SU2/blob/master/SU2_CFD/include/gradients/computeGradientsGreenGauss.hpp):

```cpp
size_t iPoint = geometry.vertex[iMarker][iVertex]->GetNode();
//
su2double volume = nodes->GetVolume(iPoint) + nodes->GetPeriodicVolume(iPoint);
```

Actually, SU2's [CVertex](https://github.com/su2code/SU2/blob/master/Common/include/geometry/dual_grid/CVertex.hpp) is a polymorphic class:

```cpp
class CVertex : public CDualGrid {
protected:
  unsigned long Nodes[1];               /*!< \brief Vector to store the global nodes of an element. */
  su2double Normal[3] = {0.0};          /*!< \brief Normal coordinates of the element and its center of gravity. */
  su2double Aux_Var;                    /*!< \brief Auxiliar variable defined only on the surface. */
  su2double CartCoord[3] = {0.0};       /*!< \brief Vertex cartesians coordinates. */
  su2double VarCoord[3] = {0.0};        /*!< \brief Used for storing the coordinate variation due to a surface modification. */
  long PeriodicPoint[5] = {-1};         /*!< \brief Store the periodic point of a boundary (iProcessor, iPoint) */
  bool ActDisk_Perimeter = false;       /*!< \brief Identify nodes at the perimeter of the actuator disk */
  short Rotation_Type;                  /*!< \brief Type of rotation associated with the vertex (MPI and periodic) */
  unsigned long Normal_Neighbor;        /*!< \brief Index of the closest neighbor. */
  su2double Basis_Function[3] = {0.0};  /*!< \brief Basis function values for interpolation across zones. */
  //...
}
```

CDualGrid stores the adjacency information, geometric information and auxiliary information in the class, and overrides base's methods for calculating some useful geometric information. So is SU2's CPrimalGrid class. 

However, OpenFOAM seems to maintain a primitive data array for mesh topology and geometry in [primitiveMesh](https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/OpenFOAM/meshes/primitiveMesh/primitiveMesh.H) class:


```cpp
class primitiveMesh
{
    // Permanent data

    // Primitive size data

        //- Number of internal points (or -1 if points not sorted)
        label nInternalPoints_;
        //- Number of points
        label nPoints_;
        //- Number of internal edges using 0 boundary points
        mutable label nInternal0Edges_;

//...

    // Shapes
        //- Cell shapes
        mutable cellShapeList* cellShapesPtr_;
//...

    // Connectivity
        //- Cell-cells
        mutable labelListList* ccPtr_;

    // On-the-fly edge addressing storage
        //- Temporary storage for addressing.
        mutable DynamicList<label> labels_;

    // Geometric data
        //- Cell centres
        mutable vectorField* cellCentresPtr_;

}
```

And OpenFOAM wraps these data with methods to access mesh topo and geom with inheritance.

DNDS does not intend to directly apply such methods at first, but intend to simplify the **MPI communications** on some **limited types** of data arrays. Communication for any complex object is secondary in DNDS, for most of the communication is needed only for arrays of basic types like `int64_t` and `double` (or the project aliases `DNDS::index` and `DNDS::real`) and their simple composite c-like-struct, which is implemented in `Array` and `ArrayTransformer` classes. Any communication on general objects would be a concept requiring the objects being able to serialize/deserialize themselves to a buffer in a given method and given order (which is closer to the communication model in PHengLEI).

The first application of DNDS, the simple CFV *euler* solver, does only invoke basic type communications in `ArrayTransformer`, and has yet to come up with any MPI-related bug (data corruption, dead lock...) since no hard MPI operation is needed outside the DNDS wrapping.

Also, DNDS recommends the user to put different kinds of data in different arrays instead of combining them at first, like in OpenFOAM:
```cpp
std::vector<real> faceArea;
std::vector<vec>  faceCent;

//not:
struct Face{
    real area;
    vec  cent;
};
std::vector<Face> faces;

```

Using DNDS provided data structure, one can consider `std::vector<simple_type>` to be able to manage its own communication pattern, somewhat like a PETSC Vector.

The reasoning behind this, is to separate different data genres, which may need different arrangements of communication, access and combination. For example, if one uses combined data:

```cpp
class Solution{
    real rho, ru, rv, rw, E, u, v, w, p, T;
public:
    void WriteStream(ByteStream&);  // ByteStream: conceptual serialization interface
    void ReadStream(ByteStream&);   // (not an actual class; represents any serialization mechanism)
};
std::vector<Solution> solutions;
```

then Write and Read would only involve the conserved variables. However, if one extends this to:

```cpp
class Solution{
    real rho, ru, rv, rw, E, u, v, w, p, T;
    real rho_1, ru_1, rv_1, rw_1, E_1;
public:
    void WriteStream(ByteStream&);  // ByteStream: conceptual serialization interface
    void ReadStream(ByteStream&);   // (not an actual class; represents any serialization mechanism)
};
```

and both X and X_1 variables need to be communicated through MPI, but at different phases of computation, then the `WriteStream` and `ReadStream` would not be sufficient for communicating; but in a polymorphic design, like using concept of templates or using virtual inheritance, the top level of abstraction must be general enough to take these matters into account. Also, it is a burden to add new communicated components to the class.

Therefore, in DNDS, it is recommended that the abstraction is delayed out of the arrays of data, or the abstraction should not be nested into the raw data arrays. Actually, DNDS is only dedicated to providing a means of using c-like random-access large arrays without the concern of communication, and any higher level of abstraction is left for the user.  

<!-- [OpenFOAM's gradient calculation](https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/finiteVolume/finiteVolume/gradSchemes/LeastSquaresGrad/LeastSquaresGrad.C):

```cpp

template<class Type, class Stencil>
Foam::tmp
<
    Foam::VolField<typename Foam::outerProduct<Foam::vector, Type>::type>
>
Foam::fv::LeastSquaresGrad<Type, Stencil>::calcGrad
(
    const VolField<Type>& vtf,
    const word& name
) const
{
    typedef typename outerProduct<vector, Type>::type GradType;

    const fvMesh& mesh = vtf.mesh();

    // Get reference to least square vectors
    const LeastSquaresVectors<Stencil>& lsv = LeastSquaresVectors<Stencil>::New
    (
        mesh
    );

    tmp<VolField<GradType>> tlsGrad
    (
        VolField<GradType>::New
        (
            name,
            mesh,
            dimensioned<GradType>
            (
                "zero",
                vtf.dimensions()/dimLength,
                Zero
            ),
            extrapolatedCalculatedFvPatchField<GradType>::typeName
        )
    );
    VolField<GradType>& lsGrad = tlsGrad.ref();
    Field<GradType>& lsGradIf = lsGrad;

    const extendedCentredCellToCellStencil& stencil = lsv.stencil();
    const List<List<label>>& stencilAddr = stencil.stencil();
    const List<List<vector>>& lsvs = lsv.vectors();

    // Construct flat version of vtf
    // including all values referred to by the stencil
    List<Type> flatVtf(stencil.map().constructSize(), Zero);

    // Insert internal values
    forAll(vtf, celli)
    {
        flatVtf[celli] = vtf[celli];
    }

    // Insert boundary values
    forAll(vtf.boundaryField(), patchi)
    {
        const fvPatchField<Type>& ptf = vtf.boundaryField()[patchi];

        label nCompact =
            ptf.patch().start()
          - mesh.nInternalFaces()
          + mesh.nCells();

        forAll(ptf, i)
        {
            flatVtf[nCompact++] = ptf[i];
        }
    }

    // Do all swapping to complete flatVtf
    stencil.map().distribute(flatVtf);

    // Accumulate the cell-centred gradient from the
    // weighted least-squares vectors and the flattened field values
    forAll(stencilAddr, celli)
    {
        const labelList& compactCells = stencilAddr[celli];
        const List<vector>& lsvc = lsvs[celli];

        forAll(compactCells, i)
        {
            lsGradIf[celli] += lsvc[i]*flatVtf[compactCells[i]];
        }
    }

    // Correct the boundary conditions
    lsGrad.correctBoundaryConditions();
    gaussGrad<Type>::correctBoundaryConditions(vtf, lsGrad);

    return tlsGrad;
}
```


[SU2's gradient calculation](https://github.com/su2code/SU2/blob/master/SU2_CFD/include/gradients/computeGradientsGreenGauss.hpp):

```cpp

template<size_t nDim, class FieldType, class GradientType>
void computeGradientsGreenGauss(CSolver* solver,
                                MPI_QUANTITIES kindMpiComm,
                                PERIODIC_QUANTITIES kindPeriodicComm,
                                CGeometry& geometry,
                                const CConfig& config,
                                const FieldType& field,
                                size_t varBegin,
                                size_t varEnd,
                                GradientType& gradient)
{
  const size_t nPointDomain = geometry.GetnPointDomain();

#ifdef HAVE_OMP
  constexpr size_t OMP_MAX_CHUNK = 512;

  const auto chunkSize = computeStaticChunkSize(nPointDomain, omp_get_max_threads(), OMP_MAX_CHUNK);
#endif

  /*--- For each (non-halo) volume integrate over its faces (edges). ---*/

  SU2_OMP_FOR_DYN(chunkSize)
  for (size_t iPoint = 0; iPoint < nPointDomain; ++iPoint)
  {
    auto nodes = geometry.nodes;

    /*--- Cannot preaccumulate if hybrid parallel due to shared reading. ---*/
    if (omp_get_num_threads() == 1) AD::StartPreacc();
    AD::SetPreaccIn(nodes->GetVolume(iPoint));
    AD::SetPreaccIn(nodes->GetPeriodicVolume(iPoint));

    for (size_t iVar = varBegin; iVar < varEnd; ++iVar)
      AD::SetPreaccIn(field(iPoint,iVar));

    /*--- Clear the gradient. --*/

    for (size_t iVar = varBegin; iVar < varEnd; ++iVar)
      for (size_t iDim = 0; iDim < nDim; ++iDim)
        gradient(iPoint, iVar, iDim) = 0.0;

    /*--- Handle averaging and division by volume in one constant. ---*/

    su2double halfOnVol = 0.5 / (nodes->GetVolume(iPoint)+nodes->GetPeriodicVolume(iPoint));

    /*--- Add a contribution due to each neighbor. ---*/

    for (size_t iNeigh = 0; iNeigh < nodes->GetnPoint(iPoint); ++iNeigh)
    {
      size_t iEdge = nodes->GetEdge(iPoint,iNeigh);
      size_t jPoint = nodes->GetPoint(iPoint,iNeigh);

      /*--- Determine if edge points inwards or outwards of iPoint.
       *    If inwards we need to flip the area vector. ---*/

      su2double dir = (iPoint < jPoint)? 1.0 : -1.0;
      su2double weight = dir * halfOnVol;

      const auto area = geometry.edges->GetNormal(iEdge);
      AD::SetPreaccIn(area, nDim);

      for (size_t iVar = varBegin; iVar < varEnd; ++iVar)
      {
        AD::SetPreaccIn(field(jPoint,iVar));

        su2double flux = weight * (field(iPoint,iVar) + field(jPoint,iVar));

        for (size_t iDim = 0; iDim < nDim; ++iDim)
          gradient(iPoint, iVar, iDim) += flux * area[iDim];
      }

    }

    for (size_t iVar = varBegin; iVar < varEnd; ++iVar)
      for (size_t iDim = 0; iDim < nDim; ++iDim)
        AD::SetPreaccOut(gradient(iPoint,iVar,iDim));

    AD::EndPreacc();
  }
  END_SU2_OMP_FOR

  /*--- Add boundary fluxes. ---*/

  for (size_t iMarker = 0; iMarker < geometry.GetnMarker(); ++iMarker)
  {
    if ((config.GetMarker_All_KindBC(iMarker) != INTERNAL_BOUNDARY) &&
        (config.GetMarker_All_KindBC(iMarker) != NEARFIELD_BOUNDARY) &&
        (config.GetMarker_All_KindBC(iMarker) != PERIODIC_BOUNDARY))
    {
      /*--- Work is shared in inner loop as two markers
       *    may try to update the same point. ---*/

      SU2_OMP_FOR_STAT(32)
      for (size_t iVertex = 0; iVertex < geometry.GetnVertex(iMarker); ++iVertex)
      {
        size_t iPoint = geometry.vertex[iMarker][iVertex]->GetNode();
        auto nodes = geometry.nodes;

        /*--- Halo points do not need to be considered. ---*/

        if (!nodes->GetDomain(iPoint)) continue;

        su2double volume = nodes->GetVolume(iPoint) + nodes->GetPeriodicVolume(iPoint);

        const auto area = geometry.vertex[iMarker][iVertex]->GetNormal();

        for (size_t iVar = varBegin; iVar < varEnd; iVar++)
        {
          su2double flux = field(iPoint,iVar) / volume;

          for (size_t iDim = 0; iDim < nDim; iDim++)
            gradient(iPoint, iVar, iDim) -= flux * area[iDim];
        }
      }
      END_SU2_OMP_FOR
    }
  }

  /*--- If no solver was provided we do not communicate ---*/

  if (solver == nullptr) return;

  /*--- Account for periodic contributions. ---*/

  for (size_t iPeriodic = 1; iPeriodic <= config.GetnMarker_Periodic()/2; ++iPeriodic)
  {
    solver->InitiatePeriodicComms(&geometry, &config, iPeriodic, kindPeriodicComm);
    solver->CompletePeriodicComms(&geometry, &config, iPeriodic, kindPeriodicComm);
  }

  /*--- Obtain the gradients at halo points from the MPI ranks that own them. ---*/

  solver->InitiateComms(&geometry, &config, kindMpiComm);
  solver->CompleteComms(&geometry, &config, kindMpiComm);
}

``` -->
