# RecMatCondition Results

Bisection = 1:

- IV10_10.cgns 1.8522952e-03, 5.6943971e-02
  - (3600, 3600), nnz is 162000
  - Cond Est:  3.394e+02 ||| rho: 529.1
  - D^-1 A Cond Est:  1.559e+01 ||| rho: 1.88
  - 0.8795 0.7736
- IV10_Square_Pert0.cgns 2.0251535e-03, 6.8872222e-02
  - (3600, 3600), nnz is 162000
  - Cond Est:  8.917e+02 ||| rho: 471.2
  - D^-1 A Cond Est:  1.909e+01 ||| rho: 1.901
  - 0.8979 0.8062
- IV10_Tri.cgns 3.2207710e-04, 2.3019284e-02
  - (9072, 9072), nnz is 326592
  - Cond Est:  3.296e+02 ||| rho: 13.95
  - D^-1 A Cond Est:  2.376e+01 ||| rho: 1.919
  - 0.9109 0.8293
- IV10_Tri_Pert0.cgns 5.9789200e-04, 3.1945860e-02
  - (9072, 9072), nnz is 326592
  - Cond Est:  1.775e+03 ||| rho: 131.7
  - D^-1 A Cond Est:  6.166e+01 ||| rho: 1.968
  - 0.9682 0.9372

Bisection = 0:

```
====================
../data/outUnsteady/PP5/IV/condTest-out_B0_Square_RecMatrix.dnds.h5
<HDF5 group "/VR_Matrix/cell2cellFace/father" (6 members)>
b'ArrayEigenUniMatrixBatch__-1_-1'
A Cond Max:   4.44e+01
max Sym deviation 1.00e-14

=== Matirx A ===
(900, 900), nnz is 40500
/home/harry/projects/DNDSR/build/../script/analysis/RecMatCondition.py:261: SparseEfficiencyWarning: spilu converted its input to CSC format
  ilu = spilu(A, fill_factor=20)
[12.47234964+0.j 12.47058755+0.j 12.46858375+0.j]
[4232.40564241+0.j 4168.45018781+0.j 4049.81297133+0.j]
Cond Est:  3.394e+02 ||| rho: 4232

=== Matirx D^-1 A ===
(900, 900), nnz is 40500
[0.1252479 -7.05308588e-06j 0.1252479 +7.05308588e-06j
 0.12525887-8.52289422e-06j]
[1.87477065+0.j 1.87026764+0.j 1.85938097+0.j]
Cond Est:  1.497e+01 ||| rho: 1.875

=== Matirx AJ ===
(900, 900), nnz is 32400
[-0.87477065+0.j -0.87026764+0.j  0.87477065+0.j]
rho: 0.8748

=== Matirx AG ===
(900, 900)
[-0.76522369+0.j -0.75736576+0.j -0.73853568+0.j]
rho: 0.7652



====================
../data/outUnsteady/PP5/IV/condTest-out_B0_Square_Pert0_RecMatrix.dnds.h5
<HDF5 group "/VR_Matrix/cell2cellFace/father" (6 members)>
b'ArrayEigenUniMatrixBatch__-1_-1'
A Cond Max:   4.65e+01
max Sym deviation 1.15e-14

=== Matirx A ===
(900, 900), nnz is 40500
[5.06408096+0.j 5.16934828+0.j 5.2900602 +0.j]
[3087.93107385+0.j 2869.11070423+0.j 1745.59750154+0.j]
Cond Est:  6.098e+02 ||| rho: 3088

=== Matirx D^-1 A ===
(900, 900), nnz is 40500
[0.11572945+0.j 0.11866979+0.j 0.11932596+0.j]
[1.88429373+0.j 1.88137307+0.j 1.88069846+0.j]
Cond Est:  1.628e+01 ||| rho: 1.884

=== Matirx AJ ===
(900, 900), nnz is 32400
[ 0.88429373+0.j -0.88429373+0.j -0.88137312+0.j]
rho: 0.8843

=== Matirx AG ===
(900, 900)
[-0.78197767+0.j -0.77681949+0.j -0.77562298+0.j]
rho: 0.782



====================
../data/outUnsteady/PP5/IV/condTest-out_B0_Tri_RecMatrix.dnds.h5
<HDF5 group "/VR_Matrix/cell2cellFace/father" (6 members)>
b'ArrayEigenUniMatrixBatch__-1_-1'
A Cond Max:   1.37e+01
max Sym deviation 9.11e-15

=== Matirx A ===
(2268, 2268), nnz is 81648
[0.80273178+0.j 0.84846799+0.j 0.95803417+0.j]
[117.06383457+0.j 106.77486864+0.j 105.23184932+0.j]
Cond Est:  1.458e+02 ||| rho: 117.1

=== Matirx D^-1 A ===
(2268, 2268), nnz is 81648
[0.1134969 +0.j 0.11403833+0.j 0.11548658+0.j]
[1.89787809+0.j 1.89377028+0.j 1.89315693+0.j]
Cond Est:  1.672e+01 ||| rho: 1.898

=== Matirx AJ ===
(2268, 2268), nnz is 61236
[0.89787809+0.j 0.89377025+0.j 0.89315666+0.j]
rho: 0.8979

=== Matirx AG ===
(2268, 2268)
[-0.7859418 +0.j -0.78509899+0.j -0.78246434+0.j]
rho: 0.7859



====================
../data/outUnsteady/PP5/IV/condTest-out_B0_Tri_Pert0_RecMatrix.dnds.h5
<HDF5 group "/VR_Matrix/cell2cellFace/father" (6 members)>
b'ArrayEigenUniMatrixBatch__-1_-1'
A Cond Max:   3.94e+01
max Sym deviation 9.81e-15

=== Matirx A ===
(2268, 2268), nnz is 81648
[0.1801201 +0.j 0.28851126+0.j 0.38374061+0.j]
[259.07923753+0.j 256.42537289+0.j 196.5516228 +0.j]
Cond Est:  1.438e+03 ||| rho: 259.1

=== Matirx D^-1 A ===
(2268, 2268), nnz is 81648
[0.08231148+0.j 0.0858822 +0.j 0.08789729+0.j]
[1.92608196+0.j 1.92245762+0.j 1.91878384+0.j]
Cond Est:  2.340e+01 ||| rho: 1.926

=== Matirx AJ ===
(2268, 2268), nnz is 61236
[0.92608196+0.j 0.92245762+0.j 0.91878359+0.j]
rho: 0.9261

=== Matirx AG ===
(2268, 2268)
[-0.84215884+0.j -0.83560892+0.j -0.83198669+0.j]
rho: 0.8422
```

- IV10_10.cgns 1.8522952e-03, 5.6943971e-02
  - condA: 339.4
  - condDIA: 14.97
  - rhoAJ: 0.8748
  - rhoAG: 0.7652
- IV10_Square_Pert0.cgns 2.0251535e-03, 6.8872222e-02
  - condA: 609.8
  - condDIA: 16.28
  - rhoAJ: 0.8843
  - rhoAG: 0.782
- IV10_Tri.cgns 3.2207710e-04, 2.3019284e-02
  - condA: 145.8
  - condDIA: 16.72
  - rhoAJ: 0.8979
  - rhoAG: 0.7859
- IV10_Tri_Pert0.cgns 5.9789200e-04, 3.1945860e-02
  - condA: 1438
  - condDIA: 23.4
  - rhoAJ: 0.9261
  - rhoAG: 0.8422