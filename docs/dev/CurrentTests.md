# Current Tests Conducted on DNDSR {#current_tests}

**[PDF Version](CurrentTests.pdf)**

## Current Implementation Status

Applications: euler, eulerSA, euler3D, eulerSA3D
SA part not yet tested here.

Reconstruction: newly written, currently only xy-basis and xy-functional.


## Test: euler: DMR with CWBAP

h=1/240 grid, PJH's derivative weights and length references. HLLEP from WZH as inviscid flux. Same parameters as WZH's paper.

Without smooth indicator:
![DMR](images/DMR.png)

With smooth indicator:
![DMR_SD](images/DMR_SD.png)

## Test: euler: DMR-480 performance

Test 10 steps of LU-SGS on TH2B, 4-nodes (96 cores):

![PerfDM480](images/PerfDM480.png)

## Test: euler3D: 128^3 calculation

3D calculation takes much more time than 2D with same cell number.

$[0,1]^3$ box, with initial value $[0.5,1,0,0,4]$ in $[0.25,0.75]^3$, and $[1,0,0,0,2.5]$ otherwise.
Inviscid wall on box faces. Use $128^3$ grid.

t=0.3:

Isosurface of $\rho = 0.654$
![Box128ISO](images/Box128ISO.png)

Vortex Core:
![Box128Vort](images/Box128Vort.png)


## Lines of Code

![Box128Vort](images/Lines.png)

