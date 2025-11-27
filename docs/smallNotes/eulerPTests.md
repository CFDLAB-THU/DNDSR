# EulerP Tests

## Periodic shock box

Test solver. 2nd order + Barth limiter + SSPRK3, CFL=0.5

<video controls>
  <source src="https://harryzhou2000.github.io/resources-0/eulerPtests/periodic_box_test3.mp4" type="video/mp4">
</video>

Number of cells iterations per second, CI/s

Using 1024^2 cells.

| Machine      | Performance (CI/s) | Power estimated ( by software) (W)                   | Efficiency (MCI/kJ) |
| ------------ | ------------------ | ---------------------------------------------------- | ------------------- |
| 1 A100       | 7.5M               | 120 (GPU) +  175 (CPU Package) + 25 (RAM)   = 320    | 23.4 MCI/kJ         |
| 2 A100       | 14.7M              | 120 (GPU) * 2 +  175 (CPU Package) + 25 (RAM)  = 440 | 33.4 MCI/kJ         |
| 32 CPU cores | 4.2M               | 360 (GPU)  (CPU Package) + 45 (RAM)  = 405           | 10.37 MCI/kJ        |

Using 16 OMP thread x 2 ranks performs nearly the same as (slightly worse than) 32 ranks.

**GPU python profile results**
![cProfile_GPU](https://harryzhou2000.github.io/resources-0/eulerPtests/periodicBox1024/cProfile_GPU.png)

**CPU python profile results**
![cProfile_CPU](https://harryzhou2000.github.io/resources-0/eulerPtests/periodicBox1024/cProfile_CPU.png)

**NSYS results**
![cProfile_GPU](https://harryzhou2000.github.io/resources-0/eulerPtests/periodicBox1024/nsys_1rank.png)
