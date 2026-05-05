---
<!-- _class: chapter -->
<!-- _paginate: false -->

<div class="ch-num">CHAPTER 9</div>

# Results

## Selected 2D / 3D Euler & NS cases

---
<!-- _class:  -->

## NACA 0012 (SA RANS)

<div class="cols">

<div>

![NACA mesh](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/01_naca0012_mesh.png)

*mesh*

</div>

<div>

![Mach AOA=5°](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/01_naca0012_mach_aoa5.png)

*AOA = 5°, Mach number*

</div>

</div>

---
<!-- _class:  -->

## NACA 0012 — AOA 15°

<div class="cols">

<div>

![Mach AOA=15°](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/01_naca0012_mach_aoa15.png)

*AOA = 15°, Mach number*

</div>

<div>

![ν̃ field](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/01_naca0012_nutilde.png)

*AOA = 15°, $\rho\tilde{\nu}$ field*

</div>

</div>

---

## Double Mach Reflection (2D Euler)

Density at *t* = 0.2, Mach 10 shock on 30° wedge.

<div class="cols">

<div>

![DITR U2R2](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/02_dm480_hm3_field.png)

*DITR U2R2, $c_2 = 0.55$*

</div>

<div>

![BDF2](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/02_dm480_bdf2.png)

*BDF2*

</div>

</div>

---
<!-- _class:  -->

## Supersonic Wedge (2D Euler)

Mach 3 inviscid flow over a 15° compression corner.

<div class="cols">

<div>

![Wedge mesh](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/12_wedge_mesh.png)

*mesh*

</div>

<div>

![Wedge density](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/12_wedge_density.png)

*Density, Re $= \infty$*

</div>

</div>

---
<!-- _class:  -->

## Wedge — viscous (Re = 100) + pressure

<div class="cols">

<div>

![Wedge pressure](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/12_wedge_pressure.png)

*Pressure, Re $= \infty$*

</div>

<div>

![Wedge viscous](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/12_wedge_viscous_density.png)

*Density, Re = 100*

</div>

</div>

---
<!-- _class:  -->

## Hypersonic Cavity (2D Navier–Stokes)

Time-averaged results.

<div class="cols">

<div>

![Cavity mesh](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/03_cavity_mesh.png)

*mesh*

</div>

<div>

![Cavity pressure](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/03_cavity_pressure.png)

*Time-averaged $p/(\rho_\infty U_\infty^2)$, 27 isolines*

</div>

</div>

---
<!-- _class:  -->

## Hypersonic Cavity — time step + Ma = 0.1

<div class="cols">

<div>

![Cavity dt comparison](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/03_cavity_dt_compare.png)

*Explicit time step comparison*

</div>

<div>

![Cavity Ma=0.1](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/03_cavity_mach01.png)

*t = 0.1 Mach, 32 isolines (explicit 2nd-order FV)*

</div>

</div>

---
<!-- _class:  -->

## Cylinder Re = 1200 (2D NS, unsteady)

<div class="cols">

<div>

![Cyl mesh](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/05_cylinder_re1200_mesh.png)

*Mesh*

</div>

<div>

![Cyl vorticity](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/05_cylinder_re1200.png)

*Vorticity*

</div>

</div>

---
<!-- _class:  -->

## 3D Turbulent Cylinder (ILES, Re = 1.2 × 10³)

Q-criterion iso-surfaces coloured by Mach number.

<div class="cols">

<div>

![3D cylinder mesh](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/13_cylinder3d_mesh.png)

*mesh*

</div>

<div>

![Q-criterion, ΔT=0.05](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/13_cylinder3d_qcrit.png)

*Q-criterion iso-surfaces*

</div>

</div>

---

## Sedov Blast Wave (2D Euler)

<div class="cols">

<div>

![Sedov density](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/08_sedov.png)

*Density*

</div>

<div>

![Sedov profile](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/08_sedov_profile.png)

*Density along diagonal*

</div>

</div>

---

## Noh Problem

<div class="cols">

<div>

![Noh density](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/09_noh.png)

*Density at t = 0.6*

</div>

<div>

![Noh profile](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/09_noh_profile.png)

*Density along diagonal*

</div>

</div>

---
<!-- _class:  -->

## M2000 Jet (2D Euler)

Mach 2000 jet

<div class="cols">

<div>

![M2000 pressure](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/10_m2000jet_pressure.png)

*$\log_{10}(p)$, Re = 100*

</div>

<div>

![M2000 density](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/10_m2000jet_density.png)

*$\log_{10}(\rho)$, Re = 100*

</div>

</div>

---
<!-- _class:  -->

## Isentropic Vortex (2D Euler) — error convergence

<div class="cols">

<div>

![IV L1 error](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/07_iv_err.png)

*$L^1$ density error vs. grid size*

</div>

<div>

![IV residual](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/07_iv_res.png)

*Residual convergence*

</div>

</div>

---

## NASA CRM (3D SA RANS)

Wing-body

<div class="cols">

<div>

![CRM mesh](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/11_crm_mesh.png)

*Surface mesh*

</div>

<div>

![CRM Cp](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/11_crm_cp.png)

*Surface $C_p$*

</div>

</div>

---
<!-- _class:  -->

## NASA CRM — $C_f$ + $C_L/C_D$

<div class="cols">

<div>

![CRM Cf](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/11_crm_cf.png)

*Surface $C_f$*

</div>

<div>

![CRM CLCD](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/11_crm_clcd.png)

*Force-coefficient convergence*

</div>

</div>
