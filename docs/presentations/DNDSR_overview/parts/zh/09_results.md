---
<!-- _class: chapter -->
<!-- _paginate: false -->

<div class="ch-num">第 9 章</div>

# 结果

## 精选 2D / 3D Euler 与 NS 算例

---
<!-- _class:  -->

## NACA 0012 (SA RANS)

<div class="cols">

<div>

![NACA mesh](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/01_naca0012_mesh.png)

*网格*

</div>

<div>

![Mach AOA=5°](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/01_naca0012_mach_aoa5.png)

*AOA = 5°, 马赫数*

</div>

</div>

---
<!-- _class:  -->

## NACA 0012 — AOA 15°

<div class="cols">

<div>

![Mach AOA=15°](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/01_naca0012_mach_aoa15.png)

*AOA = 15°, 马赫数*

</div>

<div>

![ν̃ field](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/01_naca0012_nutilde.png)

*AOA = 15°, $\rho\tilde{\nu}$ 场*

</div>

</div>

---
## 双马赫反射 (2D Euler)

*t* = 0.2 时的密度，马赫 10 激波在 30° 楔上。

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

## 超音速楔 (2D Euler)

马赫 3 无粘流，15° 压缩拐角。

<div class="cols">

<div>

![Wedge mesh](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/12_wedge_mesh.png)

*网格*

</div>

<div>

![Wedge density](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/12_wedge_density.png)

*密度，Re $= \infty$*

</div>

</div>

---
<!-- _class:  -->

## 楔 — 粘性 (Re = 100) + 压力

<div class="cols">

<div>

![Wedge pressure](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/12_wedge_pressure.png)

*压力，Re $= \infty$*

</div>

<div>

![Wedge viscous](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/12_wedge_viscous_density.png)

*密度，Re = 100*

</div>

</div>

---
<!-- _class:  -->

## 高超声速空腔 (2D Navier–Stokes)

时间平均效果：

<div class="cols">

<div>

![Cavity mesh](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/03_cavity_mesh.png)

*网格*

</div>

<div>

![Cavity pressure](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/03_cavity_pressure.png)

*时间平均 $p/(\rho_\infty U_\infty^2)$，27 条等值线*

</div>

</div>

---
<!-- _class:  -->

## 高超声速空腔 — 时间步长 + Ma = 0.1

<div class="cols">

<div>

![Cavity dt comparison](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/03_cavity_dt_compare.png)

*显式时间步长对比*

</div>

<div>

![Cavity Ma=0.1](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/03_cavity_mach01.png)

*马赫 0.1，32 条等值线（显式 2 阶 FV）*

</div>

</div>

---
<!-- _class:  -->

## 圆柱 Re = 1200 (2D NS, 非定常)

<div class="cols">

<div>

![Cyl mesh](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/05_cylinder_re1200_mesh.png)

*网格*

</div>

<div>

![Cyl vorticity](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/05_cylinder_re1200.png)

*涡量*

</div>

</div>

---
<!-- _class:  -->

## 3D 湍流圆柱 (ILES, Re = 1.2 × 10³)

Q-criterion 等值面，按马赫数着色。

<div class="cols">

<div>

![3D cylinder mesh](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/13_cylinder3d_mesh.png)

*网格*

</div>

<div>

![Q-criterion, ΔT=0.05](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/13_cylinder3d_qcrit.png)

*Q-criterion 等值面*

</div>

</div>

---
## Sedov 爆炸波 (2D Euler)

<div class="cols">

<div>

![Sedov density](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/08_sedov.png)

*密度*

</div>

<div>

![Sedov profile](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/08_sedov_profile.png)

*对角线上的密度*

</div>

</div>

---
## Noh 问题

<div class="cols">

<div>

![Noh density](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/09_noh.png)

*t = 0.6 时的密度*

</div>

<div>

![Noh profile](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/09_noh_profile.png)

*对角线上的密度*

</div>

</div>

---
<!-- _class:  -->

## M2000 射流 (2D Euler)

马赫 2000 射流

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

## 等熵涡 (2D Euler) — 误差收敛

<div class="cols">

<div>

![IV L1 error](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/07_iv_err.png)

*$L^1$ 密度误差随网格尺寸变化*

</div>

<div>

![IV residual](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/07_iv_res.png)

*残差收敛*

</div>

</div>

---
## NASA CRM (3D SA RANS)

翼身组合体

<div class="cols">

<div>

![CRM mesh](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/11_crm_mesh.png)

*表面网格*

</div>

<div>

![CRM Cp](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/11_crm_cp.png)

*表面 $C_p$*

</div>

</div>

---
<!-- _class:  -->

## NASA CRM — $C_f$ + $C_L/C_D$

<div class="cols">

<div>

![CRM Cf](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/11_crm_cf.png)

*表面 $C_f$*

</div>

<div>

![CRM CLCD](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/demos/11_crm_clcd.png)

*力系数收敛*

</div>
