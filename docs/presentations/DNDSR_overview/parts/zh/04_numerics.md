---
<!-- _class: chapter -->
<!-- _paginate: false -->

<div class="ch-num">第4章</div>

# 数值方法

## CFV · VR · Flux · 限制器 · ODE · Krylov

---
<!-- _footer: "src/CFV/VRDefines.hpp:27 · docs/theory/Variational_Reconstruction.md:21-30" -->

## Compact Finite Volume — 重构

从单元均值出发，在每个单元上以**零均值基**重构分段多项式：

$$
u_i(\mathbf{x})
= \overline{u}_i
+ \sum_{l=1}^{N_{\text{base}}} u^l_i\, \varphi^l_i(\mathbf{x})
$$

- $\overline{u}_i$ — 单元均值，存储在 `tUDof<N> = ArrayDof<N,1>` 中。
- $u^l_i$ — 重构系数，存储在 `tURec<N> = ArrayDof<Dyn,N>` 中。
- 基 $\varphi^l_i$ 局部正交化，按单元尺度归一化；每个单元的多项式阶数在运行时选择。

**支持的多项式阶数：1 – 3**（线性、二次、三次）。

```cpp
// Static capacities (src/CFV/VariationalReconstruction.hpp:1051-1054)
maxRecDOFBatch = (dim == 2) ?  4 : 10;
maxRecDOF      = (dim == 2) ?  9 : 19;
maxNDiff       = (dim == 2) ? 10 : 20;
maxNeighbour   = 7;
```

模板为**一圈节点邻居**——这正是 `BuildGhostPrimary(1)` 默认提供的内容。更宽的模板（`nGhostLayers ≥ 2`）可用于高阶变体。

---
<!-- _footer: "docs/theory/Variational_Reconstruction.md:33-106" -->
<!-- _class: dense -->

## Variational Reconstruction — 泛函

最小化每个面上**所有至 k 阶导数**的跳跃：

$$
I_f
= w_g(f)\int_f \sum_{p=0}^{k}
  w_d(p)^2\,
  \bigl\|\mathcal{D}_p u_L - \mathcal{D}_p u_R\bigr\|_{\langle\,,\,\rangle_{f,p}}^{2}\, d\Gamma
$$

<div class="cols">
<div>

**权重**

- $w_g(f)$ — 几何权重；默认 $w_g = S_f^{-1}$（面积倒数）。
- $w_d(p)$ — 无量纲导数权重；控制各阶导数的贡献强度。
- $\mathcal{D}_p u$ — 第 $p$ 阶导数张量（仅在坐标线性变化下保持协变）。

**局部系统**

$$
A^{i}_{mn} u^{n}_{i}
= \sum_{j \in S_i}
  \bigl(B^{i{\leftarrow}j}_{mn} u^{n}_{j}
        + b^{i{\leftarrow}j}_{m}(\overline{u}_j-\overline{u}_i)\bigr)
$$

迭代求解——方案见下文。

</div>
<div>

**三种内积选择**

- **Wang（法向）：** $\langle\mathcal{D}_3 u,\mathcal{D}_3 v\rangle = d_f^6\,\partial_{nnn}u\,\partial_{nnn}v$
- **Pan（X-Y 对齐）：** $\sum (\Delta_x^a\Delta_y^b\partial^{\cdot}_{xy} u)\, (\Delta_x^a\Delta_y^b\partial^{\cdot}_{xy} v)$
- **Huang（预各向同性）：** $d_f^{2p}$ 加权，方向各向同性。

**重构迭代方案**（`VariationalReconstruction.hpp:938-1031`）

- `DoReconstructionIter` — Jacobi / SOR 扫描（测试使用 Jacobi）。
- `DoReconstructionIterDiff` — Jacobian-向量乘积（GMRES 内层）。
- `DoReconstructionIterSOR` — SOR，可选反向扫描。
- 回退方案：`DoReconstruction2nd`、`DoReconstruction2ndGrad`。

</div>
</div>

---
<!-- _footer: "src/CFV/VariationalReconstruction.hpp:282-289" -->
<!-- _class: denser -->

## VR 设置 — 三个 `Construct*` 调用

```cpp
template <int dim = 2>
class VariationalReconstruction : public FiniteVolume {
public:
    void ConstructMetrics();                                                      // 通过 FiniteVolume
    void ConstructBaseAndWeight(tFGetBoundaryWeight id2faceDircWeight = …);      // 基 + 缓存的导数值
    void ConstructRecCoeff();                                                    // A, B, A^{-1} B, 辅助矩阵
    // …
};
```

<div class="cols">
<div>

### `ConstructMetrics` 构建的内容

- 单元体积、面面积、单位法向量、求积 Jacobian。
- 惯性张量、主轴坐标系、包围盒尺度。
- 每个求积点的物理坐标。
- 每个单元的光滑性尺度。

### `ConstructBaseAndWeight` 构建的内容

- `cellBaseMoment` — 每个单元的基矩。
- `faceAlignedScales`、`faceMajorCoordScale`。
- `cellDiffBaseCache`、`faceDiffBaseCache` — 所有求积点上、模板中每个邻居的缓存导数值。
- `bndVRCaches` — 用于 BC 加权 VR 的边界面缓存。

</div>
<div>

### `ConstructRecCoeff` 构建的内容

- `matrixAB`、`vectorB` — 每个邻居的右端项块。
- `matrixAAInvB`、`vectorAInvB` — 预计算的 $A^{-1}B$，用于加速 Jacobi / SOR 迭代。
- `matrixSecondary`、`matrixAHalf_GG` — 辅助重构系统。
- `matrixA`、`matrixACholeskyL`、`volIntCholeskyL` — 完整系统 + 稠密局部求解的 Cholesky 分解。

所有数组均为 `ArrayEigenMatrix*` 或 `ArrayEigenUniMatrixBatch*`——即在支持 MPI 的分布式内存块上的 Eigen 映射。

</div>
</div>

---
<!-- _footer: "src/CFV/FiniteVolume.hpp:38-86" -->
<!-- _class: dense -->

## `FiniteVolume` — 度量缓存

```cpp
class FiniteVolume : public DeviceTransferable<FiniteVolume> {
    real sumVolume, minVolume{veryLargeReal}, maxVolume, volGlobal;

    tScalarPair  volumeLocal;         // 每个单元的 volume
    tScalarPair  faceArea;            // 每个面的 area
    tRecAtrPair  cellAtr,  faceAtr;   // (NDOF, NDIFF, Order, intOrder)
    tCoeffPair   cellIntJacobiDet, faceIntJacobiDet;
    t3VecsPair   faceUnitNorm;        // 每个面求积点处的 normal
    t3VecPair    faceMeanNorm;
    t3VecPair    cellBary,  faceCent,  cellCent;
    t3VecsPair   cellIntPPhysics, faceIntPPhysics;
    t3VecPair    cellAlignedHBox, cellMajorHBox;
    t3MatPair    cellMajorCoord, cellInertia;
    tScalarPair  cellSmoothScale;

    int axisSymmetric = 0;            // 楔形轴对称
    std::set<index> axisFaces;

    // CRTP: to_device(), to_host(), device(), deviceView<B>()
};
```

<div class="callout callout-ok">

**支持 CUDA 传输。** `FiniteVolume`（以及 `VariationalReconstruction`）继承自 `DeviceTransferable<FiniteVolume>`。调用一次 `fv.to_device()` 即可将整个度量缓存迁移到 GPU 作为设备端视图。

</div>

---
<!-- _footer: "src/Euler/Gas.hpp:61-95,230" -->
<!-- _class: tight -->

## 13 种 Riemann 求解器

```cpp
enum RiemannSolverType {
    UnknownRS = 0,
    Roe       = 1, HLLC     = 2, HLLEP    = 3, HLLEP_V1 = 21,
    Roe_M1    = 11, Roe_M2  = 12, Roe_M3  = 13, Roe_M4  = 14, Roe_M5 = 15,
    Roe_M6    = 16, Roe_M7  = 17, Roe_M8  = 18, Roe_M9  = 19,
};
```

| 变体 | 熵修正 / 特征值策略 |
|---------|---------------------------------|
| `Roe`    | 标准 Roe + Harten–Yee |
| `Roe_M1` | cLLF（中心 + 局部 Lax–Friedrichs） |
| `Roe_M2` | Lax–Friedrichs |
| `Roe_M3` | LD Roe（低耗散） |
| `Roe_M4` | ID Roe（中等耗散） |
| `Roe_M5` | LD cLLF |
| `Roe_M6` | 仅 H-修正 |
| `Roe_M7` | 仅 Harten–Yee，无 H-修正 |
| `Roe_M8` | H-修正 + Harten–Yee |
| `Roe_M9` | 保留（eigScheme 9，当前 assert false） |
| `HLLC`   | Harten–Lax–van Leer–Contact |
| `HLLEP`  | HLLE，带压力修正 |
| `HLLEP_V1` | HLLEP 变体 1 |

```cpp
// 共享辅助函数
template <int dim>
RoePreamble<dim> ComputeRoePreamble(ULm, URm, gamma, dumpInfo);
```

---
<!-- _footer: "src/Euler/Gas.hpp:200-230" -->
<!-- _class: dense -->

## `RoePreamble` — 共享中间层

```cpp
template <int dim>
struct RoePreamble {
    TVec veloLm, veloRm;                     // 原始速度
    real rhoLm, rhoRm, pLm, pRm, HLm, HRm;   // 原始状态
    real veloLm0, veloRm0;                   // 法向速度分量

    TVec veloRoe;                            // Roe 平均速度
    real sqrtRhoLm, sqrtRhoRm;
    real vsqrRoe, HRoe, asqrRoe, rhoRoe, aRoe;
};
```

<div class="cols">
<div>

### Flux 签名

```cpp
template <int dim, int eigScheme>
void RoeFlux(UL, UR, ULm, URm, n, vgN,
             /*out*/ flux,
             /*out*/ dLambda,
             fixScale, gamma, dumpInfo);

template <int dim, int type>
void HLLEPFlux_IdealGas(UL, UR, ULm, URm, n, vgN,
                        flux, …, gamma, dumpInfo);

template <int dim>
void HLLCFlux(UL, UR, ULm, URm, n, vgN, …);
```

</div>
<div>

### 为什么要这样分解

所有 13 种变体共享 `ComputeRoePreamble`——Roe 平均、$H_{\text{Roe}}$、$a_{\text{Roe}}$ 等。随后由 `eigScheme` 模板参数选择耗散/熵修正策略。

- **每个 (`dim`, `eigScheme`) 一次模板实例化**，保持代码体积可控。
- **编译期分发**——Flux 核函数中无间接调用。
- **统一接口**，适用于无粘及完整 Navier-Stokes Flux：`NSFluxInvis<dim>`、`NSFluxVis<dim>(U, gradU, T, mu, n, flux, adiabaticWall, useQCR)`。

</div>
</div>

---
<!-- _footer: "src/CFV/Limiters.hpp:28-577" -->
<!-- _class: dense -->

## 限制器 — FWBAP L2 系列

<div class="cols">
<div>

### 多方向（≥ 2 方向）

- `FWBAP_L2_Multiway` — 通用 Eigen 数组。
- `FWBAP_L2_Multiway_Polynomial2D` — 2D 多项式加权范数。
- `FWBAP_L2_Multiway_PolynomialOrth` — 正交变体。
- `FMEMM_Multiway_Polynomial2D` — 修正极值-单调混合器。

**幂参数：** `p = 4`；`verySmallReal_pDiP = std::pow(verySmallReal, 1.0/p)` 稳定化零点附近的值。

</div>
<div>

### 双向（成对）

- `FWBAP_L2_Biway`
- `FWBAP_L2_Cut_Biway` — 符号截断
- `FMINMOD_Biway`
- `FVanLeer_Biway`
- `FWBAP_L2_Biway_PolynomialNorm<dim, nVarsFixed>`
- `FMEMM_Biway_PolynomialNorm<dim, nVarsFixed>`
- `FWBAP_L2_Biway_PolynomialOrth`

**Configuration**

```jsonc
"limiterProcedure":  0   // WBAP (V2)
"limiterProcedure":  1   // CWBAP (V3)  ← 推荐
"usePPRecLimiter":   true
```

</div>
</div>

> **正性保持**——`LimiterUGrad`（Euler 侧）钳制梯度；`EvaluateURecBeta` 强制重构值的单元均值正性；`EvaluateCellRHSAlpha` 强制 CFL 一致的单单元右端项缩放。

---
<!-- _footer: "src/CFV/VariationalReconstruction.hpp:1071-1086" -->
<!-- _class: dense -->

## VR 内置限制器 — 带特征变换的 WBAP

```cpp
template <int nVarsFixed>
void DoLimiterWBAP_C(tUDof<nVarsFixed>  &u,
                     tURec<nVarsFixed>  &uRec,
                     tURec<nVarsFixed>  &uRecNew,
                     tURec<nVarsFixed>  &uRecBuf,
                     tSmoothIndicator   &si,
                     bool                ifAll,
                     tFM   FM,                  // 守恒 → 特征变换
                     tFMI  FMI,                 // 特征 → 守恒变换
                     bool  putIntoNew = false);

template <int nVarsFixed>
void DoLimiterWBAP_3(...);                      // 三模态变体
```

<div class="cols">
<div>

### 流程

1. 计算每个面的光滑性指示器 `si`。
2. 将重构系数变换到特征变量（`FM`）。
3. 逐特征地、跨多方向邻域应用 WBAP 限制器。
4. 变换回来（`FMI`）。
5. 可选地写入 `uRecNew`（迭代方案的双缓冲）。

</div>
<div>

### 光滑性指示器

- `DoCalculateSmoothIndicator<nVarsFixed, nVarsSee=2>(si, uRec, u, varsSee)` — 变量子集上的经典指示器。
- `DoCalculateSmoothIndicatorV1<nVarsFixed>(si, uRec, u, varsSee, FPost)` — V1，支持用户提供的后处理。

</div>
</div>

---
<!-- _footer: "src/Solver/ODE.hpp · RELEASE_NOTES.md:11,14" -->
<!-- _class: dense -->

## 时间积分 — ODE 系列

所有积分器继承自：

```cpp
template <class TDATA, class TDTAU>
class ImplicitDualTimeStep {
    using Frhs     = std::function<void(TDATA&, TDATA&, TDTAU&, int, real, int)>;
    using Fdt      = std::function<void(TDATA&, TDTAU&, real, int)>;
    using Fsolve   = std::function<void(TDATA&, TDATA&, TDATA&, TDTAU&, real, real, TDATA&, int, real, int)>;
    using Fstop    = std::function<bool(int, TDATA&, int)>;
    using Fincrement = std::function<void(TDATA&, TDATA&, real, int)>;
    virtual void Step(TDATA &x, TDATA &xinc, const Frhs&, const Fdt&, const Fsolve&,
                      int maxIter, const Fstop&, const Fincrement&, real dt) = 0;
};
```

| `odeCode` | 类                                                | 格式                   |
|-----------|---------------------------------------------------|-----------------------|
| `103`     | `ImplicitEulerDualTimeStep`                       | 后向 Euler             |
| `0`       | `ImplicitBDFDualTimeStep`                         | BDF2 / BDF-k          |
| —         | `ImplicitVBDFDualTimeStep`                        | 变步长 BDF-k           |
| `1`       | `ImplicitSDIRK4DualTimeStep` (`schemeCode` 0…4)   | SDIRK-4 · ESDIRK2/3 · 梯形 |
| `101`     | (`1` 的别名)                                       | （向后兼容 `odeCode`） |
| **`401`** | `ImplicitHermite3SimpleJacobianDualStep`          | **HM3 + p-Multigrid** |
| `2`       | `ExplicitSSPRK3TimeStepAsImplicitDualTimeStep`    | SSP-RK3               |

`SetExtraParams(json)` 暴露特定格式的参数（如 `nMG`、`incFScale`）。

---
<!-- _footer: "src/Solver/ODE.hpp:123-363,917-1438" -->
<!-- _class: denser -->
## HM3 + p-Multigrid

**HM3**（Hermite-3）是一种三阶 A-稳定隐式格式，有三种模式：

- **U2R2** — 2 个解状态 + 2 个残差状态。
- **U2R1** — 2 个解状态 + 1 个残差状态。
- **U3R1** — 3 个解状态 + 1 个残差状态。

<div class="cols">
<div>

### 时间步内的 p-MG

在 `ImplicitHermite3SimpleJacobianDualStep::Step()` 内部，**非零 `nMG`** 触发 p-Multigrid 光滑循环：

```cpp
// 内层求解中的伪代码（第1250-1251行）
fdt (xMG, dTau, 1.0, /*upos=*/2);      // 低阶伪时间步
frhs(rhsbuf[1], xMG, dTau, iter, 1.0, /*upos=*/2);
```

`upos=2` 参数告诉求值器在**较低多项式阶**上进行求值（层级过渡）。VR 提供 `DownCastURecOrder(curOrder, iCell, uRec, downCastMethod)` 在不同阶之间投影重构系数。

</div>
<div>

### 配套功能

- **`tpMG`** — 外部双时间循环中多重网格的开关。
- **`incFScale`** — 较低 MG 层级上的增量 Flux 缩放；已集成至熵修正路径（`RELEASE_NOTES.md`）。
- **`LimiterUGrad` 中的正性保持限制器**——防止低阶粗网格修正产生负密度/压力。

### 其他 `SDIRK4` 编码

- `schemeCode = 0` — Nørsett 3 级 SDIRK-4
- `schemeCode = 1` — 6 级 ARK 族 SDIRK
- `schemeCode = 2` — Kennedy–Carpenter ESDIRK3
- `schemeCode = 3` — 梯形
- `schemeCode = 4` — ESDIRK2, `γ = 1 − √2/2`

</div>
</div>

---
<!-- _footer: "src/Solver/Linear.hpp · src/Euler/EulerEvaluator.hpp:427-580" -->
<!-- _class: dense -->

## 线性求解器 — Krylov + LU-SGS 预条件子

<div class="cols">
<div>

### Krylov 方法

```cpp
template <class TDATA>
class GMRES_LeftPreconditioned {
public:
    GMRES_LeftPreconditioned(index dofSize);
    void setSpace(int kSpace);
    bool solve(const TDATA &rhs, TDATA &x,
               FMatVec Ax, FPCApply PC,
               int maxIter, real tol);
};

template <class TDATA, class TScalar>
class PCG_PreconditionedRes { … };
```

无矩阵：调用方提供 `Ax` 和 `PC` 函子。

</div>
<div>

### 无矩阵 LU-SGS 预条件子

由 `EulerEvaluator` 提供：

```cpp
void LUSGSMatrixInit(JDiag, JSource, dTau, dt, alphaDiag, u, uRec, jacCode, t);
void LUSGSMatrixVec(alphaDiag, t, u, uInc, JDiag, AuInc);
void LUSGSMatrixToJacobianLU(alphaDiag, t, u, JDiag, jacLU);
void UpdateSGS(alphaDiag, t, rhs, u, uInc, uIncNew, JDiag,
               forward, gsUpdate, sumInc, uIncIsZero = false);
void LUSGSMatrixSolveJacobianLU(alphaDiag, t, rhs, u, uInc, uIncNew,
                                bBuf, JDiag, jacLU,
                                uIncIsZero, sumInc);
void UpdateSGSWithRec(alphaDiag, t, rhs, u, uRec, uInc, uRecInc,
                      JDiag, forward, sumInc);
```

</div>
</div>

**选择器**

```jsonc
"gmresCode": 0  // 仅 LUSGS      （廉价、鲁棒）
"gmresCode": 1  // GMRES          （无矩阵 Krylov）
"gmresCode": 2  // LUSGS + GMRES  （LUSGS 作为 GMRES 预条件子）
```

小块的直接路径：`src/Solver/Direct.hpp`（LU / LDLT）。可通过 `cfd_externals` 子模块启用可选的 **SuperLU_dist**。
