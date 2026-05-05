---
<!-- _class: chapter -->
<!-- _paginate: false -->

<div class="ch-num">第7章</div>

# 求解器与验证

## Euler 系列 · EulerEvaluator · 算例

---
<!-- _footer: "src/Euler/Euler.hpp:874-905 · app/Euler/*.cpp" -->
<!-- _class: dense -->

## Euler / N-S 系列 — 每种模型一个可执行文件

```cpp
// app/Euler/euler.cpp — the entire file
#include "EulerSolver.hpp"
int main(int argc, char *argv[]) {
    return DNDS::Euler::RunSingleBlockConsoleApp<
               DNDS::Euler::NS>(argc, argv);
}
```

<div class="cols">
<div>

### 模型枚举 (`EulerModel`)

```cpp
enum EulerModel {
    NS,        // 2D Navier-Stokes
    NS_3D,
    NS_SA,     // 2D Spalart-Allmaras
    NS_SA_3D,
    NS_2D,     // alias for NS
    NS_2EQ,    // k-omega two-equation
    NS_2EQ_3D,
    NS_EX,     // reactive / multi-species
    NS_EX_3D,
};
```

通过 `EulerModel` 模板派发，每个求解器生成一个可执行文件 — 共享源码，分离目标文件。

</div>
<div>

### RANS 模型枚举

```cpp
enum RANSModel {
    RANS_None,
    RANS_SA,       // Spalart-Allmaras (IDDES capable)
    RANS_KOWilcox, // Wilcox k-omega
    RANS_KOSST,    // Menter k-omega SST
    RANS_RKE,      // Realizable k-epsilon
};
```

每种模型都有对应的 `RANSModelTraits<>` 特化，包含其壁面边界条件、源项和谱半径。

</div>
</div>

---
<!-- _footer: "src/Euler/EulerSolver.hpp:73-148" -->
<!-- _class: tight -->

## `EulerSolver` — 顶层调度器

Euler 模块扩展了 CFV 的通用 `tUDof`/`tURec` 别名，添加了
求解器专用的数组类型，提供更高级的算子（初始化、边界锚定、
保正限制器）：

- `ArrayDOFV<N>` 继承自 `CFV::tUDof<N>` (= `ArrayDof<N,1>`)。
- `ArrayRECV<N>` 继承自 `CFV::tURec<N>` (= `ArrayDof<DynamicSize,N>`)。

```cpp
template <EulerModel model>
class EulerSolver {
    typedef EulerEvaluator<model> TEval;
    static const int nVarsFixed = TEval::nVarsFixed;

    MPIInfo                              mpi;
    ssp<Geom::UnstructuredMesh>          mesh, meshBnd;
    TpVFV                                 vfv;              // VariationalReconstruction
    ssp<Geom::UnstructuredMeshSerialRW>  reader, readerBnd;
    ssp<EulerEvaluator<model>>           pEval;
    ssp<BoundaryHandler<model>>          pBCHandler;

    // Solver state (DOF arrays)
    ArrayDOFV<nVarsFixed>                u, uIncBufODE, wAveraged, uAveraged;
    ObjectPool<ArrayDOFV<nVarsFixed>>    uPool;             // rent/return buffers
    ArrayRECV<nVarsFixed>                uRec, uRecLimited, uRecNew, uRecNew1,
                                         uRecOld, uRec1, uRecInc, uRecInc1,
                                         uRecB, uRecB1;
    JacobianDiagBlock<nVarsFixed>        JD, JD1, JDTmp, JSource, JSource1, JSourceTmp;
    ssp<JacobianLocalLU<nVarsFixed>>     JLocalLU;
    ArrayDOFV<1>                         alphaPP, alphaPP1, betaPP, betaPP1,
                                         alphaPP_tmp, dTauTmp;

    // Config + output
    Configuration                        config;            // nested sub-configs
    nlohmann::ordered_json               gSetting;
    std::string                          output_stamp;
    // ... outDist* / outSerial* / outDist2SerialTrans* for VTK
};
```

---
<!-- _footer: "src/Euler/EulerSolver.hpp:160-246 · nested Configuration struct" -->
<!-- _class: dense -->

## `Configuration` — 调节运行的一切参数

每个小节都使用 `DNDS_DECLARE_CONFIG`，因此完整的 JSON Schema 会自动生成。

<div class="cols">
<div>

- **TimeMarchControl** — `dtImplicit`、`nTimeStep`、`steadyQuit`、`useRestart`、`useImplicitPP`、`odeCode`、`odeSetting1..4`、`odeSettingsExtra`（不透明 JSON）、`dtCFLLimitScale`、…
- **ImplicitReconstructionControl** — `useExplicit`、`nInternalRecStep`、`recLinearScheme`（0 = SOR, 1 = GMRES）、`nGmresSpace/Iter`、`fpcgReset*`、`recThreshold`。
- **OutputControl** — `outputIntervalStep`、`outputFormat`（VTK、PLT、VTKHDF、series）、并行与串行写入。
- **CFLControl** — 初始 / 最大 CFL、攀升计划。

</div>
<div>

- **ConvergenceControl** — 残差阈值、监控变量。
- **DataIOControl** — 读写路径、restart 检查点。
- **BoundaryDefinition** — 每面区域 BC 类型、自由来流状态。
- **LimiterControl** — `limiterProcedure`、`usePPRecLimiter`、WBAP 阶数。
- **LinearSolverControl** — `gmresCode`、Krylov 子空间、迭代次数。
- **TimeAverageControl** — 用于统计的长时间平均。
- **EvaluatorSettings** 包装 `EulerEvaluatorSettings<model>`。
- **VFVSettings** 包装 `VRSettings`。

</div>
</div>

> `--emit-schema` 将整个Configuration树导出为单个 JSON Schema 文档 — `euler_schema.json` / `eulerSA3D_schema.json` / 等，每个约 107 KB。

---
<!-- _footer: "src/Euler/EulerEvaluator.hpp:399-612" -->
<!-- _class: denser -->
## `EulerEvaluator<model>` — 空间算子

```cpp
void EvaluateRHS(ArrayDOFV<nVarsFixed>            &rhs,
                 JacobianDiagBlock<nVarsFixed>    &JSource,
                 ArrayDOFV<nVarsFixed>            &u,
                 ArrayRECV<nVarsFixed>            &uRecUnlim,
                 ArrayRECV<nVarsFixed>            &uRec,
                 ArrayDOFV<1>                     &uRecBeta,
                 ArrayDOFV<1>                     &cellRHSAlpha,
                 bool  onlyOnHalfAlpha,
                 real  t,
                 uint64_t flags = RHS_No_Flags);
```

<div class="cols">
<div>

### 标志位

- `RHS_Ignore_Viscosity`
- `RHS_Dont_Update_Integration`
- `RHS_Dont_Record_Bud_Flux`
- `RHS_Direct_2nd_Rec` — 绕过 VR，使用基于 GG 的二阶
- `RHS_Direct_2nd_Rec_1st_Conv` — 二阶重构但一阶对流
- `RHS_Direct_2nd_Rec_use_limiter`
- `RHS_Direct_2nd_Rec_already_have_uGradBufNoLim`
- `RHS_Recover_IncFScale`

标志位按位组合 — 覆盖 p-MG 和 PP 子步骤使用的回退/诊断模式。

</div>
<div>

### 其他顶层调用

- `EvaluateDt(...)` — 基于 CFL 的局部 dt，基于谱半径。
- `EvaluateURecBeta` — 每个单元的 PP 限制器 β。
- `EvaluateCellRHSAlpha` — 每个单元的 RHS 缩放，用于 PP。
- `LimiterUGrad` — 梯度限制器，可选激波检测。
- `LUSGSMatrixInit/Vec/ToJacobianLU` 和 `UpdateSGS(WithRec)`。
- 壁面距离：`GetWallDist_AABB`、`GetWallDist_BatchedAABB`、`GetWallDist_Poisson`。
- 粘性：`muEff(U, T)`，使用 Sutherland 或常数模型。

</div>
</div>

---
<!-- _footer: "src/Euler/BoundaryConditions/ · BoundaryHandler<model>" -->
<!-- _class: dense -->

## 边界条件 — 策略模式

每种 BC 都是一个实现通用接口的类；`BoundaryHandler<model>` 在运行时将面区域 ID 路由到 BC 实例。

| BC                   | 用途                                         |
|----------------------|---------------------------------------------|
| `BCWall`             | 无滑移壁面（绝热）                          |
| `BCWallIsothermal`   | 无滑移壁面，固定温度                        |
| `BCWallInvis`        | 滑移 / 对称                                 |
| `BCSym`              | 显式对称面                                  |
| `BCFarField`         | Riemann 不变量远场                          |
| `BCIn`               | 指定入口                                    |
| `BCOut` / `BCOutP`   | 指定出口 / 压力出口                         |
| `BCPeriodic`         | 标准周期                                    |
| `BCPeriodicRot`      | 旋转周期（叶轮机械）                        |
| `BCProfileIn`        | 表格化剖面（边界层, RANS）                  |
| `BCActuator`         | 致动盘源项                                  |

专用叶轮机械 BC：`BCTotalInlet`、`BCRadialEqOutlet`、`BCMixingPlane`，以及用于攻角自适应升力匹配的 **CL 驱动**（evaluator 中的 `pCLDriver`）。

---
<!-- _footer: "cases/euler/ · cases/euler3D/" -->
<!-- _class: dense -->

## 经典基准算例 — Riemann、激波、光滑流

<div class="cols">
<div>

### Riemann / 爆炸

- **Sod** `euler_config_1DRiemann.json`
- **LeBlanc** `euler_config_1DRiemann_LeBlanc.json`
- **Sedov 1D** `euler_config_1DSedov.json`
- **Sedov 2D** `euler_config_2DSedov.json`
- **Noh (3D)** `euler3D_config_Noh.json`
- **圆柱爆炸** `euler_config_blast.json`
- **M2000 天体物理射流** `euler_config_M2000Jet.json`

### 高超声速 / 激波干扰

- **双马赫反射** 2D + 3D
- **M5 激波衍射** `euler_config_M5Diffraction.json`
- **高超声速圆柱** `euler_config_cylinderHS.json`
- **双椭圆 / 双锥** 3D
- **球激波** `euler3D_config_SphereShock.json`

</div>
<div>

### 光滑 / 定常 / 非定常

- **等熵涡** `euler_config_IV.json` — 收敛性研究
- **Taylor-Green 涡 3D** `euler3D_config_TGV.json`、`euler3D_config_BenchTGV.json`
- **顶盖驱动空腔**（含高超声速变体）
- **冯·卡门涡街** 2D + 3D
- **层流平板边界层**
- **无粘圆柱 (MG 基准)** `config_cylinderInvis_mg_bench.json`

### 旋转 / 周期坐标系

- 旋转坐标系简单收敛测试
- 旋转周期等熵涡

</div>
</div>

---
<!-- _footer: "cases/eulerSA/ · cases/eulerSA3D/ · cases/euler2EQ/" -->
<!-- _class: tight -->

## 航空航天与工业基准算例

<div class="cols">
<div>

### 外部空气动力学

- **NACA 0012** — SA（`eulerSA_config_0012_AOA15.json`）和 k-ω（`euler2EQ/...`）变体，含 O2 升阶（`..._Elev.json`）和 MG 基准（`config_0012_mg_bench.json`）。
- **30p30n** 高升力 `eulerSA_config_30p30n.json`。
- **NASA CRM** — 常规和 CRM-HL 高升力。
- **DLR-F6** 运输机翼身组合体。
- **DPW-W1** 阻力预测机翼。
- **周期山** — LES 与 RANS 对比。

</div>
<div>

### 叶轮机械

- **Rotor 37** 跨音速压气机 `eulerSA3D_config_Rotor37.json`。
- **轴流风扇 A1** `eulerSA3D_config_FanA1.json`。

</div>
</div>

---
<!-- _footer: "RELEASE_NOTES.md:9-21" -->
<!-- _class: dense -->

## v0.2.0 中新增的求解器功能

<div class="cols">
<div>

### ODE 与预处理

- **HM3 重构** — U2R2 / U2R1 / U3R1 模式、`tpMG`、`incFScale`、与 `LimiterUGrad` 的保正耦合。
- 新增 **ESDIRK2 / ESDIRK3 / Trapezoidal**。
- **ILU-OMP** 预处理器。

### 湍流

- SA 上的 **DES → DDES → IDDES** 进展。
- **ψ 项修复**、旋转修正变体、ft2 开关。
- **k-ω 两方程**模型，配有专用的 `euler2EQ / euler2EQ3D` 可执行文件。
- **BCProfileIn** 用于 RANS 入口剖面。

</div>
<div>

### 通量 / 限制器 / BC

- **Roe_M8** 通量；**HLLE+**（实验性）。
- **incFScale**（增量通量缩放）集成到熵修正中。
- **等温壁面 BC** (`BCWallIsothermal`)。
- 重构中的**轴对称楔**度量。
- `LimiterUGrad` 中的**保正重构**限制器。

### 物理

- **旋转坐标系**（周期 + 简单收敛）。
- **重叠网格探索** — 挖洞、距离图、单元-单元连通性（二维演示）。

### 工作流

- `source2nd`、`mergeMultiResidual`、`normOrd`、`restartOutAtInit`、`resBaseType` 选项。

</div>
</div>

---
<!-- _footer: "src/Euler/EulerSolver.hpp:1270-1486" -->
<!-- _class: dense -->

## 主循环 — `RunImplicitEuler`

```cpp
void RunImplicitEuler() {
    InitializeRunningEnvironment(env);
    // optional restart
    if (config.restartState.useRestart)
        ReadRestart(config.dataIO.readRestart);

    for (int step = 1; step <= config.timeMarch.nTimeStep; ++step) {
        EvaluateDt(dt, u, uRec, CFL, dtMinAll, config.timeMarch.dtImplicit,
                   config.cflControl.useLocalDt, t);

        // Inner pseudo-time loop (driven by the chosen ODE integrator)
        odeIntegrator.Step(
            u, uInc,
            /*frhs*/     [&](rhs, u, dTau, iter, alpha, upos) { pEval->EvaluateRHS(...); },
            /*fdt */     [&](u,   dTau, alpha, upos)          { pEval->EvaluateDt(...); },
            /*fsolve*/   [&](x, rhs, uInc, dTau, alpha, ...)  { Krylov + LUSGS; },
            maxInnerIter, fStop, fIncrement, config.timeMarch.dtImplicit);

        UpdateCFL();
        if (step % config.outputControl.outputIntervalStep == 0)
            PrintData(fname, series, …);
        if (step % config.outputControl.restartInterval == 0)
            PrintRestart(fname);
        if (Converged() && config.timeMarch.steadyQuit) break;
    }
}
```

上述 lambda 函数是 `EulerEvaluator`、`GMRES_LeftPreconditioned` 和 `LUSGSMatrix*` 的接入点 — ODE 积分器不知道是哪个求解器在实例化它。
