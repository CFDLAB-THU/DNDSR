---
<!-- _class: chapter -->
<!-- _paginate: false -->

<div class="ch-num">CHAPTER 10</div>

# 展望与结语

## 展望 · 指引 · 致谢

---
<!-- _footer: "docs/architecture/MeshConnectivity.md:416-458 · MeshDAGDesign.md" -->
<!-- _class: dense -->

## 展望 — 网格与拓扑

<div class="cols">
<div>

### 近期：可配置的 ghost

```cpp
struct GhostRequirement {
    int  cellRings       = 1;     // # of cell2cell rings
    bool nodeNeighbor    = true;  // cell2cell by vertex share vs face share
    bool complementNodes = true;  // ghost cells keep all their nodes
    bool complementBnds  = true;  // owned nodes keep all their bnds
};
```

- FEM 需要的 ghost 集合比 compact FV **更小**。
- 宽模板 FV（2+ 层）需要**更大**的集合。
- 当前默认 ghost 是硬编码的 — `nGhostLayers` 只调整深度，不调整邻居的*类型*。

</div>
<div>

### 中期：边实体

- 添加 `edge2node`、`cell2edge`、`node2edge`，遵循相同的 `AdjPairTracked` 规范。
- 将面插值算法提取为**通用**的 codim-K 模板。
- 用于**基于节点的 FV** 和 **FEM** 工作流。

### 长期：DMPlex 风格的 DAG

- 跨节点/边/面/单元的统一点编号。
- 参数化邻接关系 — `useCone` × `useClosure` 布尔矩阵。
- PetscSF 风格的通信结构。
- PetscSection 风格的拓扑与离散化解耦。
- 提案见 `docs/architecture/MeshDAGDesign.md`（765 行）。

</div>
</div>

---
<!-- _footer: "RELEASE_NOTES.md:20 · src/EulerP/ · docs/dev/ideas.md" -->
<!-- _class: dense -->

## 展望 — 求解器、并行、V&V

<div class="cols">
<div>

### 求解器

- **反应/多组分**（`NS_EX`）— 成熟度提升，发布验证算例。
- **重叠网格** — 已有 2D 演示（挖洞、距离图、单元-单元连通性）；扩展到 3D 并添加完整的传递算子。
- **完整（组装）Jacobian** — 目前仅支持无矩阵 + LU-SGS。组装分块 CSR Jacobian 后可启用 AMG + SuperLU_dist + PETSc 作为预条件子。

### 并行

- 将 CUDA 覆盖从 `EulerP` 扩展到完整的 `Euler` 求值器。
- 更多 SoA 核函数；通过固定设备内存进行 GPU 可知的 MPI。
- 为 VR 重构提供更广泛的设备覆盖。
- Task-based execution实验。

</div>
<div>

### V&V 工作流

- 标准化的**验证与确认**工具链，每次发布时运行固定算例集，向文档站发布收敛图。
- 更多叶轮机械算例：NASA Rotor 67、低速风扇级。
- 气动声学的噪声源诊断。

### 文档与开发体验

- **扩展 Python 示例**以匹配当前 C++ 覆盖。
- 对剩余模块（Solver、Geom、CFV、Euler、EulerP）进行 **Clang-tidy 清理** — 与 DNDS 相同的方案。
- **贡献者入门**指南，已在 `docs/dev/` 中拟定大纲。

</div>
</div>

---
<!-- _footer: "docs/architecture/ · docs/theory/ · docs/guides/" -->
<!-- _class: dense -->

## 延伸阅读

<div class="cols">
<div>

### 架构

- **`array_infrastructure.md`** — 自底向上遍历 `Array` → `ArrayTransformer` → `ArrayPair` → `ArrayDof`。
- **`MeshConnectivity.md`** — AdjPairTracked 状态机、ghost 规格 DSL、DAG 展望。
- **`Serialization.md`** — 分层 I/O、跨进程重启、偏移模式。
- **`Paradigm.md`** — 延迟抽象哲学，与 OpenFOAM/SU2 对比。

### 理论

- **`Variational_Reconstruction.md`** / `.pdf` — 面泛函、内积选择及局部系统的完整推导。
- **`Shape_Functions.md`** — 逐单元形函数与积分。

</div>
<div>

### 指南

- **`building.md`** — 外部依赖、头文件、CMake 预设。
- **`array_usage.md`** — 如何使用 `Array`/`ArrayDof` 编写代码。
- **`geom_usage.md`** — 网格构建与 VR 流水线。
- **`python_geom_guide.md`** — 完整的 Python Geom API 参考。
- **`serialization_usage.md`** — HDF5 检查点、重分布。
- **`style_guide.md`** — C++ 和 Python 规范。
- **`examples.md`** — 可运行的 `examples/ex_*.cpp` 程序。

### 测试

- **`docs/tests/overview.md`** — 黄金值、确定性、测试套件汇总。
- 各模块测试页面：`docs/tests/{dnds,geom,cfv,euler,solver}_unit_tests.md`。

</div>
</div>

---
<!-- _class: lead -->
<!-- _paginate: false -->
<!-- _footer: "github.com/CFDLAB-THU/DNDSR · cfdlab-thu.github.io/DNDSR" -->

# 谢谢

**三条命令即可运行**

```bash
cmake --preset release-test
cmake --build build -t euler -j32
mpirun -np 4 ./build/app/euler.exe cases/euler_config_IV.json
```

<br>

**代码** · [github.com/CFDLAB-THU/DNDSR](https://github.com/CFDLAB-THU/DNDSR) **文档** · [cfdlab-thu.github.io/DNDSR](https://cfdlab-thu.github.io/DNDSR) **发布说明** · `RELEASE_NOTES.md` (v0.2.0)

<br>

*清华大学 CFD 实验室*
