# WaveformInversionUST

[English](README.md) | [简体中文](README.zh-CN.md)

**面向二维超声计算机断层成像（USCT）的 MATLAB 声速全波形反演运行时。**

WUST 根据复数声压观测、采集几何和初始模型重建声速图，将频率延拓、九点 Helmholtz 求解器与非线性共轭梯度更新组织为完整的数值流程。CPU 与 GPU 后端共享物理模型和离散系数。

仓库提供两条接入路径：MATLAB 函数服务数值算法开发，Python JSON/HDF5 接口服务脚本、基准评测和 agent。[usct-benchlab](https://github.com/lucian-dw/usct-benchlab) 通过 `fwi_wust` 算法调用本运行时；WUST 也支持独立部署和运行。

**快速导航：**[仓库结构](#仓库结构) · [运行一次重建](#运行一次重建) · [Python-与-MATLAB-接口](#python-与-matlab-接口) · [数据与结果](#数据与结果) · [开发入口](#开发入口)

## 数值模型与功能

优化变量为慢度 $s=1/c$，其中 $c$ 是以 m/s 为单位的声速。重建在指定的更新区域和声速边界内进行，衰减固定为零。

对于每个频率和发射器，$p(s)$ 表示所选接收器上的预测声压，$d$ 表示对应观测。WUST 拟合一个复数源系数，并以此构造数据拟合目标：

$$
a^\star=\frac{p^H d}{p^H p},
\qquad
\Phi_f(s)=\frac12\sum_{\mathrm{TX}}\|a^\star p(s)-d\|_2^2.
$$

| 组成 | 实现 |
| --- | --- |
| 声波传播 | 优化九点 Helmholtz 离散，配合完全匹配层（PML） |
| CPU 求解器 | 双精度稀疏 LU，在当前模型和频率下供正演、伴随和曲率求解复用 |
| GPU 求解器 | 复数单精度 CUDA Block LU，支持发射器分批计算 |
| 导数 | 离散慢度梯度与方向曲率 |
| 优化流程 | 频率延拓、ROI 梯度滤波、预条件非线性共轭梯度和慢度投影更新 |
| 波数模型 | `continuum`；或采用方形网格、显式仿真时间步长与参考声速的 `kwave-ldr9` |
| 数据准备 | 频域声压导入、RF 重采样与 DTFT、k-Wave 仿真 |

目标函数、离散导数与更新公式详见[数值约定](docs/CONTRACTS.md)。

## 仓库结构

| 位置 | 职责 | 推荐入口 |
| --- | --- | --- |
| `Runtime/python/` | 请求定义、JSON/HDF5 传输与 MATLAB 进程执行 | [wust_runtime.py](Runtime/python/wust_runtime.py)、[contracts.py](Runtime/python/contracts.py)、[wust_io.py](Runtime/python/wust_io.py) |
| `Runtime/matlab/` | 观测准备、FWI 迭代与数据文件读写 | [wust_run.m](Runtime/matlab/wust_run.m)、[wust_prepare_observations.m](Runtime/matlab/wust_prepare_observations.m)、[wust_reconstruct.m](Runtime/matlab/wust_reconstruct.m) |
| `Runtime/solver/` | Helmholtz 装配、CPU/GPU 求解与 CUDA MEX | [HelmholtzSolver.m](Runtime/solver/HelmholtzSolver.m)、[ninePointCoefficients.m](Runtime/solver/ninePointCoefficients.m) |
| `Runtime/schemas/` | 请求与数组文件的 JSON schema | [wust.request.json](Runtime/schemas/wust.request.json) |
| `Runtime/native/` | 原生 k-Wave 仿真启动器与源分批计算 | [部署指南](Runtime/README.md) |
| `Runtime/third_party/` | 随仓提供的 k-Wave CUDA 源码及许可证 | [k-Wave CUDA](Runtime/third_party/kspaceFirstOrder-CUDA/) |
| `Runtime/tests/` | Python 协议测试与 MATLAB 数值验证 | [验证指南](docs/validation.md) |
| `examples/` | 仿真、RF 准备与重建示例 | [simulate_and_reconstruct.m](examples/simulate_and_reconstruct.m) |
| `docs/` | 数值定义、协议与开发参考 | [运行时 API](docs/RUNTIME_API.md)、[中文流程说明](docs/PIPELINE_ZH.md) |

阅读数值主线时，从 `wust_reconstruct.m` 进入，沿目标与梯度调用阅读 `wust_oracle.m`，再通过 `HelmholtzSolver.m` 理解离散波动方程。接入外部系统时，从 `wust_runtime.py` 和 `contracts.py` 中的请求字段开始。

## 运行一次重建

### 1. 配置环境

在 Linux 或 macOS 上使用 **Python 3.12+**，从仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy h5py
python Runtime/python/wust_runtime.py describe
```

`describe` 以 JSON 返回操作、协议 schema、后端和预算控制信息。

| 功能 | 环境依赖 |
| --- | --- |
| CPU 重建 | MATLAB |
| GPU 重建 | MATLAB Parallel Computing Toolbox、受支持的 CUDA 编译器与已编译的 CUDA MEX |
| FIR RF 降采样 | MATLAB Signal Processing Toolbox |
| 声学仿真 | k-Wave MATLAB 工具箱 |
| 原生批量仿真 | Linux，以及 [Runtime/README.md](Runtime/README.md) 所述的 CUDA/HDF5 构建环境 |

使用 GPU 重建时，在 MATLAB 中编译 MEX 函数：

```matlab
addpath('Runtime/matlab');
wust_build_mex;
```

通过 `config.backend` 选择重建后端。下方请求示例使用 `cpu`。

### 2. 提供观测与初始模型

运行时提供四种操作：

| 操作 | 输入 | 输出 |
| --- | --- | --- |
| `reconstruct` | `wust.frequency_input` 或 `wust.measurements`，以及 `wust.initial_model` | `wust.reconstruction` |
| `ingest_frequency` | `wust.frequency_input` | 可复用的 `wust.measurements` |
| `prepare` | `wust.rf` 与 RF 准备配置 | `wust.measurements` |
| `simulate` | `wust.simulation_input` 与仿真配置 | `wust.rf` |

重建接收两份输入文件：频域声压观测，以及包含更新掩码的初始声速模型。每份输入由 JSON 描述文件和 HDF5 数组文件组成。`reconstruct` 在一个 MATLAB 进程内完成观测准备和整段频率计划。

使用 `ingest_frequency` 保存可复用的规范频域观测，使用 `prepare` 转换 RF 测量，使用 `simulate` 生成 RF 数据。四种操作的完整字段见[运行时 API](docs/RUNTIME_API.md)。

### 3. 创建并执行请求

将以下内容保存为 `request.json`，按采集数据填写绝对路径和物理参数。示例安排在 100 kHz 更新两次，在 120 kHz 更新一次：

```json
{
  "schema": "wust.request",
  "schema_version": 2,
  "operation": "reconstruct",
  "input_manifest": "/absolute/input/frequency.json",
  "initial_manifest": "/absolute/input/initial.json",
  "output_manifest": "/absolute/output/reconstruction.json",
  "frequency_schedule_hz": [100000, 100000, 120000],
  "max_updates": 3,
  "save_history": false,
  "config": {
    "backend": "cpu",
    "bounds_mps": [1300, 1800],
    "stencil_bounds": [1400, 1700],
    "pml_m": 0.004,
    "pml_strength": 10,
    "max_update_mps": 12,
    "step_damping": 0.25,
    "source_batch_size": 2,
    "wavenumber": "continuum",
    "filter_cutoff": 0,
    "filter_order": 4
  }
}
```

`bounds_mps` 设置声速范围，`stencil_bounds` 设置构造九点离散系数所用的固定声速区间。`pml_m` 是以米为单位的 PML 厚度，`max_update_mps` 设置每步声速变化上限，单位为 m/s。频率计划从输入观测已有的频率中选择。

创建输出目录，为每次运行选择新的输出描述文件：

```bash
mkdir -p /absolute/output
python Runtime/python/wust_runtime.py run /absolute/request.json \
  --matlab /path/to/matlab --timeout-s 300
```

启动器向 stdout 返回一份 JSON 结果，向 stderr 写入诊断信息。`--timeout-s` 设置本次调用的耗时上限。保存的描述文件指向已完成的 HDF5 数组，并包含重建配置与迭代记录。

## Python 与 MATLAB 接口

### Python 脚本与 agent

执行请求时，可以选择前面的命令行方式，也可以使用下面的 Python 入口。导入运行时模块时设置 `PYTHONPATH=Runtime/python`；程序通过 `capabilities()` 查询接口，通过 `run()` 执行请求文件：

```bash
PYTHONPATH=Runtime/python python - <<'PY'
from wust_runtime import capabilities, run

spec = capabilities()
print(spec["operations"])
result = run(
    "/absolute/request.json",
    matlab="/path/to/matlab",
    timeout_s=300,
)
print(result["metadata"]["completion"])
PY
```

接入 agent 时，宿主提供路径、后端和耗时预算，请求文件描述观测、初始模型与数值计划。[usct-benchlab 的 fwi_wust 适配器](https://github.com/lucian-dw/usct-benchlab) 将这套协议接入统一算法 API。

通过 Python codec 读取已保存的重建：

```bash
PYTHONPATH=Runtime/python python - <<'PY'
from wust_io import read_artifact

manifest, arrays = read_artifact(
    "/absolute/output/reconstruction.json", "wust.reconstruction"
)
sound_speed_mps = arrays["c_mps"]
print(sound_speed_mps.shape)
print(manifest["metadata"]["completed_updates"])
PY
```

同一模块提供 `write_artifact(path, name, metadata, arrays, axes, units)`，用于创建输入文件。数组字段和单位定义在 [contracts.py](Runtime/python/contracts.py) 中，物理元数据说明见[运行时 API](docs/RUNTIME_API.md)。

### 直接调用 MATLAB

MATLAB 调用方可以直接使用观测准备与重建函数。下面的示例复用前面的请求文件：

```matlab
addpath('Runtime/matlab');
request = jsondecode(fileread('/absolute/request.json'));
obs = wust_prepare_observations( ...
    wust_read_artifact(request.input_manifest, ...
        {'wust.frequency_input', 'wust.measurements'}));
initial = wust_read_artifact(request.initial_manifest, 'wust.initial_model');

cfg = request.config;
cfg.update_mask = initial.update_mask;
cfg.frequency_schedule_hz = request.frequency_schedule_hz;
cfg.max_updates = request.max_updates;
cfg.save_history = request.save_history;

result = wust_reconstruct(obs, initial.initial_mps, cfg);
sound_speed_mps = result.c_mps;
```

`wust_prepare_observations` 准备几何、频率顺序、有效接收器数量和观测能量。重建过程直接复用这个 prepared 对象；编辑声压或掩码后，再调用准备函数刷新派生量。`wust_prepare` 提供对应的 RF 到频域观测路径。

## 数据与结果

### 数组约定

| 物理量 | 布局与单位 |
| --- | --- |
| 声速 / 更新掩码 | `[y,x]`，m/s / 布尔值 |
| 网格 | `shape_yx`、`spacing_yx_m`、`origin_yx_m`、`origin_kind: pixel_edge`（外侧像素边界） |
| TX/RX 坐标 | `[element,xy]`，米 |
| 频域输入 | 复数总声压，轴顺序为 `[frequency,tx,rx]` 或 `[tx,rx,frequency]` |
| 规范观测 | `Y[tx,rx,frequency]`、同形状掩码、严格升序的 Hz 频率 |
| RF | `pressure[time,rx,tx]` 与真实采样时刻 `time_s` |
| 规范 Fourier 变换 | $\Delta t\sum_n p(t_n)e^{-2\pi i f t_n}$ |
| 采样索引 | MATLAB 从 1 开始的列主序索引 |

声压元数据声明 `fourier_sign`、`pressure_type`、`data_units` 和 `spectrum_normalization`。对于声明了 `real_pressure: true` 的正号频谱，准备阶段通过共轭转换到规范形式；`Pa*s` 使用 `dtft_dt` 归一化。复数数组分别存入实部与虚部 HDF5 数据集。

TX 和参与拟合的 RX 采样点位于最外层 Dirichlet 环内部，更新掩码位于 PML 内侧的成像区域。每个参与评价的发射器与频率使用至少两个有效接收器拟合复数源系数，有效的零观测也参与计算。

BenchLab 使用 `[y,x]` 坐标列和 `[frequency,tx,rx]` 声压布局。其适配器在协议边界将位置转换为 WUST 的 `[x,y]` 约定。

### 频率计划与输出

| 配置或输出 | 含义 |
| --- | --- |
| 省略 `frequency_schedule_hz` 或设为 `null` | 每个规范频率更新一次 |
| 重复的 Hz 条目 | 在对应频率重复更新 |
| 空计划或 `max_updates: 0` | 返回初始模型，更新次数为零 |
| `max_updates` | 限制计划中的更新次数 |
| `save_history: true` | 保存包含初始模型的 `history_mps[y,x,model_step]` |
| `c_mps[y,x]` | 最终声速数组，float64 |
| `records` | 每次更新的目标值、残差、步长、计算量与耗时 |
| `completion` | 请求计划与更新预算的执行状态 |

每条迭代记录对应计划中该频率在**本次更新前**的模型目标值和残差。设置 `config.evaluate_final_residual: true`，可以在全部规范频率上评价最终模型，并获得对应计算量与耗时。默认的 `final_data_residual` 为 `null`。

## 开发入口

按开发任务定位相应实现：

| 开发任务 | 阅读与扩展位置 |
| --- | --- |
| 增加或修改观测准备 | `wust_prepare.m`、`wust_prepare_observations.m`、`wust_geometry.m` |
| 修改目标函数或梯度 | `wust_oracle.m`、`directional_curvature.m` |
| 修改模型更新 | `wust_reconstruct.m`、`wust_ncg_direction.m`、`wust_roi_gradient.m`、`wust_project_slowness_update.m` |
| 修改传播离散或后端 | `HelmholtzSolver.m`、`ninePointCoefficients.m`、CUDA MEX 源码 |
| 扩展协议操作或字段 | `Runtime/python/contracts.py`、Python/MATLAB 调用方和生成的 schema |

[AGENTS.md](AGENTS.md) 说明项目开发约定。每项数值修改配套对应的梯度、求解器或端到端参考验证。协议修改后生成 schema：

```bash
python Runtime/python/export_schemas.py
python -m unittest discover -s Runtime/tests -p 'test_*.py'
```

在 MATLAB 中运行 CPU 数值验证：

```matlab
addpath('Runtime/tests');
test_runtime('cpu');
test_ingest_frequency;
test_resampling;
```

编译 CUDA MEX 后，使用 `test_runtime('gpu')` 验证 GPU 路径。[验证指南](docs/validation.md) 提供公开文件接口的集成命令，[仿真示例](examples/simulate_and_reconstruct.m) 展示仿真、RF 准备与重建的完整流程。

## 文档、许可证与引用

- [运行时 API](docs/RUNTIME_API.md)：请求、数组、配置和结果字段。
- [数值约定](docs/CONTRACTS.md)：目标、导数、投影与调度。
- [部署指南](Runtime/README.md)：MATLAB、CUDA MEX 与原生仿真。
- [中文流程说明](docs/PIPELINE_ZH.md)：数据流与接入方式。
- [来源与许可](ATTRIBUTION.md)：源码来源和第三方条款。

运行时采用 [MIT 许可证](LICENSE.txt)。随仓提供的 [k-Wave CUDA 源码](Runtime/third_party/kspaceFirstOrder-CUDA/License.md) 采用 LGPL-3.0-or-later。MATLAB 与 k-Wave MATLAB 工具箱保留各自许可证。

本项目源自 [Rehman Ali 的 WaveformInversionUST](https://github.com/rehmanali1994/WaveformInversionUST)。相关研究请引用：

> Rehman Ali et al., *2-D Slicewise Waveform Inversion of Sound Speed and Acoustic Attenuation for Ring Array Ultrasound Tomography Based on a Block LU Solver*, IEEE Transactions on Medical Imaging, 2024. [doi:10.1109/TMI.2024.3383816](https://doi.org/10.1109/TMI.2024.3383816)。
