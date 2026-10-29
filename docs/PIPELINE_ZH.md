# WUST 使用流程

## 观测

~~~text
已有复数总压力 + 几何 + mask + 单位/傅里叶约定
                     |
                     v
              reconstruct 请求

已有 RF -> prepare -> 规范频域观测（显式数据准备）
介质模型 -> simulate -> RF（数据集/fixture 工具，不在生产重建内）
~~~

WUST 不依赖 BenchLab，不从病例真值生成生产观测。输入可以是 frequency_input，
也可以是已经准备的 measurements。位置用 [x,y]，图像用 [y,x]；完整规范见
[CONTRACTS](CONTRACTS.md)。

## 一次重建

~~~text
Python launcher：检查协议、路径和时限
  -> 一个 MATLAB 进程
       -> prepare_observations：轴/单位/mask、频率排序、落点与傅里叶转换
       -> 完整 Hz 计划 -> 按 max_updates 截断
       -> 每项：Helmholtz 正演 -> 复源尺度消元 -> 残差与伴随
               -> NCG/曲率步长 -> 慢度更新 -> 限幅与边界约束
  -> 写出 HDF5 数值数组 -> JSON 成功索引
~~~

CPU/GPU 必须显式选择，不在失败后自动切换。独立 ingest_frequency 操作用于导出
规范观测，不要求每次重建额外启动一次 MATLAB。RF 降采样采用抗混叠 FIR。
native-batch 仿真中每个 TX 是独立场，TX/RX 不必相同；设备与批量大小显式提供。

## 结果

声速 c_mps 为米/秒。日志包含完整/实际频率计划、更新数、更新时间和逐频更新前 loss。
计划结束不是收敛。超时由外层回收所属进程组，不伪造最终图像；未计算的最终残差留空。
fixture 生成和正式重建分别计时，CPU、GPU、native 验收各自记录。

[部署](../Runtime/README.md) · [协议示例](RUNTIME_API.md) ·
[历史研究](research_index.md)
