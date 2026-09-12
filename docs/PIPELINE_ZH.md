# k-Wave → FWI 管线整理

## 分层

```text
benchlab：数据集、裁剪/缩放、物理坐标、split、评价合同
    ↓ 计算网格上的 SoS + 阵元索引 + 源波形
wust_simulate：2D k-Wave / native128 独立 TX 批处理
    ↓ FIR 抗混叠、保留 RF [time, RX, TX] 与实际阵元坐标
wust_prepare：显式时间窗 / DTFT / 可选落点相位修正 / 显式 mask
    ↓ 复数观测 [TX, RX, frequency]
wust_reconstruct：逐频 FWI、源幅相消元、伴随梯度、NCG
    ↓ Helmholtz 九点差分 + PML + CUDA Block-LU
物理网格 SoS、更新历史、残差、真实成本 → benchlab 评价
```

FWI 是优化流程，Block-LU 是线性求解器。上游演示脚本保留；
新接口在 Runtime 内，不依赖 diffusion/DPS，也不搬运所有研究试验分支。

## 两项不能混淆的改进

**native128 是速度改进。** CUDA 中每个 TX 使用独立传播场，共享介质，
以 rank-2 cuFFT planMany 与批量 kernel 调度全部发射，不是同时激励后分离，
不做测量融合。不同 batch 使用独立线性地址，恢复原始 RX 逻辑顺序。

**抗混叠是数值合同修复。** 历史 1:ds:end 抽点不低通。旧 ds5 在原始
fs≈10.584 MHz 示例中，保存 Nyquist≈1.058 MHz，不能声称支持 1.25 MHz。
本版生成端使用 FIR/polyphase 后降采样；ds2 示例保存 fs≈5.292 MHz，
实际 fs 随案例 dt 改变。旧 native 入口仍残留过抽点版本，故未直接照搬。

## 保留与边界

- 常用 29 频 0.30:0.025:1.00 MHz 是调用配置，不写死。
- 保存 grid-snapped 阵元，避免名义圆环与实际落点的传播相位误差。
- 保留零吸收 k-Wave-aware LDR9；必须提供原始 dt 和参考声速。
- 保留 CUDA 错误检查、GPU 波场驻留、多 RHS 求解。
- 导数审计补齐九点质量模板；不冒充所有历史实验的逐位重放。
- 不纳入 KCI D1/D3/FD、网络、3D slab、在线学习和失败研究支线。
- 不自动删除 RF、不自动占用 /dev/shm、不调度其他用户的 GPU。

## benchlab 后续职责（本轮不实现）

固定 compatibility commit，按 CONTRACTS.md 封装 MAT 输入，调用 MATLAB
函数或 Python launcher，接收完整计算域输出，再由 benchlab 裁剪评价。
已有 review 分支仅作需求参考，未修改。

## 发布定位

这是带可执行 smoke 的首个独立 runtime，不是所有历史分支的重新认证。
本轮已在 A100 从源码编译，完成全 128 路 serial/native 同输入对照（RF
和 DTFT 误差为 0）、8 路独立 MATLAB 参考对照（约 1e-6），以及 1024²
计算网格 native128 生成。生产尺寸生成约 91.30 秒，FIR 1.34 秒，29 频
提取 0.39 秒；具体硬件与采样内存见 VALIDATION.md。小网格含进程/I/O 的
43.29 倍加速不能外推成大网格 kernel 加速比。生成 RF 已保留。
