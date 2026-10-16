# In-Place TTT 加速设计：Token-Tiled Residual Fusion（TTRF）

基于 HuangYu227/N 的 17e8b17（In-Place FFN TTT）代码，本方案只做 backend acceleration，不改变 TTT 数学目标。

## 1. 当前真实计算图

每个 anchor layer 使用：
- hidden 维度约为 C_h = 3 C；
- fast state ΔW ∈ R^{C × C_h}，FP32；
- read：R = Z ΔW^T；
- clean write：V = DWConv_5(Y) P；
- update：ΔW' = ΔW + s V^T Z；
- output：Y_out = cast(B + R)。

当前实现位于 worldttn/inplace_ttt.py 与 diffusion/model/nets/basic_modules.py。五个 layer 是 3/7/11/15/19。17e8b17 已将 valid-token reduction 改成 mean，并把训练/推理 update cap 对齐到 0.01。

## 2. 核心观察

我们没有采用 TTT-Video 的 Tensor Parallel/DSMEM 方案。原因是当前单层 fast state 大约：

2240 × 6720 × 4 bytes ≈ 57.4 MiB，

不可能整体驻留单个 SM 的 shared memory。

我们自己的优势是 streaming chunk 的生命周期非常明确：

old ΔW 只读；clean chunk 形成 candidate；commit 后才替换 runtime state。

因此最合适的 GPU tiling 单位不是整个 model-parallel shard，而是 C × C_h state tile。

## 3. TTRF-V1：Read Fusion

当前：
R = Z ΔW^T
Y = cast(B + R)

TTRF 直接逐元素计算：

Y[n,c] = cast(B[n,c] + Σ_h Z[n,h] ΔW[c,h])

所以没有任何算法变化，只改变 reduction 的执行位置。

理论上它删除了 materialized correction tensor 到 residual-add/cast 之间的一次 kernel boundary，并把 correction 结果直接写入最终 output。

## 4. TTRF-V2：Update Fusion

当前：
U = V^T Z
ΔW' = ΔW + sU

TTRF 直接计算：

ΔW'[c,h] = ΔW[c,h] + s Σ_n V[n,c] Z[n,h]

因此也是严格的逐元素等价。

这里的关键是 state tile 在一次 kernel 中完成 token reduction 和 state write，不再产生独立 U 张量。

## 5. 严格数学证明

定理 1（Read equivalence）：

令 R=ZΔW^T。矩阵乘法定义给出：
R[n,c]=Σ_h Z[n,h]ΔW[c,h]。
TTRF 对完全相同的有限求和逐项计算并加 B，因此实数算术下完全相同。GPU 浮点下只可能有 reduction 顺序导致的 rounding difference。

定理 2（Update equivalence）：

令 U=V^TZ。则 U[c,h]=Σ_n V[n,c]Z[n,h]。
TTRF 使用同一个有限求和，并执行 ΔW+sU，因此实数算术下与 reference 完全相同。

定理 3（mean + clip）：

当前 s = η q / m，其中 m 是 valid token 数，q 是 clip scale。
将 s 作为 kernel scalar 即可得到：
ΔW' = ΔW + ηq/m V^TZ。
所以 mean normalization 和 clipping 无需改变。

## 6. 为什么这是我们自己的加速思路

论文/开源 TTT kernel 可以提供启发，但它们主要解决 TTT-MLP 的 model-parallel、TMA、DSMEM 和 producer/consumer 问题。

我们的 TTRF 利用了 SANA-WM 的特定事实：
1. fast state 是单矩阵 residual adapter；
2. base point projection 与 fast correction 必须在 dtype cast 前合并；
3. clean-only write 和 noisy read 生命周期不同；
4. state 是 FP32，而 base/output 是 BF16/FP8-compatible；
5. chunk 是天然同步边界。

因此名称和设计都是针对本模型计算图提出，而不是复刻别人的 kernel。

## 7. 最佳组合

推荐按以下顺序启用：

V0：PyTorch reference（唯一数学 oracle）
V1：TTRF-Read
V2：TTRF-Read + TTRF-Update
V3：TTRF + fused depthwise target preparation
V4：只有在算法验证 GDN+TTT 后，再做 GDN/TTT cross-module fusion。

第一版不要做：
- 强制低秩 ΔW；
- 强制 diagonal P；
- 把整个 ΔW 放 SMEM；
- 改变 chunk update semantics；
- 用 FP8 存储 ΔW。

这些都会改变算法或数值稳定性，不能作为纯 backend acceleration。

## 8. 性能判据

必须在目标 L20/H100 上用 CUDA events 测：
- B=1/2；
- chunk token 数；
- read-only inference；
- clean read+write；
- full 121-frame rollout；
- peak allocated/reserved；
- kernel count；
- max abs / relative L2。

TTRF 只有在：
1. numerical gate 通过；
2. operator latency 不劣于 reference；
3. end-to-end rollout 有稳定收益；
之后才能设为 production backend。

## 9. 当前代码审核

本分支新增：
- worldttn/inplace_ttt_triton.py
- tests/ttn/test_inplace_ttt_triton.py
- docs/INPLACE_TTT_TTRF_ACCELERATION.md

默认没有替换现有 reference path，因此不会偷偷改变现有实验结果。

重要限制：当前会话没有你们 L20/CUDA runtime，不能诚实地声称已经完成实机 kernel benchmark 或编译验收。因此 Triton backend 必须经过你们目标机器的 numerical/performance gate 后再接 production default。

## 10. 下一步

如果 TTRF-V1/V2 在 L20 上成立，我建议下一步把 target preparation 也融合进去，形成：

DWConv_5(Y) -> P -> V^T Z -> clipped ΔW commit

并进一步研究：

GDN recurrent state + FFN fast state

是否能共享 token streaming pipeline。

最终系统贡献可以形成：
Algorithm：GDN + adaptive FFN plastic memory
Systems：TTRF state-tile backend

这样算法创新与 kernel 创新是相互支撑的。
