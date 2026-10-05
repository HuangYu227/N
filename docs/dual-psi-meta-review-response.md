# 对 1f1eaa3 代码审查的回复

已逐条对照代码和用户批准的《双时间尺度 ψ 与可微 Meta-TTT 实现计划》。
这里需要区分实现缺陷、算法设计选项和需要实验证明的收益。

## 已核实并整改的工程问题

1. `train_clip()` 原先对每个 meta optimizer step 创建 `FirstUpdateProbe`，
   确实复制了所有 rank-local trainable parameters。现在 CLI 复用首次更新审计，
   普通 step 不复制全模型；显式 `audit_update=True` 才做额外审计。
   续训进程首个更新继续检查核心梯度。每步/chunk 的稳定性指标保留。
2. 原先 noisy `record_noise()` 和 `local_anchor_stats()` 不受诊断开关约束。
   普通推理现在跳过这些统计和 CPU 序列化；训练、teacher 对齐及显式诊断仍采集。
   因此诊断耗时和普通推理耗时必须分别标注。此修复尚没有 A100 性能测量。
3. 原先 Local 统计覆盖为最后一个 timestep。现在保留最后一次完整记录，并增加
   每个 anchor 的 `local_trajectory`，保留每次调用的真实 σ/timestep、gradient norm、
   Local ζ norm、Transport change 和 pre-Correct support/query loss。Clean 不增加条目。
   所有轨迹都是 detached 统计，不持有训练计算图。

本轮在 Python3.10.16/PyTorch2.11.0+cpu 下分批验证了 78 个相关用例，
包含 4 个新增回归以及 Meta runtime/training、真实 sampler、teacher 对齐、
CPU FSDP2/offload/exact resume、CLI 和日志。输出、η 梯度、参数更新、S/ψ 的
观测开关对照使用零误差阈值；没有放宽现有测试阈值。Python 编译和 git diff --check 通过。
这不是全库复跑，也不能作为四卡 CUDA 或性能验收结果。

## 数学和实验设计：不应直接称为实现错误

**独立 tanh。** 当前 `C=Cbase+δP*tanh(ψP)+tanh(ζ)` 确实不同于
`C=Cbase+δψ*tanh(ψP+ΔψL)`。当前公式与已批准计划第 2.1/2.3 节逐字一致，
不是实现过程中替换了模型。原批准设计并不能证明它优于另一公式；命名上明确写为
Persistent ψ + ephemeral Local ζ，可以避免歧义。独立分支保留了 Persistent
饱和时 Local 的调整能力，但实际好坏仍需实验，不能凭名称切换数学。

**Noise gate。** 目前没有显式 gate 属实，批准计划第 2.2 节明确要求第一版不加。
“同样 η”不等于“实际更新同样强”：g、clipping 和最终 ζ 都依赖 noisy features。
高噪声是否损害输出，是合理风险，却尚没有分噪声区间的性能证据，不能标成已证实 P0 bug。
尤其多 frame σ 混合时，`mean(1-σ)` 只缩放整体更新，不能排除高噪声 token 对
梯度方向的支配。若要研究可靠观察，应该区分全局 gate 与逐 token confidence-weighted
inner objective；后者也改变归一化分母。这些应作为明确的新算法对照，不能悄悄改默认。

**Checkerboard。** 它确实只用部分 write observations，但也是批准计划第 2.2 节的
明确主路径。支持补充 all-write 对照；目前没有证据证明任一选择更优。Query 仅未被直接
用于 Local inner loss；hidden features 已混合，Correct 又使用全部 K/V。因此即使
pre-Correct 的 query residual 降低，也不能称为严格独立的泛化证据，after-Correct 更不能。

**Full meta-gradient。** 完整 Local meta 和 TBPTT 内 Persistent future credit 是本轮
明确交付目标，不能把它换成 detached gradient 后仍宣称实现了同一个机制。
但 first-order Local 仍能立即影响当前输出，这一点成立；它是很有价值的计算/机制对照。
建议以后提供 `local_meta` 开关，同时保留完整版本，比较 Local detached、Local full-meta、
Persistent detached/live 的训练价值。当前两开关已经能区分 Local 有无和 Persistent
meta 有无，只是还不能单独隔离 Local 的 meta-gradient 收益。

CPU oracle 只能证明所测梯度/生命周期，不能证明真实 DiT 的速度、显存或训练收益。
四卡验收未通过，正式训练仍应等待；本轮依用户要求暂缓排查那次通信/退出错误。

## 建议的后续决策

先接受上述工程整改。数学不同时切换 gate、support 和 meta-credit 三个变量。
验收完成后，从同一初始化、case/seed/noise、history/camera protocol 出发，对
Local immediate adaptation、Local meta-credit、Persistent meta-credit 分别做训练对照。
先记录按真实 σ 分组的收益与开销，再决定 noise gate 与 all-write 是否作为主方法；
最终方法由证据选择，不因前一轮批准过便预设优越性，也不因代码审查建议便预设优越性。
