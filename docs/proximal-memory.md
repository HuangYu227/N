# TLA + Direct-S Meta-TTT + protected/participative memory

这是新的训练候选，不是已证实优于双 ψ 的模型。代码分支 `codex/ttn-proximal-memory`，基于 `809d185`；旧配置仍运行旧数学。新配置从原始 SANA 加载视觉/camera 表征，从 step 0 joint training。没有读取旧 step100 权重作为初始化，没有引入 visual Softmax 或新的 MLP/LoRA。

## 1. 哪些问题得到机制上的修复

| 旧问题/证据 | 本版处理 | 能证明什么、还不能证明什么 |
|---|---|---|
| Cayley 右旋保持 S 奇异值，ψ 只能改变方向；旧实验的 Local 效果很弱 | 直接适配完整 S；默认 R=I，旧 ψ/controller/generators 不参与优化 | 不再受旋转流形限制；不能保证 S 的有效秩或视频质量必然提升 |
| 单步 Delta update 对不同特征方向响应不同，弱方向可能收敛很慢 | 一次带旧状态先验的加权最小二乘求解 | 固定特征下有唯一解，目标不增加；不是所有单个历史/当前 residual 都必须下降 |
| 只保留一个初始 compressed reference，固定 read 混合收益有限 | 保留早期 clean K/V/W、最近帧和参与度选出的中间 token，将其放入同一 inner objective | reference 内容参与求解而非只缩放输出；效果要和无缓存、prefix-FIFO 对照 |
| 13 latent、GT history、TBPTT=2 与长 rollout 有差距 | 主候选 121 latent / 40 predicted chunks / TBPTT=4 / generated clean history | 窗口内未来 credit 延长；没有全视频反传，sampler/CFG 仍存在训练评估差异 |
| 只看到 inner loss 下降或非零 FP32 delta，可能实际 BF16 输出没有变化 | 同时记录求解量、固定输入下的 gate/proj FP32 贡献和量化后的 paired output 差异 | 证明路径/幅度；不能等同于长程 MSE 改善 |

此前单 case H1/H2 中，native-GT 将长序列 MSE 从约 1.032 降到 0.374，而 TTN-GT 约 1.035。这支持 native history 污染的重要性，不支持“只改 S 就能解决全部误差”。本版保留 generated-history 训练来让整个 backbone 接触自己的历史；native GDN/FFN/camera 的计算和缓存格式不改变。

## 2. 内部目标、解和梯度

每个 batch/head 独立。`T=S_old` 是本版默认 incoming prior。当前源与历史源 `s` 的 K/V/W 来自各自 hidden features，未来 GT 不进入 inner objective。训练的 flow loss 仍使用原来的监督目标。

设 `Z_s = sum(w_s)`，非空源的 `Wbar_s=W_s/Z_s`。当前源系数为 1，总历史系数默认 `h=0.25`，在该 batch/head 非空的 prefix / selected-middle / recent 三类间均分。空源贡献严格为零。

\[
J(X)=\frac12\sum_s a_s\|\bar W_s^{1/2}(K_sX-V_s)\|_F^2+
\frac\lambda2\|X-T\|_F^2,
\quad G=\sum_s a_sK_s^T\bar W_sK_s,
\quad B=\sum_s a_sK_s^T\bar W_sV_s.
\]

\[
\lambda=\kappa\,\mathrm{tr}(G)/D+\epsilon,
\qquad S^*=T+(G+\lambda I)^{-1}(B-GT),
\qquad Y=QS^*.
\]

实现使用 `torch.linalg.solve`，没有显式求逆。`G` 是 PSD，`lambda>0`，所以目标强凸、解唯一。这一步替代旧 Correct，后面不会再做一次 Delta update，也不会叠加旧 sink read mixing 或 observed replay。K/V、β、历史权重、归一化分母和 lambda 都保持 live。

令 `M=G+lambda I`，微分满足

\[
dS^*=M^{-1}[dB-dG\,S^*+\lambda\,dT+d\lambda(T-S^*)].
\]

因此未来 flow loss 能穿过 clean S 更新及选中的 K/V/W 回到前面特征。测试额外切断 S 与 native 路径，仅保留 history 路径，再与 detach-history 对照，避免把直接梯度误认为 meta-credit。Top-C 索引是离散选择，不对排序求导；被选中张量仍有梯度。

固定 K/V/W 时，旧 S 的齐次传递矩阵为 `lambda*(G+lambda*I)^-1`，特征值 `lambda/(sigma_i(G)+lambda)` 在 `(0,1]`，零特征方向保留旧状态。对于 `D=112,kappa=16`，条件数上界 `1+D/kappa=8`；先验保留率下界约 `1/8`，上界 1。lambda 会随 K 尺度变化，以上不是完整神经递归的范数界，也不是视频稳定性定理。κ=16、历史权重0.25是保守初始候选，尚未校准为最优值。

## 3. 网络、数值范围与状态生命周期

| 对象 | 结构/范围 | 生命周期/训练 |
|---|---|---|
| QKV、Norm、proj、output gate、native camera | 保留原 SANA，5 anchor × 20 head × 112 dim；SiLU gate 不设幅值上界 | offline 参数，SANA-inherited LR=1e-6 |
| β | 每 anchor Linear(2240,20)+sigmoid，理论上(0,1)，零头初始化0.5 | TTN-new LR=1e-5；参与当前及 retained observation 的权重 |
| S | FP32 `[B,5,20,112,112]`，没有人为截断幅值 | episode state；noisy 临时解，clean 一次提交；不存入 checkpoint |
| λ | κ·trace(G)/112+1e-6，严格正 | 每 head、每调用随 live 特征计算；没有额外可学习 step |
| retained K/V/W | K/V FP32；W 非负，求解时每源再归一化 | 仅成功 clean commit 更新；TBPTT 边界 detach；episode reset 清空 |
| selection Q | 与 clean K 同一坐标、FP32，数值不设硬阈值 | detached，只用于最近 query 累积评分 |
| 旧 ψ / controller / generators | 新候选 ψ 恒零，controller/generators 保留参数键但冻结 | 为保持初始化 RNG/旧加载实现一致；日志明确 disabled，不算活跃创新分支 |

默认 prefix=10 latent frames（含初始 observed frame 与随后 generated clean frames），recent=4，capacity=16 frame-equivalents，余下2 frame-equivalents 是 selected middle。不是16个token，也不是16个chunk。所有新写入 token 必须有真实 patch grid 与绝对 frame ID，padding/重复重叠帧不参与。

在22×40 patch、FP32、5 anchors、20 heads、D112、B=1时，K/V/W/Q/ID 的满缓存逻辑载荷约 **1.769 GiB**，CFG两分支约加倍。这不含 autograd、候选临时张量、allocator、SANA/native cache；训练内存必须实测。容量约束限制持久缓存，不意味着 TBPTT 内的计算图只占这个数值。

Noisy solver 每次重算临时 S，不写入 persistent S、retained cache、committed IDs。Clean forward 从 incoming prior 另算一次，五个 anchor 及缓存检查全通过才提交。Prefill 只使用 observed frame 的当前源。前四 predicted chunks 收集缓存，第五起将历史加入目标。TBPTT=4 同时截断 S 和 K/V/W，只提供窗口内未来 credit。淘汰 token 不再作为显式约束，过去写入 S 的影响仍保留。

## 4. DeepForcing / YUME 的迁移边界

核对的本地 DeepForcing 版本：`ea9961aa4bd5b554daeecdcf59e37c495fba9df6`，路径 `E:/Research/WorldTTT/workspace/DeepForcing`。

| 来源 | 已迁移的机制 | 与原文/原代码的区别 |
|---|---|---|
| [DeepForcing causal_model.py](https://github.com/cvlab-kaist/DeepForcing/blob/ea9961aa4bd5b554daeecdcf59e37c495fba9df6/wan/modules/causal_model.py) | 早期 prefix 保护、recent 保留、有限预算 middle；跨 head 累积 recent Q·K 的参与度 Top-C；按绝对ID稳定排序 | 原来用于 Softmax KV；这里用于 TLA 的回归观测约束，不声称复现原来的 attention 输出 |
| 同一代码的 key temporal realignment | 可选 `memory_position=temporal-realign`，只改变临时读取视图的 K 时间 RoPE，V/原ID不变 | 不复制 `21-capacity//1560` 等模型特定偏移；按实际网格/近期帧构造位置；默认absolute，因为没有camera/world transport |
| DeepForcing clean cache 更新 | 只在 clean commit 发布缓存 | 适配现有 TTN 事务；CFG 分支各自选择，失败不半提交 |
| [YUME](https://github.com/stdstu12/YUME/tree/111c3fab7fb020d1e261a68be6ec78a3fecc8d5b) 的 initial/recent/compressed-far 思路 | 用作多时间尺度记忆组织的参考，和上面的分区一致 | 其多尺度 Conv3d 历史编码依赖不同 backbone/训练；没有把权重不兼容模块直接拼入 SANA |

YUME 的推理重采样/时间回退需要额外 solver 工作，不能视作免费 memory 更新。其 learned compressed-history 路线需要另一轮训练与消融。本版没有引入这两项，也没有把 native GDN 的 recurrent state 冒充可裁剪 KV。

## 5. 指标与对照

每个 clean chunk/anchor/head 保留原来的 S old/pred/current 尺度、K/V/Q RMS、β/W、写入方向；新增 lambda、条件数界、先验保留界、当前/历史 objective、更新比例、两者梯度范数/夹角。Noisy 首/中/末 solver 调用采样这些指标；flow-training调用记录一次。heldout-query 是单独 support-only 求解的诊断，不改变生产写入。

`effective_delta_norm` 是 FP32 投影贡献。`output_effect` 是同输入/camera/gate 下，S* 与 prior 两个 anchor 输出经过实际 dtype 后的 norm、相对变化和 changed_fraction。需两者一起判断是否有有效作用；不能只看非零梯度。Prefix SHA 在形成完整prefix后记录，episode结束审计。评估继续输出逐帧/逐chunk MSE、tail、S spectrum和teacher差异。

最少训练对照：同初始化/数据顺序下 (1) `memory_selection=none` 的current-only Direct-S；(2) `fifo` 的**protected-prefix FIFO**；(3) `participative`；然后单独比较absolute/temporal-realign。这些都是新的训练数学身份，不能用修改JSON绕过 checkpoint config 校验来冒充相同训练。旧双 ψ 是独立baseline。

训练history采用4 solver steps、CFG1、native缓存2chunk；常规质量评估采用20steps、CFG4.5、同缓存2chunk。必须额外运行4steps/CFG1的匹配评估以区分sampler shift。保持固定case/seed；目前历史结果只有一个case，不能推断数据集泛化。新训练前先通过[LTU验收](proximal-memory-ltu.md)。

## 6. 本地验证记录（2026-10-08）

| 检查 | 实际结果 | 范围 |
|---|---|---|
| CPU 主回归，Python3.10 / Torch2.11 | 285 passed，22 skipped，0 failed | FP64 oracle、gradcheck/gradgradcheck、独立 inner/cache future-credit、TBPTT、事务失败、prefix保护、checkpoint/保留/续训、旧默认数学及camera |
| 推理入口与评估身份 | 15 passed，0 failed | 普通固定case和自定义图生视频记录新配置、prefix SHA、缓存字节数；旧模式继续可用 |
| Bash/Slurm 启动器模拟 | 3 passed，0 failed | 单卡和三卡选择正确测试集，非法五任务拓扑在srun前失败；不连接集群 |
| 单GPU：用户 tianshou 环境，Python3.11.14 / Torch2.5.1+cu121，RTX4050Laptop 6GiB | 19 passed，0 failed | D112 FP32求解/外层梯度对齐FP64；缩小尺寸真实anchor代码的BF16训练、offload、实际量化输出影响；native camera回归 |

CPU/GPU JUnit结果在工作区的 `../verification/proximal-final-cpu.xml`、`proximal-callers.xml`、`proximal-launcher.xml`、`proximal-gpu.xml`。跳过项依赖CUDA或隔离的多节点allocation；不算通过。未运行完整SANA模型、LTU FSDP2/NCCL、121latent容量或新候选视频质量实验；旧MSE趋势不能当作本版结果。
