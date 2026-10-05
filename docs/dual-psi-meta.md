# 双时间尺度 ψ 与可微 Meta-TTT

本分支以 c9f2970 为基线。默认配置继续使用旧 delayed、detached ψ；
`configs/worldttn/dual_psi_meta.json` 显式开启 Local 与 Persistent meta。
本批交付实现和验收入口，不启动正式重训。

## 本地验证结果

提交 1f1eaa3 的本地验证（2026-10-05），Windows / Python3.10.16 / PyTorch2.11.0+cpu：
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/ttn -q --tb=short`
完成 **583 passed、12 skipped**。跳过项不能作为 CUDA/A100 验收结果。
覆盖 FP64 gradient/meta oracle、gradcheck/gradgradcheck、原 SANA teacher replay、
真实 sampler loop、CPU FSDP2/offload/future credit 和 exact resume。
另通过 Python 编译检查、Git whitespace 检查、改动脚本及 LTU 文档 13 段 Bash 语法检查。
未运行真实 DiT/CUDA 四卡验收，未启动正式训练。
下一步按 [LTU 验收指令](dual-psi-meta-ltu.md) 顺序执行。

## 网络、初始化与参数归属

五个 anchor 为 3/7/11/15/19，每个 20 heads、D=112、16 个 rank-2 generators。
QKV、Norm、proj、output_gate、native camera 投影/Norm/UCPE/SDPA/cache
继承 SANA 权重，从 step 0 以 SANA-inherited LR=1e-6 联合训练。

TTN-new 使用 LR=1e-5：β 为 2240→20 的零初始化 Linear 后接 sigmoid；
controller 沿用局部 14→128→128、全局 6→128→128 的 SiLU 网络，加上
4 维 intrinsics 后由五个 260→320 的零初始化输出头生成系数。
U/V 各为 5×20×16×112 的随机单位向量。新增五个 Local η logits，
softplus 后初值 .01，常数初始化不消耗 RNG。

ψ 不是 MLP 或 optimizer parameter。Persistent ψ 是 episode 内的 fast state；
Local ζ 是一次 noisy call 内的临时系数状态。本文沿用“双时间尺度 ψ”作为机制名称，
公式明确为两个独立 tanh 的系数增量，并非 tanh(ψP+ζ)。没有前 50 step 冻结适配。

## 单 head 的数学

令 P=[u1,v1,…,uM,vM]，A(C)=Σ Cm(um vmᵀ−vm umᵀ)，B=I−A/2。

\[
C=C_{base}+\delta_P\tanh\psi^P+\tanh\zeta,
\quad R=B^{-1}(I+A/2),\quad T=S_{old}R.
\]

实现 R=I+P L Pᵀ，只做 2M=32 阶 solve。δP 沿用原配置 1，δL 固定 1。
精确算术下 A 为斜对称矩阵，R 正交，因此 Transport 保留 S 的 Frobenius
范数和奇异值。它不能单独修复低 stable-rank，也不保证完整 recurrence 范数有界。

### Local：当前 noisy 输出立即使用

沿用 w=mβ/(eps+mean_valid β)。实际 patch grid (T,H,W) 和 episode 内绝对
latent frame ID 定义 support：有效新写 token 中 (x+y+f)%2==0；query 是余集。
padding、已提交重叠帧不参与。维度不一致报错，空 support 零更新，空 query 指标 null。

\[
L_L(\zeta)=\frac{\sum_{i\in\mathcal S}w_i\|K_iT(\zeta)-V_i\|^2}
{\epsilon+\sum_{i\in\mathcal S}w_i\|V_i\|^2},\qquad
g_L=\left.\nabla_\zeta L_L\right|_{\zeta=0}.
\]

V 来自当前 noisy hidden feature，没有 GT。逐 head clipping γ=1：

\[
\psi_L^*=-\operatorname{softplus}(a_\ell)\,
g_L\min(1,\gamma/\max(\|g_L\|,\epsilon)).
\]

第一版没有噪声门控。从 **S_old** 用适配后的系数重新 Transport，
再调用原 Correct/Read；不会对 T(0) 再右乘一次 R。不同 CFG branch 独立计算
其 gradient/update，η 仍是共享的 per-anchor slow parameter。

\[
S_{tmp}=T_{eff}+\frac{\alpha_S}{\epsilon+\sum_iw_i\|K_i\|^2}
 K^T W(V-KT_{eff}),\qquad O=Q S_{tmp}.
\]

随后仍由原 camera fusion、output gate、proj 输出。Local 不写 runtime、
camera cache 或 committed-frame registry，每个 solver timestep 重新计算。

### 可微解析梯度

E=KT−V，Local 的 H=dL/dT=2KᵀW_support E/(eps+Σ_support w‖V‖²)。
Cayley 满足 dR=B⁻¹(dA)B⁻¹。定义 X=S_old B⁻¹P，Y=H B⁻ᵀP，得到

\[
\frac{\partial L}{\partial C_m}=
\langle X_{u_m},Y_{v_m}\rangle-\langle X_{v_m},Y_{u_m}\rangle.
\]

因为 B⁻¹P=P+.5 P L(PᵀP)，只需现有低秩因子。
生产路径用普通 PyTorch tensor 运算，无 nested autograd.grad、dense inverse
或新 CUDA/Triton kernel。S、K/V、β、U/V、controller、分母、clipping scale
都保持 live；统计值才 detach。flow loss 经过 gL 返回 θ，因此包含
∂gL/∂θ=∂²LL/(∂ζ∂θ)，不是 detached first-order 近似。

### Persistent：clean commit 后，下一个 chunk 生效

Clean 禁用 Local，使用旧 ψP。保持原 raw weighted innovation objective：

\[
L_P=\frac{\sum_iw_i\|KT_P-V\|^2}{2DN_{write}},\quad
H_P=K^TW(KT_P-V)/(DN_{write}),
\]
\[
g_P=\delta_P(1-\tanh^2\psi^P)\,\partial L_P/\partial C,
\qquad\psi^P_{next}=\psi^P-.01\,clip_1(g_P).
\]

开启 persistent_meta 后，gP、clipping 和 ψP_next 保留图；仅 TBPTT 边界
同时 detach S/ψ。当前 clean S 使用旧 ψP，新 ψP 滞后一 chunk。
TBPTT=2 只提供窗口内下一 chunk 的 future credit，不代表整个视频的梯度。

Correct 包含 (I−η KᵀWK) 的 observation-conditioned implicit forgetting。
固定 K/V/W 时 αS∈(0,2) 使齐次项非扩张；V 驱动项、变化的 features 和
有限精度仍使完整序列没有统一范数上界。本批不加 retention/global write gate。

## 生命周期、FSDP 与兼容

五个 anchor 全部完成且候选有限后，原子发布 clean S/ψ。Noisy forward
从不 commit；prefill 使用 identity Transport，不执行 Local/Persistent adaptation。
FSDP clean 输出 hook 同时暴露 S 和 live gP，再恢复原 SANA block 接口。
side-channel 的梯度因此能触发 unshard/reduction hooks。
stage()/commit_chunk() 不放入 checkpoint recomputation 区域。

两个开关默认 false，缺失字段的旧 checkpoint 按 false 解释。四种训练配置
分别为 false/false、true/false、false/true、true/true。新数学进入 checkpoint
和 exact-resume identity，不能用旧 optimizer resume 静默升级结构。
S/ψ/计算图不保存，仍只在完整 clip 后保存。

Meta 首版仅 reference/reference，报告 psi_implementation=live_projected_meta。
旧 optimized backend 与任一新开关组合在 forward 前报错。
本批不新增加速实现或加速实验；普通 reference 四卡验收记录耗时与显存，检查计算图生命周期。

## 指标与解释边界

train.jsonl 的 ranks/chunks/anchors 与 stability.jsonl 记录：

- Local support/query 的 before、after_transport、after_correct loss/residual RMS；
  raw/clipped gradient、clip fraction/scale、η、ψL norm、Transport change。
- Persistent raw/clipped gradient、ψP norm/实际 update、tanh saturation；
  S old/predicted/committed norm/RMS，Correct ratio，K/V/prediction/residual RMS，β/W 分布。
- noisy model timestep 与实际 σ；训练与真实 rollout 均直接读取 scheduler.sigmas，
  原始 sampled timestep 单独记录，不能把 controller τ 当作噪声或累计视频时间。
- 首次更新审计的 parameter_update.by_origin、模块级 delta/gradient，FSDP 明确为 local shards；
  后续每步保留 anchor_gradients、Local/Persistent 与状态稳定性记录；
  exposure 包含 clips/有效 latent frames/预测 latent frames/预测 chunks；耗时和显存沿用原记录。

实际更新审计暂存 CPU 参数副本，真实 DiT 约 2.5 GiB/rank，逐参数差值还需要临时空间。
CLI 仅审计本次进程的首个更新（续训首步也检查）；train_clip 默认不复制参数，
显式 audit_update=True 才额外生成 optimizer_updates。不能把首步当作稳定吞吐。
Benchmark 的 iteration_seconds 包含 CLI 审计；train_update 的 seconds 不包含外层审计。
global norm 用各 shard norm 的平方和开方，不能直接平均 shard norms。

普通推理不采集 noisy Local 统计或序列化噪声信息；明确开启 state_diagnostics 的
评估、teacher 对齐以及训练会采集。TTNSession/Runtime 也可显式 collect_local_stats=True。
这些开关仅控制观测，不进入模型数学配置或 exact-resume identity。
训练仍保留每 step/chunk 的原有完整 Local/Persistent 指标。
显式观测时，各 anchor 的 local 保存最后一次完整统计，local_trajectory 另外保存每次
noisy call 的 call index、真实 σ/timestep、raw/clipped gradient norm、ζ norm、
Transport change、support/query 的 pre-Correct loss。Clean 不增加轨迹条目。
诊断采集仍有 GPU→CPU 同步开销，耗时报告必须标明是否开启；此改动没有新的 A100 性能结果。

query 仅从 inner loss 排除；Correct 后仍使用它自己的 V，因此 after_correct 的
query 降低不是独立泛化证据。不能保证每次 learned Local step 都降低 support loss。
单 case latent MSE 也不能当作完整视频质量结论。

每 25 step、seed3407 保留 original/matched teacher、13/61 rollout 和谱/写入诊断。
周期任务额外做 13-frame no-local/no-persistent；61-frame 六组机制用独立顺序 array。
关闭分支只测当前贡献。meta-gradient 的训练价值需要四种训练配置的对照。
没有有效 revisit pair 的指标继续为 null。

保存沿用 manifest/SHA256、原子发布、空间预算、引用保护：最新两套完整 bundle；
长期模型点 25/50/100/250/最终500；每25结果/日志保留。非关键模型只在
全部评估子项通过后释放，失败/固定引用保留，旧实验不自动清理。

## 验收与后续

CPU oracle/gradcheck/gradgradcheck、真实 anchor 的即时输出、隔离 Persistent
future credit、camera、offload、FSDP、checkpoint/resume 和清理保护均有 runnable tests。
CUDA/真实 DiT/四 A100 的结果必须由 [LTU 验收步骤](dual-psi-meta-ltu.md) 产生。

正式科研顺序：完整 Meta-TTT 验收 → H1/H2 → generated-history 训练 →
961 raw/121 latent 长程训练。工程验收的 97 raw/13 latent/TBPTT2 不是正式训练。
长短序列报告同 optimizer steps 和近似同 exposure/compute 的两种对照。
