# TTN 第一批工程与加速验收

基线：`origin/main` 的 `392b89f9de25b86f6c63d63202f1c988842483a2`；整合加速提交：`d4f5fb2db432ea78d4b98d565e05fe59a9076fff`。
本轮不提交正式训练。reference/reference 仍是默认；训练数学、camera、loss、S/ψ 更新和训练长度未调整。

## 已实现的行为

- 每套新检查点保存 `model.pt`、所有 `rank-*.pt`、`manifest.json`。SHA256、大小、step、ID、rank/backend/world 在发布 `last.pt` 前校验。两套完整模型与优化器 bundle 保留，非 milestone 优化器不会无限累计。
- `.checkpoint-owner.json` 限定自动清理所属实验；旧格式、外国 owner、部分写入、未知文件的目录不自动删除。`.keep` 或同输出根目录内的 checkpoint `.pt` 引用固定整套 bundle；外部评估快照只需要保留模型硬链接。
- 保存前要求空闲空间覆盖 `1.15 × (两份新增模型 + 所有新增 rank 分片) + 2 GiB`；两份模型预算包含不支持硬链接时的发布复制。已有 bundle 在预算期间继续保留。
- 保存锁协调同输出根目录的保存、清理与快照创建；冲突最多等待600秒，获取锁后再检查磁盘预算。正常错误会解锁。SIGKILL 可能留下锁，确认相应作业退出后人工处理；不会自动删除陌生锁或部分目录。
- gather 前的 reshard/state/局部 detach 准备及每次 gather 后的 rank0 CPU 拷贝都通过 `_phase` 同步局部错误，再进入下一次 tensor collective；还会核对各 rank 的模型键顺序/shape/dtype。`full_tensor()` 通信本身失败时记录参数名和原异常后退出，不再用可能已失效的进程组尝试同步错误。硬 SIGKILL/通信故障仍依靠 Slurm `--kill-on-bad-exit=1` 和进程组 timeout 结束。保存失败时不会提交后继。
- 原日志170 / checkpoint160 的恢复以验证过的160为起点，明确登记未保存161..170，在新目录续训。`--after-job` 等待活跃父任务时暂用其声明终点，worker 必须在启动前重新验证。
- benchmark/profile/layout 默认没有 checkpoint I/O；`--benchmark-save-checkpoint` 显式开启。checkpoint 输入推导 scope 并恢复原数据/profile，仍严格检查 exact-resume 配置。execution core/ψ backend 进入训练 identity；正式 resume/unfreeze 不能切换 backend。旧 identity 缺少该字段时按 reference/reference 兼容。只有 benchmark 允许 backend 差异，其他数据/scope/TBPTT 等配置仍严格相同；`run_config.resume_execution` 和 `[TTN resume execution]` 记录这次豁免。
- 旧格式 saver 不遵守新增保存锁。对旧源创建 benchmark 快照必须先停止写入，再传 `--source-stopped`；存在 `squeue` 时自动检查本用户的 TTN 作业，运行或排队均拒绝，查询失败也拒绝。没有 Slurm 时该选项是人工确认，创建期间不能重启源写入进程。新 manifest bundle 继续由保存锁协调。
- `failure-job*-rank*-pid*.json` 记录首个 Python 异常及 phase/step；空间耗尽可能使 JSON 写入失败，原 stderr traceback 仍保留。job 登记包含 stdout/stderr/workdir，导出不再向 sacct 查询不支持的日志路径字段。
- 仅 Full Stage C 可选 reuse/reference、reuse/projected、compiled/projected；Identity、No-TTT、GT-history、teacher 对齐、阶段诊断要求 reference。

## 本地验证结果

- 本轮完整 `tests/ttn`：514 passed、3 skipped、6 warnings，626.24秒。环境为 Python3.10.16 / PyTorch2.11.0+cpu。包含局部 gather 准备/CPU 拷贝故障注入、tensor collective 原异常记录、backend resume/unfreeze 约束、实际 benchmark 豁免及旧源停训检查。整合提交37506aa的505项通过记录也保留在交付目录；A100证据仍待集群验收。
- 同基线392b89f的 reference 比对：CPU小模型、Full Stage C、非零controller、TBPTT=2；参数、梯度、Adam状态、S/ψ等862个张量逐位一致。
- 整合的加速回归覆盖 reuse/reference、reuse/projected 的输出、S/ψ、outer gradients、optimizer update、TBPTT=1/2/4；使用原有容差。compiled 的CPU检查采用 aot_eager，不代表生产CUDA/Inductor已验收。
- 5个启动脚本与文档7段Bash指令通过语法检查；补丁在干净392b89f基线上通过 `git apply --check`。
- 分支审查发现的损坏bundle保留、backend覆盖、快照锁竞争、首错导出与清理错误覆盖原异常问题均已修复并加入回归。

CUDA/A100正确性、真实耗时和显存结果尚未取得；下面的LTU步骤用于补齐这些证据。

## 同步到 LTU 独立目录

从 GitHub 的独立分支 `codex/ttn-retrain-v2` 拉取到新目录，然后执行：

```bash
export ROOT=/data/group/zhaolab/home/z2zhang/huangyu
export PYTHON="$ROOT/envs/worldttn/bin/python"
export PROJECT_ROOT="$ROOT/WorldTTN-retrain-v2"
test -x "$PYTHON" || exit 1
test ! -e "$PROJECT_ROOT" || { echo '目标已存在，请先检查，不覆盖'; exit 1; }
git clone --branch codex/ttn-retrain-v2 --single-branch https://github.com/HuangYu227/N.git "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
source tools/ttn_cache_env.sh
"$PYTHON" -c 'import sys, torch; print(sys.executable, torch.__version__, torch.version.cuda)'
git diff --check
```

不运行 pip/conda 安装。若目标已存在，只在确认它属于这批工作且分支/改动匹配后复用。现有 WorldTTN 训练目录不切分支、不应用补丁。交付包 `ttn-retrain-v2.patch` 仍可用于离线同步，以392b89f为基线；不要重复应用到已拉取的新分支。

## 1. CPU 工程检查与只读旧检查点验证

```bash
cd "$PROJECT_ROOT"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON" -m pytest -q \
  tests/ttn/test_checkpoint_integrity.py tests/ttn/test_checkpoint_failure_distributed.py \
  tests/ttn/test_slurm_chain.py tests/ttn/test_execution_cli.py \
  tests/ttn/test_benchmark_snapshot.py tests/ttn/test_export_analysis.py

export SOURCE_RUN="$ROOT/WorldTTN/output/worldttn/ucpe-C-20261004T063947Z/joint"
"$PYTHON" -m tools.ttn_check_checkpoint "$SOURCE_RUN/last.pt"
```

旧bundle预期显示 `legacy_unverified`：所有分片元数据已检查，旧格式没有 SHA256 manifest。缺失/损坏则停止，不提交基准或续训。`--prune` 才显式应用管理目录的保留规则；本步骤没有删除。

## 2. 固定完整基准快照

这里的 SOURCE_RUN 来自旧保存器，先只读检查作业列表。确认源训练和续训链已结束，且快照创建期间不会重新提交源训练，再执行带 `--source-stopped` 的创建命令；本步骤不会取消任何作业。脚本保守阻止本用户所有名称以 `ttn` 开头的运行/排队任务，便于此次停训验收。

```bash
squeue --me --noheader --format='%i|%j'
export BENCHMARK_SNAPSHOT="$PROJECT_ROOT/output/benchmark-snapshot-$(date -u +%Y%m%dT%H%M%SZ)"
"$PYTHON" -m tools.ttn_benchmark_snapshot --training-run "$SOURCE_RUN" --output "$BENCHMARK_SNAPSHOT" --source-stopped
"$PYTHON" -m tools.ttn_check_checkpoint "$BENCHMARK_SNAPSHOT/last.pt"
```

模型与 optimizer/RNG/cursor 都硬链接到快照，不复制几十GiB。创建时持有源保存锁；新保存器会协调等待，旧保存器必须保持停止。快照独立于源后续保留清理。

## 3. A100 算子正确性，再做真实四卡 update 验收

第一组是单卡算子测试，涵盖 d=112、非连续布局、Correct/Read、ψ梯度和 outer gradients：

```bash
cd "$PROJECT_ROOT"
OP_JOB=$(sbatch --parsable --time=00:30:00 tools/ttn_slurm_operators.sbatch --variants V0 V1 V2 --repeats 20 --warmup 5)
echo "OP_JOB=${OP_JOB%%;*}"
```

确认上述正确性通过后依次提交 V0、V1、V2，**一次只提交一组，检查结果后再继续**。每个 benchmark 独立恢复同一快照，执行一个冷 update + 10个稳定 update。它会在私有输出中优化参数，绝不写回源快照，也不产生 checkpoint。

```bash
export BENCHMARK_MODE=throughput STABLE_STEPS=10
export GDN_DISABLE_COMPILE=1 CUDA_LAUNCH_BLOCKING=0
unset TORCH_LOGS TRAIN_SCOPE OUTPUT

# V0：reference；记录此结果作为四卡基准
TTN_CORE_BACKEND=reference TTN_PSI_BACKEND=reference \
  sbatch --time=01:00:00 tools/ttn_slurm_benchmark.sbatch

# V0结束并通过后：V1
TTN_CORE_BACKEND=reuse TTN_PSI_BACKEND=reference \
  sbatch --time=01:00:00 tools/ttn_slurm_benchmark.sbatch

# V1结束并通过后：V2
TTN_CORE_BACKEND=reuse TTN_PSI_BACKEND=projected \
  sbatch --time=01:00:00 tools/ttn_slurm_benchmark.sbatch
```

每组查看 `output/benchmark-JOBID/performance.json`、`first_update.json`、`train.jsonl`、`slurm-ttn-perf-JOBID.out`。要求四 rank 有限、核心梯度无缺失、同一checkpoint/hash/scope，除 execution backend 外 training identity 相同，冷 update 单列；稳定耗时取最慢 rank。V1/V2的 `resume_execution.benchmark_override` 应为 true，V0为 false。检查不存在 `last.pt` 或 `last-resume-*`，排除保存开销。比较逐步 loss 及现有容差内的行为，不能仅凭速度批准切换。

真实尺寸 CUDA/offload backward 回归测试单独申请一张GPU，并在进程启动前建立本地编译缓存：

```bash
srun --partition=short --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:30:00 --mpi=none \
  bash -c '
    set -euo pipefail
    cd "$PROJECT_ROOT"
    acceptance_tmp="$(mktemp -d /tmp/worldttn-acceptance-${SLURM_JOB_ID}.XXXXXX)"
    export TMPDIR="$acceptance_tmp/tmp" TMP="$acceptance_tmp/tmp" TEMP="$acceptance_tmp/tmp"
    export TORCHINDUCTOR_CACHE_DIR="$acceptance_tmp/inductor" TRITON_CACHE_DIR="$acceptance_tmp/triton"
    export CUDA_CACHE_PATH="$acceptance_tmp/cuda" TORCH_EXTENSIONS_DIR="$acceptance_tmp/extensions"
    export PYTHONPYCACHEPREFIX="$acceptance_tmp/pycache" TORCHINDUCTOR_COMPILE_THREADS=1
    mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
      "$CUDA_CACHE_PATH" "$TORCH_EXTENSIONS_DIR" "$PYTHONPYCACHEPREFIX"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON" -m pytest -q \
      tests/ttn/test_compiled_ttn.py tests/ttn/test_acceleration.py
  '
```

V3最后测：先运行算子 V3，再按相同脚本设置 `TTN_CORE_BACKEND=compiled TTN_PSI_BACKEND=projected`。`compile_warmup` 必须证明稳定窗口无新 specialization/recompile；编译/预热时间单列。

## 4. 同权重 rollout 比对

使用同一个 immutable checkpoint、固定case、seed3407、20采样步、camera window=2，对优化路径与 TTN reference 直接比较 latents 和各 chunk 的 S/ψ。13和61 latent长度都执行；不把与SANA的性能差异当成加速正确性证据。

```bash
export TRAINING_RUN="$BENCHMARK_SNAPSHOT"
export FIXED_CASES="$ROOT/WorldTTN/output/worldttn/ucpe-C-20261004T063947Z/fixed-cases.pt"
export COMMAND=evaluate CROSS_ATTN_BACKEND=math CACHED_BLOCKS=2 STEPS=20 SEED=3407
export TTN_COMPARE_REFERENCE=1
unset TTN_ABLATION STATE_DIAGNOSTICS HISTORY_SOURCE CAMERA_ATTENTION CAMERA_ABLATION EVAL_METHODS ADAPTER OUTPUT

# 每次指定明确 backend 与长度，完成后检查 summary.backend_comparison。
FRAMES=13 TTN_CORE_BACKEND=reuse TTN_PSI_BACKEND=projected \
  sbatch --time=01:00:00 tools/ttn_slurm_eval.sbatch
FRAMES=61 TTN_CORE_BACKEND=reuse TTN_PSI_BACKEND=projected \
  sbatch --time=01:00:00 tools/ttn_slurm_eval.sbatch
```

对V1/V3重复上述两个长度；V3仅在前面的 compiled 正确性通过后执行。阶段teacher/mechanism评估继续 reference/reference。

本地验证使用 CPU PyTorch 2.11，不能代替 LTU Python3.11 / PyTorch2.9.1+cu128。A100 gate 通过后的下一批是 H1/H2 实现和评估，不直接开始 retraining；加速默认值也要以 A100 正确性、耗时和显存结果为依据。

## 重训前最后检查：直接联合训练、指标与权重保留

本轮完整 CPU 回归为530 passed、4 skipped、6 warnings（709.74秒）；之后修补 prefill 摘要与手工 `.keep` 保护，重跑受影响的 stability/snapshot/chain/stage-evaluation 检查为63 passed、1 skipped（8.82秒）。reference 对原主分支的862张量仍逐位一致，新增 detached 写入诊断字段不参与旧日志字段比较；此比对使用 legacy optimizer 验证原数学，新 origin LR 分组另由实际 update/分组测试验证。验收文档9段 Bash 及启动脚本语法检查通过。以上都是 CPU 证据，GPU 验收尚未运行。

本轮遵循用户确认的初始化：五个 anchor 的 QKV、Norm、proj、output_gate 及 native camera 保留 SANA 预训练权重；只初始化新增 TTN 参数。新增参数沿用原初始化规则（beta/controller 输出头为零，controller 主体随机，generators 随机归一化），不是把全部权重改成随机数。每个 episode 的 S/ψ 从零开始，属于运行状态，不是 optimizer 参数。

### 参数来源与对照

- 新联合实验 `train_scope=dit, optimizer_policy=origin`：`beta_proj`、controller、generators 使用 `ttn_new` 组 LR=1e-5；所有继承 SANA 参数，包括五个 anchor 内的视觉映射及相机参数，使用 `sana_inherited` 组 LR=1e-6。从首次 optimizer update 开始联合训练。
- 可选 corrected-warmup control：`train_scope=ttn-new` 只训练新增参数50步，再解冻 DiT。它与历史 `ttn-visual` warmup 的参数范围不同；历史结果不能冒充这个新对照。
- 旧权重没有 policy 字段时按 legacy 续训，保留旧 Adam 分组。禁止 exact resume/unfreeze 静默切换 policy；新实验使用新输出目录，不从历史 TTN adapter 初始化。
- 当前保留 native SANA camera、GT-history 训练、97 raw / 13 latent、TBPTT=2、原 flow loss 和 S/ψ 数学。训练全 clip camera cache / 推理 window=2 的范围差异仍需 H1/H2 控制实验，不能宣称已消除。

### 指标在哪里、什么时候记录

| 文件/阶段 | 记录内容 |
|---|---|
| `run_config.json` | base SHA256、初始化来源、seed、参数名/来源组/LR、训练与推理 protocol、执行 backend |
| `first_update.json` | 本次调用首个真实 optimizer update 的梯度、参数变化、optimizer state 与缺失核心梯度；不是只运行诊断 backward |
| `train.jsonl` / `stability.jsonl` | 每 step、rank、clean chunk、anchor 的 innovation loss/RMS/相对 V 残差、Q/K/V 与 K@S 尺度、S_pred/S_new/ΔS norm/RMS、beta/w/eta_s、写入数、raw/clipped inner grad、ψ 与实际 Δψ、controller 系数、outer grad、耗时/显存及 anchor 梯度尺度 |
| 新 `write_direction` | 每 head 的 lag 1/2/4 写入 cosine：原始矩阵方向与按实际 Cayley transport 对齐后的方向。只保留最近4次 detached 诊断写入；零写入为 null，历史不足时不生成对应 lag |
| 每25步 `evaluations/step-*/` | 固定 case/seed3407，同一 checkpoint 的 teacher 分段/跨层对齐与13/61 latent rollout；每 head 的 S_prev/S_pred/S_new/ΔS top singular values、stable rank、top1 energy fraction；GT MSE、最后 chunk MSE、误差斜率及具备有效回访时的回访指标 |

13 latent 训练只有4个预测 chunk，lag4 不会出现；61 latent rollout 才覆盖这个 lag。训练不做 SVD，谱诊断留在阶段评估中。FSDP 梯度明示为裁剪后的本地分片尺度，不能把一个 rank 的数值当全模型梯度。固定 case 仍是诊断样本；没有独立测试集或有效回访时，不能据此宣称泛化或回访性能。

实时摘要中的 `max clean innovation/V` 只统计预测 chunk，避免 S=0 的 prefill 固定基线1遮住训练变化；prefill 完整指标继续保存在 JSON 日志。

低 stable rank 与高写入 cosine 是待解释的关联，尚不能证明 memory capacity 是性能瓶颈。坐标变化和同场景重复观察都影响解释；本轮没有加入 write-diversity loss、retention、global gate、clamp 或新的训练目标。诊断的额外耗时/显存要在 A100 验收中实测。

### 权重保留

- 每个成功训练段保存一次；新完整 bundle 验证后，只保留最新两套可续训的模型+优化器/RNG/游标，保证失败回退。
- 每25步评估，默认长期保留模型点为25、50、100、250及最终 target（500）。可用 `--keep-model-steps` 改关键点；没有评估的 step 不会因此额外保存模型。
- 其他评估点在 long/short/align 全部成功且 checkpoint 身份一致后，只释放该快照的 `last.pt` 硬链接；日志、指标和 protocol 永久保留。失败/排队评估保留权重供排查；旧 snapshot、显式 retain、带 `.keep` 的快照及手工/机制快照不自动删除。
- 删除一个硬链接不一定立即释放磁盘块；仍被续训包/其它快照引用的 inode 继续保留。待评估、失败或未知的部分保存目录会增加空间，不能把“两套”当整个输出的硬上限。
- 当前完整 FP32 模型约10 GiB，四 rank 优化器等约20 GiB。两套续训包加五个关键模型约110 GiB上限估算（硬链接重合时更少），保存还需要临时空间。169 GiB是用户上次清理后的历史值，不代表现在仍有这么多空间；以保存前的实际路径预算为准。

### LTU 上新增分组的一个 update 验收

这是一次验收，不是500步正式训练。先完成上述 A100 正确性检查；下面用全新 TTN 和原 SANA 初始化，reference/reference，检查真实四卡梯度、分组与完整保存。

```bash
cd "$PROJECT_ROOT"
unset ADAPTER RESUME UNFREEZE OUTPUT BATCH_FILE BENCHMARK_SNAPSHOT BENCHMARK_MODE
export COMMAND=train CONFIG="$PROJECT_ROOT/configs/worldttn/reference_sana_camera.json"
export DATASET_ROOT="$ROOT/datasets/sana-wm-example"
export STAGE=C TRAIN_SCOPE=dit OPTIMIZER_POLICY=origin PARALLEL=fsdp2
export BACKBONE_LR=1e-6 MAX_STEPS=1 SAVE_EVERY=1 TBPTT=2 SEED=3407
export ACTIVATION_OFFLOAD=cpu TEXT_ENCODER_DEVICE=cpu CROSS_ATTN_BACKEND=math
export TTN_CORE_BACKEND=reference TTN_PSI_BACKEND=reference MEMORY_TRACE=1 CUDA_TRACE=0
export GDN_DISABLE_COMPILE=1 CUDA_LAUNCH_BLOCKING=0 DISTRIBUTED_TIMEOUT=1800
CHECK_JOB=$(sbatch --parsable --mem=256G --time=01:00:00 tools/ttn_slurm_train.sbatch)
CHECK_JOB=${CHECK_JOB%%;*}
echo "CHECK_JOB=$CHECK_JOB"
```

完成后核对 `slurm-JOB-C/run_config.json` 的 policy/origin groups，`first_update.json` 四 rank 核心梯度无缺失、继承 anchor 参数实际参与更新，`stability.jsonl` 全部有限，`ttn_check_checkpoint .../last.pt` 的 manifest/分片校验通过；记录最慢 rank 耗时和每卡峰值。训练记录应为 `train_scope=dit`、step1，不能把验收的一步算作新正式实验的起点。

### 后续正式启动方式（本轮不执行）

先取得 A100 验收证据并完成 H1/H2，再使用独立目录创建新的主实验。`fresh` 不读 ADAPTER，也不继承交互终端残留的训练 profile；本轮参考执行方式仍是 reference/reference。该示例会提交 Slurm 作业，不能作为只读检查命令运行。

```bash
cd "$PROJECT_ROOT"
"$PYTHON" -m tools.ttn_slurm_chain fresh \
  --warmup-steps 0 --target-step 500 --segment-steps 10 \
  --eval-every 25 --keep-model-steps 25 50 100 250 500 \
  --output "$PROJECT_ROOT/output/worldttn/direct-joint-$(date -u +%Y%m%dT%H%M%SZ)"
```

corrected-warmup control 使用另一个输出目录并把 `--warmup-steps` 改成50；不要同时提交两套作为这次验收。长度、数据、seed、LR来源分组与其余 protocol 应一致，后续解释再分别报告 optimizer steps、chunk exposure 与 GPU-hours。
