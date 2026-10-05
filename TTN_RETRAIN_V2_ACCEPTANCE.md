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
