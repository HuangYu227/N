# Proximal memory：LTU 验收与训练顺序

数学与来源见 [proximal-memory.md](proximal-memory.md)。以下是准备好的指令，本地编写文档不会提交任何服务器任务。本分支已发布到 GitHub；按用户当前资源使用四卡。旧的 full-memory/dual-psi 作业不会自动改成新方案。

## 1. 发布后建立独立目录

在登录节点 Bash 执行；永久文件仍在共享 NFS，worker 为每个节点分配 `/tmp` 编译目录，使用原 Conda 环境。

```bash
export ROOT=/data/group/zhaolab/home/z2zhang/huangyu
export PROJECT_ROOT="$ROOT/WorldTTN-proximal-memory"
export PYTHON="$ROOT/envs/worldttn/bin/python"
git -C "$ROOT/WorldTTN-full-training" fetch origin \
  refs/heads/codex/ttn-proximal-memory:refs/remotes/origin/codex/ttn-proximal-memory
git -C "$ROOT/WorldTTN-full-training" worktree add --detach "$PROJECT_ROOT" \
  refs/remotes/origin/codex/ttn-proximal-memory
cd "$PROJECT_ROOT"
git log -1 --oneline
"$PYTHON" -c 'import torch; print(torch.__version__, torch.version.cuda)'
df -h "$PROJECT_ROOT"
squeue -u "$USER" -o '%.18i %.20j %.12T %.10M %.4D %R'
```

已有同名目录时不要覆盖/删除；复用前核对其 commit。LTU 预期 Python3.11、Torch2.9.1+cu128。更改卡数必须是 fresh run，旧 world-size checkpoint 不可 exact resume 到另一拓扑。下面使用四卡验收，不固定节点，不绕开 Slurm 分配。

## 2. 单卡算子与四卡 FSDP2 小模型门槛

```bash
export PROXIMAL_MEMORY_TESTS=1 FULL_MEMORY_TESTS=0 GDN_DISABLE_COMPILE=1
OP_JOB=$(sbatch --parsable --nodes=1 --ntasks=1 --mem=32G --time=00:30:00 \
  tools/ttn_slurm_meta_tests.sbatch)
OP_JOB=${OP_JOB%%;*}
META_JOB=$(sbatch --parsable --nodes=4 --ntasks=4 --mem=32G --time=00:30:00 \
  --kill-on-invalid-dep=yes --dependency="afterok:$OP_JOB" \
  tools/ttn_slurm_meta_tests.sbatch)
META_JOB=${META_JOB%%;*}
printf 'OP_JOB=%s\nMETA_JOB=%s\n' "$OP_JOB" "$META_JOB"
sacct -X -j "$OP_JOB,$META_JOB" --format=JobID,State,ExitCode,Elapsed
```

单卡包括 FP64 oracle、meta 导数、真实 anchor/native camera、BF16 输出以及 offload 小模型。四卡包括 isolated-cache future credit、TBPTT截断、offload梯度、保存恢复后模型/Adam/RNG/游标/runtime。四卡测试模型很小，只检验分布式接线，不测完整模型容量。任一失败都先看首个失败，不把随后 NCCL 退出当成根因。

四卡的 live-cache、detached-cache-control、offload/resume 三组分别在独立 `srun` step 中执行，使用新 Python 进程和不同 rendezvous 端口。每组四个 rank 并行，三组依次执行；失败立即停止。主作业日志记录 `[TTN acceptance]`，详细日志在 `output/meta-tests-$META_JOB/case-{0,1,2}/rank-{0,1,2,3}.out`，包含测试 ID 和 chunk/forward/backward 阶段。此隔离防止各 rank 在不同 pytest 测试中重新建立通信组；不改变模型数学、通信超时或比较阈值。

## 3. 完整 SANA，25 latent，2 steps

只有上一步全部 COMPLETED/0:0 后执行。数据目录默认使用原项目目录；如果实际数据根不同，只修改 DATASET_ROOT。

```bash
export CONFIG="$PROJECT_ROOT/configs/worldttn/proximal_memory_acceptance.json"
export DATASET_ROOT="$ROOT/datasets/sana-wm-example"
export COMMAND=train PARALLEL=fsdp2 STAGE=C TRAIN_SCOPE=dit OPTIMIZER_POLICY=origin
export TBPTT=4 MAX_STEPS=2 SAVE_EVERY=2 BACKBONE_LR=1e-6 SEED=3407
export ACTIVATION_OFFLOAD=cpu TEXT_ENCODER_DEVICE=cpu CROSS_ATTN_BACKEND=math
export TTN_CORE_BACKEND=reference TTN_PSI_BACKEND=reference MEMORY_TRACE=1
export DISTRIBUTED_TIMEOUT=1800 GDN_DISABLE_COMPILE=1
unset ADAPTER RESUME UNFREEZE BATCH_FILE TTN_ENTRY_MODULE
export OUTPUT="$PROJECT_ROOT/output/proximal-25-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUTPUT"
CHECK_JOB=$(sbatch --parsable --nodes=4 --ntasks=4 --partition=day --mem=256G \
  --time=02:00:00 --output="$OUTPUT/slurm-%j.out" tools/ttn_slurm_train.sbatch)
CHECK_JOB=${CHECK_JOB%%;*}
printf 'CHECK_JOB=%q\nCHECK_RUN=%q\n' "$CHECK_JOB" "$OUTPUT" > "$PROJECT_ROOT/output/proximal-check.env"
tail -F "$OUTPUT/slurm-$CHECK_JOB.out"
```

256GiB 是起始申请，不是已经验证的内存上限；新缓存和 solve 需要重新测量。不要因为显卡有空闲就忽略主机内存分配。正常训练在step2保存一套完整bundle；验证前不会自动启动500step。

结束后查看以下记录并进行完整SHA审计：

```bash
source "$PROJECT_ROOT/output/proximal-check.env"
sacct -X -j "$CHECK_JOB" --format=JobID,State,ExitCode,Elapsed
"$PYTHON" -u - "$CHECK_RUN" <<'PY'
import json, sys
from pathlib import Path
from worldttn.checkpoint_integrity import audit_checkpoint
p=Path(sys.argv[1])
for line in (p/'train.jsonl').read_text().splitlines():
    r=json.loads(line)
    print('step', r['step'], 'loss', r['loss'], 'seconds', r['seconds'])
    for rank in r['ranks']:
        phases=rank.get('memory_phases', [])
        print('rank', rank['rank'], 'sampled RSS GiB',
              max([x.get('host_rss_bytes',0) for x in phases] or [0])/2**30,
              'peak GPU GiB', rank.get('peak_allocated_bytes',0)/2**30)
        a=rank['chunks'][-1]['anchors'][-1]
        print('last chunk/anchor', a['block'], 'proximal', a['proximal'])
print(json.loads((p/'first_update.json').read_text()))
print('验证 checkpoint SHA256...', flush=True)
print(audit_checkpoint(p/'last.pt', expected_step=2))
PY
```

核对核心梯度无缺失、β和继承参数确实更新、ψ明确disabled、历史参与度及output_effect有实际数值、prefix审计无异常。空query的null不算失败；不要要求每一个query residual都下降。无穷、NaN、跨窗口持续上涨的活跃内存或 cache超预算需要先修复。

## 4. 完整121 latent，至少3 steps，再做真实恢复对照

25latent检查通过后，新的fresh目录运行正式长度。此时只改horizon，不从短程优化器续训，以免混淆恢复身份；两个config的其他设置相同。

```bash
export CONFIG="$PROJECT_ROOT/configs/worldttn/proximal_memory.json"
export MAX_STEPS=3 SAVE_EVERY=3
export OUTPUT="$PROJECT_ROOT/output/proximal-121-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUTPUT"
LONG_JOB=$(sbatch --parsable --nodes=4 --ntasks=4 --partition=day --mem=256G \
  --time=06:00:00 --output="$OUTPUT/slurm-%j.out" tools/ttn_slurm_train.sbatch)
LONG_JOB=${LONG_JOB%%;*}
printf 'LONG_JOB=%q\nLONG_RUN=%q\n' "$LONG_JOB" "$OUTPUT" > "$PROJECT_ROOT/output/proximal-long.env"
```

检查40chunks/step是否全部完成、窗口边界活跃GPU分配是否回落、RSS是否达到稳定平台，分别报告冷步和稳定步。历史log的总RSS可能含已释放但allocator保留内存，不能只凭RSS未下降判定泄漏。6小时是申请上限，实际时间尚未实测。

相同拓扑和数据/随机种子完成两个独立fresh运行：A连续2步；B保存1步→退出→`ADAPTER=.../last.pt RESUME=1 MAX_STEPS=2`在新OUTPUT续一步（CONFIG仍是121latent）。对A/B的最终目录使用已有 `python -m tools.ttn_compare_resume A_RUN B_RESUMED_RUN`。不能用校验SHA成功代替恢复一致性；若参数/Adam不同，先区分初始化、history noise、游标与CUDA算子差异，不宣称exact resume通过。

## 5. 质量评估与500step

在固定snapshot上复用 `tools/ttn_slurm_eval.sbatch`。常规评估 `COMMAND=stage-evaluate FRAMES=121 STEPS=20 CFG_SCALE=4.5 CACHED_BLOCKS=2`，单卡、day分区，预留多case需要的时限。另用相同 `TRAINING_RUN`、`FIXED_CASES`、`SEED`，`COMMAND=evaluate STEPS=4 CFG_SCALE=1`，输出到独立目录作sampler匹配对照。

阶段评估 CLI 与核心都接受 `1+3n >=61`，121帧 long 先生成固定案例，short/align 共用它的噪声前缀。已有61帧的 fixed-cases bundle 不能用于121帧；不要覆盖旧文件或静默重选案例。当前新链首次25step评估之前尚未创建案例时，由long创建121帧版本；其后的各次评估沿用同一文件。

完整工程门槛通过后才运行下面的fresh长程候选；不要与旧的自动launch500链一起提交。`segment_steps=1`先避免新的慢step跨Slurm时限，测量后可按预算增加；每段仍会保存，最新两套完整bundle自动保留。

```bash
cd "$PROJECT_ROOT"
"$PYTHON" -m tools.ttn_slurm_chain fresh \
  --world-size 4 --warmup-steps 0 --target-step 500 --segment-steps 1 \
  --partition day --time-limit 06:00:00 --memory 256G \
  --config "$PROJECT_ROOT/configs/worldttn/proximal_memory.json" \
  --dataset-root "$DATASET_ROOT" --tbptt 4 --seed 3407 --backbone-lr 1e-6 \
  --eval-every 25 --eval-frames 121 --eval-cases 1 --eval-steps 20 --eval-cfg-scale 4.5 \
  --keep-model-steps 25 50 100 250 \
  --output "$PROJECT_ROOT/output/proximal-C-$(date -u +%Y%m%dT%H%M%SZ)"
```

此处eval_cases=1用于与原固定case趋势衔接，不足以做泛化结论。正式科研结论需再加独立scene/seed并核对训练集重叠，报告MSE、视觉collapse位置、运动/回访、GPU-hours和latent exposure。

长期保留模型25/50/100/250/最终500；每25step评估日志保留。非关键评估模型仅在评估成功且汇总一致后释放，被固定引用或失败的快照保留。代码不清理旧实验。磁盘要求以保存器当次打印的 `required_free_bytes` 为准，还要给两套bundle和长期关键点留空间；内存缓存不是持久磁盘文件。

## 6. 通信端口占用与原目录恢复

`423688` 在 `init_process_group` 之前遇到 `MASTER_PORT=38688 / EADDRINUSE`，没有开始 step9。训练启动器默认不再从 job ID 推算端口：rank0 的 TCPStore 直接绑定系统分配的端口并持有 socket，再原子发布到本次 allocation 唯一的共享文件。其他 rank 校验 job/address/world 后连接；不存在探测空端口后关闭再绑定的竞态。batch 的 `[TTN Slurm]` 会先显示 `MASTER_PORT=0`，实际端口见 `[TTN init]` 和 `distributed.json`。

每段使用新的 `.rendezvous-JOB.XXXXXX/endpoint.json`，包括 requeue。这个小文件只记录通信地址，不是模型状态；恢复身份、数学和随机种子不变。显式 `MASTER_PORT` 仍走原 `env://`，占用时直接失败，不抢占其他进程。登录终端提交前 `unset MASTER_ADDR MASTER_PORT TTN_RENDEZVOUS_FILE`，让默认路径生效。

更新代码前确认该目录的训练/评估均停止。只用 `git diff --quiet` 与 `git diff --cached --quiet` 检查 tracked 改动；不要因为 untracked Slurm 日志而删除文件。显式 fetch 分支，再 `git merge --ff-only`，不 reset 工作目录。

先用既有 `tools/ttn_slurm_train.sbatch` 提交四卡 `COMMAND=distributed-check PARALLEL=fsdp2`（short，10分钟，每节点8G）；它不载入训练模型。成功时 `distributed.json` 必须记录四个不同节点、backend=nccl、all_reduce_sum=10。

恢复复用原 `FORMAL_RUN/chain.json`，不要另建正式训练目录：

```python
from pathlib import Path
import json, os
from tools.ttn_slurm_chain import require_checkpoint, submit
run = Path(os.environ["FORMAL_RUN"]).resolve()
manifest = run / "chain.json"
plan = json.loads(manifest.read_text())
assert Path(plan["output"]).resolve() == run and plan["world_size"] == 4
assert plan["target_step"] == 500
require_checkpoint(run, 8, 4)  # 完整 SHA/model/分片及训练记录审计
job = submit(plan, manifest, 8, dependency=os.environ["PROBE_JOB"])
```

这段仅用于已确认最后保存为step8、该链没有活动作业的恢复；执行前从 `jobs.jsonl` 与 `squeue` 排除重复提交。`afterok` 让通信检查失败时训练不启动；成功后从step9恢复原模型、Adam、RNG和游标，再沿用500step自动链与评估设置。真实模型的 bitwise 连续/恢复对照仍需要独立测量，通信检查不能代替它。

当时可用59GiB，保存器最低预算约47.63GiB，余量很小。原目录继续保留最新两套bundle；25/50/100/250等长期模型点还会新增占用，不能据此保证500step磁盘够用。删除旧实验前另外核对依赖；此修复不删除权重或日志。
