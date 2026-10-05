# LTU 双时间尺度 Meta-TTT 验收

按顺序执行；上一关检查通过才进入下一关。本批不提交 500-step 正式训练。
97 raw/13 latent、TBPTT2 是工程数据，不能验证 40-chunk 稳定性。
本地改动需要先同步到独立的 codex/ttn-dual-psi-meta checkout；下面不修改旧目录。

## 0. 固定环境与空间

```bash
export ROOT=/data/group/zhaolab/home/z2zhang/huangyu
export PROJECT_ROOT="$ROOT/WorldTTN-dual-psi-meta"
export PYTHON="$ROOT/envs/worldttn/bin/python"
cd "$PROJECT_ROOT"
git branch --show-current
"$PYTHON" -c 'import sys,torch,shutil; print(sys.version); print(torch.__version__); print("free GiB:",shutil.disk_usage(".").free/2**30)'
```

要求现有 Python3.11/PyTorch2.9.1+cu128。新脚本在 GPU worker 内检查版本、
CUDA 与单张可见卡；跳过 CUDA 测试不能算 GPU gate 通过。
以下完整验收预计保留四套独立生产 checkpoint，建议启动前有至少 200 GiB。
真正的保存预算由已有 preflight 计算：预计新增临时峰值×1.15+2 GiB。
永久数据/结果/权重仍在共享 NFS，worker 编译缓存在节点 /tmp；不装包或改 Conda。

```bash
unset ADAPTER RESUME UNFREEZE BATCH_FILE CAMERA_ATTENTION CAMERA_ABLATION
unset TTN_ABLATION HISTORY_SOURCE EVAL_METHODS STATE_DIAGNOSTICS META_TEST_OUTPUT
unset MASTER_ADDR MASTER_PORT RANK LOCAL_RANK WORLD_SIZE NODE_RANK LOCAL_WORLD_SIZE
export CONFIG="$PROJECT_ROOT/configs/worldttn/dual_psi_meta.json"
export DATASET_ROOT="${DATASET_ROOT:-$ROOT/datasets/sana-wm-example}"
export COMMAND=train
export STAGE=C TRAIN_SCOPE=dit OPTIMIZER_POLICY=origin BACKBONE_LR=1e-6
export PARALLEL=fsdp2 TBPTT=2 SANA_CP_SIZE=1
export TTN_CORE_BACKEND=reference TTN_PSI_BACKEND=reference
export ACTIVATION_OFFLOAD=cpu CROSS_ATTN_BACKEND=math
export GDN_DISABLE_COMPILE=1 GDN_DISABLE_COMPLEX_COMPILE=0 CUDA_LAUNCH_BLOCKING=0
```

若服务器已有 DATA_DIR/VAE_CACHE_DIR/BASE_WEIGHTS 的有效覆盖，保留原值；
DATASET_ROOT 必须指向原训练数据，而非此文示例创建的新数据目录。
新配置不改变 SANA data.num_frames=97。

## 1. CPU、单卡 CUDA oracle 与 native camera

```bash
"$PYTHON" -m pytest tests/ttn/test_meta_core.py tests/ttn/test_meta_runtime.py \
  tests/ttn/test_meta_training.py tests/ttn/test_meta_sampler.py tests/ttn/test_compare_resume.py -q

OP_JOB=$(sbatch --parsable --export=ALL --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  tools/ttn_slurm_meta_tests.sbatch); OP_JOB=${OP_JOB%%;*}
echo "OP_JOB=$OP_JOB"
sacct -X -j "$OP_JOB" --format=JobID,State,ExitCode,Elapsed
tail -n 30 "slurm-ttn-meta-tests-${OP_JOB}.out"
```

单卡任务执行 FP64 analytical/meta 与 autograd oracle、gradcheck/gradgradcheck，
以及实际 SANA camera helper 的输出/缓存/梯度、真实 sampler loop 的噪声记录/commit 检查。CPU 日志里的 CUDA skips
正常；GPU job 中 CUDA 两组 oracle 与四组 camera 用例必须真的执行。

## 2. 四卡 FSDP2 future credit、offload、exact resume

```bash
META_JOB=$(sbatch --parsable --export=ALL tools/ttn_slurm_meta_tests.sbatch)
META_JOB=${META_JOB%%;*}; echo "META_JOB=$META_JOB"
sacct -X -j "$META_JOB" --format=JobID,State,ExitCode,Elapsed
tail -n 40 "slurm-ttn-meta-tests-${META_JOB}.out"
```

每个 rank 应两项通过：gP 在窗口内接收 future credit、窗口末尾截断；
sharded 模型与同四例 unsharded batch 比较参数/梯度，误差阈值沿用原测试。
独立比较保存恢复后的参数、Adam、RNG、游标，使用零容差。
这是小型真实 TTN/FSDP 检查，下一步才加载完整预训练 DiT。

## 3. 完整 DiT，一个真实 optimizer step

```bash
unset OUTPUT ADAPTER RESUME UNFREEZE
CHECK_JOB=$(MAX_STEPS=1 SAVE_EVERY=1 sbatch --parsable --export=ALL --mem=256G \
  --time=01:00:00 tools/ttn_slurm_train.sbatch)
CHECK_JOB=${CHECK_JOB%%;*}; echo "CHECK_JOB=$CHECK_JOB"
export CHECK_RUN="$PROJECT_ROOT/output/worldttn/slurm-${CHECK_JOB}-C"
sacct -X -j "$CHECK_JOB" --format=JobID,State,ExitCode,Elapsed
tail -F "slurm-ttn-ddp-${CHECK_JOB}.out" |
  grep --line-buffered -E '^\[TTN (trainable|progress|checkpoint|retention)|^\[DiT update\]|Error:|oom-kill'
```

tail 只看日志，Ctrl+C 不取消训练。任务 COMPLETED 后执行：

```bash
"$PYTHON" - <<'PY'
import json,os
from pathlib import Path
from worldttn.checkpoint_integrity import audit_checkpoint
p=Path(os.environ['CHECK_RUN'])
r=json.loads((p/'run_config.json').read_text())
assert r['camera_attention']=='sana'
assert r['execution']['psi_implementation']=='live_projected_meta'
assert r['training']['meta_ttt']['local_update'] and r['training']['meta_ttt']['persistent_meta']
assert {g['name']:g['lr'] for g in r['parameters']['optimizer_groups']} == {'ttn_new':1e-5,'sana_inherited':1e-6}
h=json.loads((p/'first_update.json').read_text())
for x in h['ranks']:
    assert not x['missing_core_gradients']
    print('rank',x['rank'],'origins',x['by_origin'])
assert sum(x['groups']['ttn_system.local_eta_logits']['delta_norm']**2 for x in h['ranks'])>0
print(audit_checkpoint(p/'last.pt',1))
PY
```

raymap_embedder 在原配置未启用时不参与梯度，是已知条件分支；不能把
它与断开的核心 QKV/proj/gate/η 混为一谈。任一失败先查看 failure-*.json 和原异常。

## 4. 四步普通训练验收，检查计算图是否滞留

```bash
unset OUTPUT
STABILITY_JOB=$(RESUME=1 ADAPTER="$CHECK_RUN/last.pt" MAX_STEPS=5 SAVE_EVERY=5 \
  sbatch --parsable --export=ALL --mem=256G --time=01:00:00 tools/ttn_slurm_train.sbatch)
STABILITY_JOB=${STABILITY_JOB%%;*}; echo "STABILITY_JOB=$STABILITY_JOB"
export STABILITY_RUN="$PROJECT_ROOT/output/worldttn/slurm-${STABILITY_JOB}-C"
sacct -X -j "$STABILITY_JOB" --format=JobID,State,ExitCode,Elapsed
```

这是从 step1 续训到 step5 的普通 reference/reference 验收，不运行加速基准。
step2 是新 allocation 的首步，step3–5 用于检查稳定窗口。
只在结束时保存一套完整 checkpoint；不是每步保存。
检查 train.jsonl 和 Slurm memory 日志中四个样本的耗时、GPU peak 和 host RSS，
后三个样本不能持续上升而不解释。审计副本的 CPU/time 成本包含在时间内。
有 timestep/data 随机差异，有限窗口里峰值波动本身不等于图滞留。

```bash
"$PYTHON" - <<'PY'
import json,os
from pathlib import Path
p=Path(os.environ['STABILITY_RUN'])
rows=[json.loads(line) for line in (p/'train.jsonl').read_text().splitlines()]
assert [x['step'] for x in rows]==[2,3,4,5]
for x in rows:
    print('step',x['step'],'seconds',x['seconds'],
          [(y['rank'],round(y['peak_allocated_bytes']/2**30,3)) for y in x['ranks']])
from worldttn.checkpoint_integrity import audit_checkpoint
print(audit_checkpoint(p/'last.pt',5))
PY
```

## 5. 生产 DiT 退出/续训与未中断两步严格比较

继续使用相同 CONFIG/数据/seed/LR/world size/execution/offload，不切换 backend。
先跑新的两步连续任务；它和第3关的一步任务均从同一 SANA 初始化开始。

```bash
unset OUTPUT ADAPTER RESUME UNFREEZE
CONT_JOB=$(MAX_STEPS=2 SAVE_EVERY=2 sbatch --parsable --export=ALL --mem=256G \
  --time=01:00:00 tools/ttn_slurm_train.sbatch)
CONT_JOB=${CONT_JOB%%;*}
export CONT_RUN="$PROJECT_ROOT/output/worldttn/slurm-${CONT_JOB}-C"
```

连续任务 COMPLETED 后，从已核验的一步 CHECK_RUN 提交新 allocation 续训：

```bash
RESUME_JOB=$(RESUME=1 ADAPTER="$CHECK_RUN/last.pt" MAX_STEPS=2 SAVE_EVERY=2 \
  sbatch --parsable --export=ALL --mem=256G --time=01:00:00 tools/ttn_slurm_train.sbatch)
RESUME_JOB=${RESUME_JOB%%;*}
export RESUME_RUN="$PROJECT_ROOT/output/worldttn/slurm-${RESUME_JOB}-C"
sacct -X -j "$CONT_JOB,$RESUME_JOB" --format=JobID,State,ExitCode,Elapsed
```

两个任务都 COMPLETED 后比较；不放宽容差掩盖 CUDA 非确定性或游标问题：

```bash
"$PYTHON" -m tools.ttn_compare_resume "$CONT_RUN/last.pt" "$RESUME_RUN/last.pt"
```

应为 exact_match，step2、world4，比较模型/Adam/RNG/data cursor。
独立 checkpoint ID、时间戳自然不同，不参与逐 tensor 比较。

## 6. 固定 case rollout、teacher 与两分支贡献

使用完成的 step2。这里手动保留验收 snapshot；周期链则仅关键模型长期保留。

```bash
export TRAINING_RUN=$("$PYTHON" -m tools.ttn_eval_snapshot --training-run "$RESUME_RUN")
export FIXED_CASES="$PROJECT_ROOT/output/meta-fixed-cases.pt"
export OUTPUT="$PROJECT_ROOT/output/worldttn/meta-acceptance-evaluation-${RESUME_JOB}"
export ACCEPT_EVAL="$OUTPUT"
EVAL_JOB=$(COMMAND=stage-evaluate STEPS=20 SEED=3407 sbatch --parsable --export=ALL \
  --time=01:00:00 tools/ttn_slurm_eval.sbatch)
EVAL_JOB=${EVAL_JOB%%;*}; echo "EVAL_JOB=$EVAL_JOB"
sacct -X -j "$EVAL_JOB" --format=JobID,State,ExitCode,Elapsed
```

summary.json 要求 completed；long/short/align/no-local/no-persistent 五项完成。
每个 rollout 的 commits=预测chunk数+prefill，solver 多次 noisy call 不增加 commits。
查看 case-000-ttn.pt、各 state 文件与 summary.json 中的 episode/chunk 记录；S/ψ 谱、写入方向和两类 inner
gradient 独立分析。新 ψ 成功工作不意味着 step2 已经优于训练成熟的 SANA。

```bash
# 上一任务完成后，单独测 61-frame 六组机制；自动选择 0-5%1 顺序数组。
"$PYTHON" -m tools.ttn_submit_mechanism --evaluation "$ACCEPT_EVAL"
```

Full 需要先复现同 checkpoint/case/noise 的 source rollout，再读贡献对比。
机制关闭只描述当前使用价值；四种 Local/Persistent-meta 训练配置才检验训练价值。
没有 valid revisit pair 保留 null。通过所有关卡后进入 H1/H2，
随后 generated-history，再规划 121 latent 长程及 matched exposure/compute 控制。
