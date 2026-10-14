# Monitor an already-running TTN experiment

`tools/ttn_wandb_monitor.py` reads `train.jsonl` without importing PyTorch,
loading weights, allocating GPU tensors, or modifying the training process.
It backfills completed steps and follows appended records when `--follow` is
given. Each step is uploaded in one scalar log call. Stopping or losing the
monitor does not stop training; local JSONL remains the source of truth.

## Start in a separate tmux pane

The following selects the most recently initialized valid GPU5 run. Check the
printed directory if you have deliberately started more than one experiment.
Only tool/docs files changed in this update; the training implementation is
unchanged. The monitoring SDK lives in its own virtual environment so installing
it does not change the active training environment's dependencies.

The explicit Bash invocation also works when your tmux shell is zsh or ksh;
their `read -p` option has different semantics. Prompts read from `/dev/tty`
because standard input below contains the script.

```bash
bash <<'BASH'
set -euo pipefail
ROOT=/mnt/einsia/aws01-nvme/einsia-shared/homes/gaomingju/workspace/yuhuang
PY_BIN="$ROOT/envs/worldttn-fp8/bin/python"
cd "$ROOT/WorldTTN-dual-psi-fp8"
git fetch origin refs/heads/codex/ttn-dual-psi-fp8:refs/remotes/origin/codex/ttn-dual-psi-fp8
git merge --ff-only origin/codex/ttn-dual-psi-fp8

RUN="$("$PY_BIN" - <<'PY'
from pathlib import Path
files = list(Path('output').glob('state-ttt-h100-gpu5-*/run_config.json'))
if not files:
    raise SystemExit('No initialized GPU5 run found; provide its full RUN path.')
print(max(files, key=lambda p: p.stat().st_mtime).parent.resolve())
PY
)"
test -n "$RUN"
printf 'Training directory: %s\n' "$RUN"
WB_ENV="$ROOT/envs/worldttn-wandb"
test -x "$WB_ENV/bin/python" || "$PY_BIN" -m venv "$WB_ENV"
"$WB_ENV/bin/python" -m pip install 'wandb==0.30.0'

  read -r -p 'W&B entity (the identifier in your profile URL): ' WB_ENTITY < /dev/tty
  read -r -s -p 'W&B API key (hidden): ' WANDB_API_KEY < /dev/tty
  printf '\n'
  export WANDB_API_KEY
  export CUDA_VISIBLE_DEVICES=''
  exec "$WB_ENV/bin/python" -u tools/ttn_wandb_monitor.py \
    "$RUN" --entity "$WB_ENTITY" --project WorldTTN --follow
BASH
```

Do not paste your key into a chat, command argument, Git file, `.bashrc`, or a
global tmux environment. The prompt places it only in this child process's
environment; there is no `wandb.login()` call. SDK config/cache/data directories
are isolated under `RUN/wandb-monitor`. Inherited `WANDB_*` sweep/service/settings
variables are cleared inside the monitor process except for your key, and the
destination is explicitly the public W&B API. Ordinary `wandb login` can update shared
login credentials, affecting future processes under the same Linux account.
This prevents accidental credential/config replacement; it is not a security
boundary against another person using the same Linux UID. Use separate Unix
accounts for actual credential isolation.

The website URL is printed and saved to `RUN/wandb-monitor/run-url.txt`.
Restart the same command with the same entity/project to resume its W&B run ID.
Identity is published only after SDK initialization succeeds. Old identity files
from failed initialization can be corrected when there is no successful-online
URL, initialized marker, or offline run. The previous uninitialized identity is
archived after the corrected target initializes; successful runs cannot be
silently redirected to another entity/project. Preserve every character in the
entity identifier, including a trailing hyphen if your workspace name has one.
Only one monitor can hold the directory lock. W&B's restored next history step
controls replay, rather than a local cursor that might precede network upload.
`Ctrl+C` finishes this monitoring session and leaves training running.

Online monitoring requires network access and an explicit process-local API key.
For a one-time offline export omit `--follow` and add `--mode offline`. Offline
SDK sessions do not support online resume; do not treat repeating an offline
export as exactly-once synchronization. A monitor failure never changes model
training or checkpoint retention.

## Metrics

- `train/*`: completed optimizer step, loss, outer gradient norm, timing, batch,
  world size and TBPTT.
- `memory/rankN/*`: recorded training GPU allocated/reserved peaks and maximum
  **sampled** host RSS. SDK system metrics are disabled, so charts do not mistake
  the CPU-only monitor's usage or another job's GPU for our training usage.
- `anchor19/rank0/all/*` and `.../tail5_chunks/*` (also anchors3/7/11/15): state
  RMS/norm, innovation, correction, beta and Local/clean State-TTT diagnostics,
  including curvature, eta, coverage, loss changes and measured output effects.
  Spectral stable rank/top1 values appear only when the source log contains them;
  they are not computed or fabricated from RMS.
- `training_prediction/*`: training velocity MSE and one-step x0/DC/detail/adjacent
  diagnostics. These are not PSNR/SSIM or generated-video quality scores.
- `anchor_gradients/*`, `parameter_update/*`, `health/*`: existing gradient/update
  diagnostics; first-update-only measurements remain first-update-only.
- `monitor/skipped_nonfinite_values`: nonfinite encountered scalars are excluded
  rather than reported as zero. The original JSONL retains complete evidence.

Mean/max summaries are descriptive across recorded chunks/heads **within each
rank**. Prefill is excluded; tail5 refers to the last five predicted chunks.
Different heads/chunks are not independent statistical samples. Charts update
after a full optimizer step is appended, not after each kernel or latent frame.
Full arrays stay in local JSONL. Checkpoints, videos, raw data, source code and
environment variable contents are not explicitly uploaded by this tool.

## Read-only check for existing W&B use on the shared machine

```bash
ps -eo user,pid,comm | awk 'NR == 1 || tolower($3) ~ /wandb/'
"$ROOT/envs/worldttn-fp8/bin/python" - <<'PY'
import netrc
from pathlib import Path
path = Path.home() / '.netrc'
if not path.exists():
    print('No shared ~/.netrc file.')
else:
    try:
        saved = any('wandb' in host.lower() for host in netrc.netrc(str(path)).hosts)
        print('Shared ~/.netrc contains W&B credentials:', saved)
    except (OSError, netrc.NetrcParseError):
        print('Shared ~/.netrc exists but could not be checked safely.')
PY
```

This prints process owner/PID/name and a boolean, never command arguments or API
keys. A service process is evidence of active/recent SDK use. Saved credentials
alone do not prove active use; no process match cannot prove nobody uses W&B
(offline runs, custom backend names and other credentials are possible).

SDK references: [authentication](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_login.py),
[environment variables](https://docs.wandb.ai/guides/track/environment-variables/),
[run resume](https://docs.wandb.ai/guides/runs/resuming/).
