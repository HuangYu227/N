"""Advance the approved 3 -> 4 -> 500 schedule only after explicit acceptance."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def validate_records(rows, expected):
    from worldttn.inplace_acceptance import validate_records as validate
    return validate(rows, expected)


def wait_for_stage(output, stage):
    path = output / (stage+'-state.json')
    while True:
        if path.exists():
            state = json.loads(path.read_text())
            if state['status'] in ('completed', 'failed'):
                if state['exit_code'] != 0:
                    raise RuntimeError(stage+' failed; no further training submitted')
                return state
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    python = '/home/newuser001/miniconda3/envs/wm/bin/python'
    window = subprocess.check_output(['tmux', 'display-message', '-p', '-t', os.environ['TMUX_PANE'],
                                      '#{window_name}'], text=True).strip()
    sys.path.insert(0, str(root))
    from worldttn.checkpoint_integrity import audit_checkpoint
    stages = (('capacity', 3), ('resume-check', 4))
    for index, (stage, expected) in enumerate(stages):
        if stage == 'resume-check':
            command = [python, '-u', str(root/'tools/inplace_launch_stage.py'), '--stage', stage,
                       '--root', str(root), '--output', str(output), '--window', window]
            subprocess.run(command, check=True)
        wait_for_stage(output, stage)
        rows = [json.loads(line) for line in (output/'training/train.jsonl').read_text().splitlines() if line]
        report = validate_records(rows, expected)
        report['checkpoint'] = audit_checkpoint(output/'training/last.pt', expected_step=expected)
        (output/(stage+'-acceptance.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('[InPlace accepted] '+json.dumps(report), flush=True)
    print('[InPlace schedule] All acceptance gates passed; resuming to cumulative step500', flush=True)
    subprocess.run([python, '-u', str(root/'tools/inplace_launch_stage.py'), '--stage', 'train',
                    '--root', str(root), '--output', str(output), '--window', window], check=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        state_path = Path('/home/newuser001/huangyu/.worldttt-experiment/current.json')
        state = json.loads(state_path.read_text())
        # A completed stage can still fail the numerical gate; never label that as successful training.
        state.update(status='failed', exit_code=1, acceptance_error=str(error), pid=os.getpid())
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(state_path)
        raise
