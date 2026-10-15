"""Prepare, but do not launch, the gated four-step follow-up."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from omegaconf import OmegaConf
from worldttn.distillation.evaluate import phase4_gate, video_review_ready


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def prepare(eight_run, review_file):
    import torch
    eight_run, review_file = Path(eight_run).resolve(), Path(review_file).resolve()
    best = eight_run/'best-adapters.pt'
    fingerprint = sha256(best)
    review = json.loads(review_file.read_text())
    if review.get('source_sha256') != fingerprint or review.get('video_review') is not True:
        raise ValueError('30/60s videos must be reviewed for this exact best-8 adapter')
    if not video_review_ready(eight_run, fingerprint):
        raise ValueError('Complete 30/60s teacher/student/TTT-off videos are required')
    payload = torch.load(best, map_location='cpu', weights_only=True)
    step = payload['counts']['student']
    metrics = json.loads((eight_run/f'eval-{step:06d}'/'metrics.json').read_text())
    if not phase4_gate(metrics['results'], True):
        raise ValueError('Best-8 student did not pass both quality gates and speed gate')
    launch=eight_run/'launch.json'
    if not launch.is_file() and eight_run.name=='training': launch=eight_run.parent/'launch.json'
    if launch.exists():
        from worldttn.distillation.checkpoint import digest
        identity=json.loads(launch.read_text())
        config_path=Path(identity['config'])
        if digest(config_path)!=identity['config_sha256']: raise ValueError('Eight-step config changed')
    else:
        config_path=ROOT/'configs/distillation/longlive_ttn_step200_8step.yaml'
    config = OmegaConf.load(config_path)
    config.student_steps = 4
    config.stage4_source = str(best)
    config.stage4_review = str(review_file)
    if 'capacity_proof' in config: del config['capacity_proof']
    output = ROOT/'output'/'stage4-configs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output.mkdir(parents=True, exist_ok=False)
    path = output/'stage4.yaml'
    OmegaConf.save(config, path)
    (output/'source.json').write_text(json.dumps({'source':str(best),'sha256':fingerprint,
        'best_student_step':step,'review':str(review_file),'metrics':metrics['score']},indent=2))
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eight-run', required=True, type=Path)
    parser.add_argument('--review-file', required=True, type=Path)
    args = parser.parse_args()
    print(prepare(args.eight_run, args.review_file))
