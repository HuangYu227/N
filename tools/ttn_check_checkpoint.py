"""Read-only full-bundle validation; explicit --prune applies new-run retention."""
import argparse
import json
from pathlib import Path
import uuid
from worldttn.checkpoint_integrity import (acquire_checkpoint_lock, release_checkpoint_lock,
                                          audit_checkpoint, prune_bundles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--prune", action="store_true", help="apply retention only to this tool's managed bundles")
    args = parser.parse_args()
    lock = Path(args.checkpoint).parent / ".checkpoint-save.lock"
    token = "manual-retention-" + uuid.uuid4().hex
    if args.prune: acquire_checkpoint_lock(lock, token)
    try:
        print(json.dumps(audit_checkpoint(args.checkpoint, args.expected_step), indent=2))
        print(json.dumps(prune_bundles(args.checkpoint, apply=args.prune), indent=2))
    finally:
        if args.prune:
            try: release_checkpoint_lock(lock, token)
            except OSError as error: print(f"[TTN retention] could not release save lock: {error}", flush=True)


if __name__ == "__main__": main()
