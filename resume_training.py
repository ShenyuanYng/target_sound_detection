"""Continue one existing experiment, using its saved settings and training state."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path

import torch

from train import main as train_main
from utils.project_paths import training_path
from utils.training_config import validate_training_data_config
from utils.tse_runtime import OVERRIDE_FILENAME, read_runtime_overrides, validate_runtime_overrides


def resume_settings(checkpoint, run_dir):
    """Keep the saved configuration, including the original data split and LR."""
    if not (training_path(run_dir) / ".tse_run_created").is_file():
        raise ValueError("Only standalone TSE runs may resume")
    hp = argparse.Namespace(**dict(checkpoint["hyper_parameters"]))
    overrides = read_runtime_overrides(run_dir)
    if overrides and hp.dataset != 'audioset_strong_dual_fsd_wrong_tse':
        raise ValueError('Runtime overrides apply only to experiment 3')
    for name, value in overrides.items():
        setattr(hp, name, value)
    validate_training_data_config(hp)
    run_dir = training_path(run_dir)
    hp.save_path = str(run_dir.parent)
    hp.save_name = run_dir.name
    hp.test_after_fit = False
    callbacks = [state for state in checkpoint.get("callbacks", {}).values()
                 if isinstance(state, dict) and "best_model_path" in state and "dirpath" in state]
    if len(callbacks) != 1:
        raise ValueError("Expected one saved ModelCheckpoint callback")
    # Reuse the old callback directory so Lightning restores best-k bookkeeping
    # instead of resetting it when TensorBoard creates a new per-job version.
    checkpoint_dir = Path(callbacks[0]["dirpath"])
    if not checkpoint_dir.is_dir() or checkpoint_dir.resolve().parent != run_dir:
        raise ValueError("Original checkpoint directory is missing or outside this run; do not move the experiment")
    return hp, str(checkpoint_dir)


def configure_runtime(run_dir, updates):
    """Persist only explicit overrides for the next process; never edit a checkpoint."""
    run_dir = training_path(run_dir).resolve(strict=True)
    validate_runtime_overrides(updates)
    if not updates:
        raise ValueError('Supply at least one execution override')
    with (run_dir / '.resume.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checkpoint = torch.load(str(run_dir / 'mylast.ckpt'), map_location='cpu', weights_only=False)
        if checkpoint['hyper_parameters']['dataset'] != 'audioset_strong_dual_fsd_wrong_tse':
            raise ValueError('Runtime overrides apply only to experiment 3')
        settings = {**read_runtime_overrides(run_dir), **updates}
        hp = argparse.Namespace(**{**checkpoint['hyper_parameters'], **settings})
        validate_training_data_config(hp)
        path = run_dir / OVERRIDE_FILENAME
        history = json.loads(path.read_text()).get('history', []) if path.exists() else []
        history.append(dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            changes=updates, checkpoint_epoch=checkpoint['epoch'], global_step=checkpoint['global_step']))
        record = dict(schema=1, settings=settings, history=history)
        temporary = run_dir / f'.tse_runtime_overrides.{os.getpid()}.tmp'
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        os.replace(temporary, path)
        print(f'Next resume settings saved to {path}: {settings}. Running processes are unchanged.', flush=True)


def completed_epochs(checkpoint):
    # completed may lag epoch end while a ModelCheckpoint hook is saving.
    # 'processed' marks a completed training+validation epoch before that hook.
    progress = checkpoint["loops"]["fit_loop"]["epoch_progress"]["total"]
    return max(int(progress["completed"]), int(progress["processed"]))


def continue_run(run_dir):
    run_dir = training_path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    # Protect against accidentally submitting two independent continuation chains.
    with (run_dir / ".resume.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checkpoint = torch.load(str(run_dir / "mylast.ckpt"), map_location="cpu", weights_only=False)
        hp, checkpoint_dir = resume_settings(checkpoint, run_dir)
        done = completed_epochs(checkpoint)
        print(f"Restoring {run_dir}: completed_epochs={done}, target_epochs={hp.epochs}, "
              f"global_step={checkpoint['global_step']}", flush=True)
        if done < hp.epochs:
            del checkpoint
            train_main(hp, resume_run_dir=str(run_dir), checkpoint_dir=checkpoint_dir)
            checkpoint = torch.load(str(run_dir / "mylast.ckpt"), map_location="cpu", weights_only=False)
            done = completed_epochs(checkpoint)
            if done < hp.epochs:
                raise RuntimeError(f"Training exited before target: {done}/{hp.epochs}")
        temporary = run_dir / f".training_complete.{os.getpid()}.tmp"
        temporary.write_text(json.dumps({"completed_epochs": done, "target_epochs": hp.epochs,
                                         "job_id": os.environ.get("SLURM_JOB_ID")}, indent=2))
        os.replace(temporary, run_dir / "training_complete.json")
        print(f"Training complete: {done}/{hp.epochs}. No additional epochs needed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument('--gradient-checkpointing', choices=('on', 'off'))
    parser.add_argument('--condition-mode', choices=('all', 'sampled_correct_wrong', 'sampled_one'))
    parser.add_argument('--condition-backward', choices=('combined', 'sequential'))
    parser.add_argument('--configure-only', action='store_true',
                        help='Save execution overrides for the next continuation; do not start training')
    args = parser.parse_args()
    updates = {}
    if args.gradient_checkpointing is not None:
        updates['tse_gradient_checkpointing'] = args.gradient_checkpointing == 'on'
    if args.condition_mode is not None:
        updates['tse_train_condition_mode'] = args.condition_mode
    if args.condition_backward is not None:
        updates['tse_condition_backward'] = args.condition_backward
    if args.configure_only:
        configure_runtime(args.run_dir, updates)
    elif updates:
        parser.error('Use --configure-only to persist execution overrides before resuming')
    else:
        continue_run(args.run_dir)
