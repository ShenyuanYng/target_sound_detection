"""Resume the audited 16-mixture fitting check directly on two CPU threads."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time


class EvaluationInterrupted(Exception):
    """The checkpoint is safe, but the current audio export is incomplete."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direct-cpu', action='store_true', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true', required=True)
    parser.add_argument('--threads', type=int, choices=(1, 2), default=2)
    parser.add_argument('--steps', type=int, default=384)
    parser.add_argument('--max-seconds', type=int, default=46800)
    parser.add_argument('--exp-def')
    args = parser.parse_args()
    if not 1 <= args.steps <= 384 or not 1 <= args.max_seconds <= 46800:
        parser.error('Bounds must be 1..384 steps and 1..46800 seconds')
    root = Path(args.output).resolve()
    allowed = Path('/home/syy12632604/projects/TSE/experiments/diagnostics').resolve()
    if allowed not in root.parents or not root.is_dir():
        parser.error('Use an existing audited directory inside TSE/experiments/diagnostics')

    # Apply limits before importing torch or any numerical library.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(args.threads)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    os.environ['XDG_CACHE_HOME'] = str(root/'runtime_cache')
    os.environ['TORCH_HOME'] = str(root/'torch_cache')
    current_nice = os.nice(0)
    if current_nice < 10:
        os.nice(10-current_nice)

    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    lock = (root/'.smallfit_direct_cpu.lock').open('a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another direct CPU writer holds this output directory')

    started = time.monotonic()
    run_id = f'{int(time.time())}-{os.getpid()}'
    state = dict(status='starting', stage='loading', pid=os.getpid(),
        host=socket.gethostname(), run_id=run_id, device='cpu',
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), direct_cpu=True,
        threads=args.threads, nice=os.nice(0), step=None, checkpoint_step=None,
        latest_complete_evaluation_step=None, prediction_export_complete=False,
        max_steps=args.steps, max_seconds=args.max_seconds, stop_signal=None)
    lock.seek(0)
    lock.truncate()
    lock.write(json.dumps(state)+'\n')
    lock.flush()

    def status(**changes):
        state.update(changes, updated_unix=time.time(), elapsed_seconds=time.monotonic()-started)
        temporary = root/'direct_cpu_status.json.tmp'
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
        temporary.replace(root/'direct_cpu_status.json')

    def request_stop(signum, _frame):
        state['stop_signal'] = signal.Signals(signum).name
        if state['stage'] == 'evaluating':
            raise EvaluationInterrupted()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    status()
    try:
        import random
        import numpy as np
        import torch
        from tools.run_tse_smallfit import (load_samples, predict, evaluate,
            create_tse_model, TSERILoss, yaml_to_parser, validate_tse_config,
            file_sha256, emit)

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        hp = yaml_to_parser(args.exp_def or str(
            project/'exp_defs/tse_strong_dual_fsd_hierarchy_v2.yaml')).parse_args([])
        validate_tse_config(hp)
        gt = json.loads((root/'groundtruth_checks.json').read_text())
        checks = json.loads((root/'loss_metric_checks.json').read_text())
        if gt.get('status') != 'passed' or checks.get('loss', {}).get('status') != 'passed':
            raise ValueError('Ground truth and loss audits must have passed')
        if checks.get('metrics', {}).get('reference_checks') != 'passed':
            raise ValueError('Independent metric checks must have passed')
        manifest, samples = load_samples(root)
        if len(samples) != 16 or manifest['source_config'] != vars(hp):
            raise ValueError('Expected exactly 16 audited samples with unchanged configuration')
        core_hashes = manifest['code_sha256']
        for filename, expected in core_hashes.items():
            path = (project/filename).resolve()
            if project not in path.parents or file_sha256(path) != expected:
                raise ValueError(f'Audited source changed: {filename}')
        metadata_hash = file_sha256(root/'prepared_manifest.json')
        runner_hashes = {name: file_sha256(project/name) for name in (
            'tools/run_tse_smallfit.py', 'tools/tse_smallfit_checks.py',
            'tools/run_tse_smallfit_direct_cpu.py')}
        checkpoint = root/'smallfit_last.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['source_config'] != vars(hp):
            raise ValueError('Checkpoint configuration differs')
        if saved.get('prepared_manifest_sha256', metadata_hash) != metadata_hash:
            raise ValueError('Checkpoint and prepared sample metadata differ')
        if saved.get('audited_code_sha256', core_hashes) != core_hashes:
            raise ValueError('Checkpoint and audited source fingerprints differ')
        seed = int(manifest['seed'])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device('cpu')
        model = create_tse_model(hp).to(device)
        loss_fn = TSERILoss(hp).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        step = int(saved['step'])
        if not 0 <= step <= args.steps:
            raise ValueError('Checkpoint step is outside requested cumulative bounds')
        order = [tuple(item) for item in saved['order']]
        expected_order = [(i, name) for i in range(16) for name in ('a', 'b', 'wrong')]
        if sorted(order) != sorted(expected_order):
            raise ValueError('Checkpoint query order does not match the 48 fixed queries')
        rng = random.Random()
        rng.setstate(saved['rng_state'])
        for name, restore in (('torch_rng_state', torch.set_rng_state),
                              ('numpy_rng_state', np.random.set_state),
                              ('python_rng_state', random.setstate)):
            if name in saved:
                restore(saved[name])
        del saved
        previous = root/'latest_evaluation.json'
        if previous.exists():
            previous_report = json.loads(previous.read_text())
            if previous_report.get('evaluated_mixtures') == 16:
                state['latest_complete_evaluation_step'] = previous_report['step']
                state['prediction_export_complete'] = previous_report['step'] == step
        status(status='running', stage='training', step=step, checkpoint_step=step,
            prepared_manifest_sha256=metadata_hash, audited_code_sha256=core_hashes,
            runner_sha256=runner_hashes)
        emit('direct_cpu_started', **state)

        def save_checkpoint():
            temporary = root/'smallfit_last.direct_cpu.tmp'
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                step=step, source_config=vars(hp), rng_state=rng.getstate(), order=order,
                torch_rng_state=torch.get_rng_state(), numpy_rng_state=np.random.get_state(),
                python_rng_state=random.getstate(), prepared_manifest_sha256=metadata_hash,
                audited_code_sha256=core_hashes, runner_sha256=runner_hashes,
                direct_cpu_run_id=run_id), temporary)
            temporary.replace(checkpoint)
            status(checkpoint_step=step)

        def run_evaluation():
            status(stage='evaluating', prediction_export_complete=False)
            report = evaluate(model, loss_fn, samples, root, device, step)
            # Save every completed evaluation, since the inherited helper replaces latest.
            summary = {key: value for key, value in report.items() if key != 'cases'}
            summary.update(run_id=run_id, unix_time=time.time())
            with (root/'evaluation_history.jsonl').open('a') as handle:
                handle.write(json.dumps(summary, allow_nan=False)+'\n')
            status(stage='training', latest_complete_evaluation_step=step,
                prediction_export_complete=True)

        durations = []
        reason = 'step_limit'
        while step < args.steps:
            if state['stop_signal'] or time.monotonic()-started >= args.max_seconds:
                reason = 'signal' if state['stop_signal'] else 'time_limit'
                save_checkpoint()
                break
            next_step = step+1
            if (next_step-1) % len(order) == 0:
                rng.shuffle(order)
            index, name = order[(next_step-1) % len(order)]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            tick = time.monotonic()
            prediction, losses = predict(model, loss_fn, samples[index], name, device)
            loss = losses['loss']
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite small-fit loss')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(),
                hp.gradient_clip_val, error_if_nonfinite=True)
            optimizer.step()
            step = next_step
            duration = time.monotonic()-tick
            durations.append(duration)
            record = dict(run_id=run_id, direct_cpu=True, step=step,
                sample=samples[index]['id'], condition=name, loss=float(loss.detach()),
                gradient_norm=float(gradient), seconds=duration,
                elapsed=time.monotonic()-started)
            del prediction, losses, loss
            with (root/'fit_history.jsonl').open('a') as handle:
                handle.write(json.dumps(record, allow_nan=False)+'\n')
            status(step=step, last_training_loss=record['loss'])
            emit('fit_step', **record)
            stop_signal = bool(state['stop_signal'])
            stop_time = time.monotonic()-started >= args.max_seconds
            final_step = step == args.steps
            evaluate_now = step % 96 == 0 or final_step
            if step % 8 == 0 or evaluate_now or stop_signal or stop_time:
                save_checkpoint()
            if state['stop_signal'] or stop_time:
                reason = 'signal' if state['stop_signal'] else 'time_limit'
                break
            if evaluate_now:
                run_evaluation()
        if (step == args.steps and not state['stop_signal']
                and time.monotonic()-started < args.max_seconds):
            if state['latest_complete_evaluation_step'] != step:
                run_evaluation()
            reason = 'step_limit'
        complete = step == args.steps and state['latest_complete_evaluation_step'] == step
        status(status='completed' if complete else 'stopped', stage='finished',
            stop_reason=reason, step=step,
            seconds_per_step=sum(durations)/len(durations) if durations else None,
            note='A completed bounded run is not evidence of successful separation')
        emit('direct_cpu_finished', **state)
    except EvaluationInterrupted:
        status(status='stopped', stage='finished', stop_reason='signal_during_evaluation',
            prediction_export_complete=False,
            note='Checkpoint saved before evaluation; some pred WAV files may be from this incomplete export')
    except BaseException as exc:
        status(status='failed', stage='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
