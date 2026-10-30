"""Fit one audited mixture and its A condition on an explicitly chosen GPU."""
import argparse
import ast
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time


def executable_ast(path):
    """Ignore comments and standalone string statements, never executable syntax."""
    class DropStandaloneStrings(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return None
            return self.generic_visit(node)
    tree = DropStandaloneStrings().visit(ast.parse(path.read_text(), filename=str(path)))
    return ast.dump(tree, include_attributes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--threads', type=int, choices=(1, 2), default=2)
    parser.add_argument('--gpu-memory-gib', type=int, choices=range(4, 25), default=16)
    parser.add_argument('--max-seconds', type=int, default=14400)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument('--resume', action='store_true')
    resume.add_argument('--resume-from', help='Read-only absolute last.pt from the completed 1000-step run')
    parser.add_argument('--view-dir', help='Optional PROJECT/data/name containing only run.log and audio links')
    parser.add_argument('--reference-code-dir', help='Read-only audited source used to verify comment-only changes')
    args = parser.parse_args()
    if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', args.gpu_uuid):
        parser.error('Specify a complete GPU UUID, not an ordinal or automatic selection')
    if not 1 <= args.steps <= 2000 or not 1 <= args.max_seconds <= 14400:
        parser.error('Bounds must be 1..2000 steps and 1..14400 seconds')
    project = Path(__file__).resolve().parents[1]
    shared = Path('/home/syy12632604/projects/TSE').resolve()
    if project != shared:
        parser.error('Execute the shared project runner; separate code snapshots are unsupported')
    root = Path(args.output).resolve()
    if not root.is_dir() or (project/'experiments') not in root.parents:
        parser.error('Use an audited run under shared TSE/experiments')
    source_checkpoint = Path(args.resume_from) if args.resume_from else None
    if source_checkpoint is not None:
        if (not source_checkpoint.is_absolute() or source_checkpoint.name != 'last.pt'
                or not source_checkpoint.is_file() or args.steps != 2000):
            parser.error('--resume-from requires an absolute existing last.pt and --steps 2000')
        source_checkpoint = source_checkpoint.resolve()
        if source_checkpoint.parent == root or any((root/name).exists() for name in (
                'last.pt', 'history.jsonl', 'latest_evaluation.json', 'continuation_origin.json')):
            parser.error('--resume-from requires a fresh, separate run directory')
    elif not args.resume and any((root/name).exists() for name in ('last.pt', 'history.jsonl', 'latest_evaluation.json')):
        parser.error('Preserve existing runs; use --resume or choose a fresh directory')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    os.environ['XDG_CACHE_HOME'], os.environ['TORCH_HOME'] = str(root/'runtime_cache'), str(root/'torch_cache')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    priority = os.nice(0)
    if priority < 10:
        os.nice(10-priority)
    sys.path.insert(0, str(project))
    lock = (root/'.singlefit.lock').open('a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another singlefit writer holds this directory')
    started = time.monotonic()
    state = dict(status='starting', stage='loading', pid=os.getpid(), host=socket.gethostname(),
        gpu_uuid=args.gpu_uuid, slurm_job_id=os.environ.get('SLURM_JOB_ID'), step=0,
        checkpoint_step=None, evaluation_step=None, stop_signal=None,
        max_steps=args.steps, max_seconds=args.max_seconds, threads=args.threads,
        pytorch_memory_budget_gib=args.gpu_memory_gib)

    def write_json(path, value):
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)

    def status(**changes):
        state.update(changes, updated_unix=time.time(), elapsed_seconds=time.monotonic()-started)
        write_json(root/'status.json', state)

    def request_stop(signum, _frame):
        state['stop_signal'] = signal.Signals(signum).name

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        status()
        import random
        import numpy as np
        import soundfile as sf
        import torch
        from tools.run_tse_smallfit import (predict, create_tse_model, TSERILoss,
            validate_tse_config, file_sha256, label_name, emit)
        from tools.tse_smallfit_checks import measure
        from nets.dprnntf_original.audio_utils_torch import do_stft, do_istft

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('The explicitly selected GPU must be the only usable CUDA device')
        device = torch.device('cuda:0')
        total_memory = torch.cuda.get_device_properties(0).total_memory
        budget = args.gpu_memory_gib * 2**30
        if budget >= total_memory:
            raise ValueError('PyTorch memory budget must be smaller than total GPU memory')
        torch.cuda.set_per_process_memory_fraction(budget / total_memory, device=0)
        manifest_path = root/'singlefit_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('schema') != 1 or manifest.get('condition') != 'a' or manifest.get('seed') != 71:
            raise ValueError('Expected schema 1, fixed condition a, and seed 71')
        hp = argparse.Namespace(**manifest['source_config'])
        validate_tse_config(hp)
        manifest_hash = file_sha256(manifest_path)
        origin_path = root/'continuation_origin.json'
        origin = json.loads(origin_path.read_text()) if args.resume and origin_path.exists() else None
        if source_checkpoint is not None:
            if file_sha256(source_checkpoint.parent/'singlefit_manifest.json') != manifest_hash:
                raise ValueError('Copy the original manifest bytes without rewriting source_config')
            origin = dict(source_checkpoint=str(source_checkpoint),
                source_checkpoint_sha256=file_sha256(source_checkpoint), start_step=1000)
        input_source = Path(origin['source_checkpoint']).parent if origin else root
        hashes = manifest['code_sha256']
        if not hashes:
            raise ValueError('Missing audited source fingerprints')
        reference_value = args.reference_code_dir or (origin or {}).get('reference_code_dir')
        reference = Path(reference_value).resolve() if reference_value else None
        if reference_value and (not Path(reference_value).is_absolute() or not reference.is_dir()):
            raise ValueError('--reference-code-dir must be an existing absolute directory')
        if source_checkpoint is not None:
            origin['reference_code_dir'] = str(reference) if reference else None
        actual_hashes, verification = {}, {}
        for filename, expected in hashes.items():
            path = (project/filename).resolve()
            if project not in path.parents:
                raise ValueError(f'Audited source must remain inside the shared project: {filename}')
            actual = actual_hashes[filename] = file_sha256(path)
            check = dict(expected_sha256=expected, actual_sha256=actual, method='sha256_exact')
            if actual != expected:
                historical = (reference/filename).resolve() if reference else None
                if historical is None or reference not in historical.parents or file_sha256(historical) != expected:
                    raise ValueError(f'Audited reference SHA does not match: {filename}')
                if executable_ast(path) != executable_ast(historical):
                    raise ValueError(f'Executable source differs from the audited reference: {filename}')
                check.update(method='ast_equal_ignoring_standalone_strings',
                    reference_file=str(historical), reference_sha256=expected)
            verification[filename] = check
        item = manifest['sample']
        directory = (root/item['folder']).resolve()
        if root not in directory.parents or set(item['files']) != {'mix', 'background', 'a', 'b', 'wrong'}:
            raise ValueError('Expected a sample folder inside this run with the five fixed waveforms')
        arrays, audio_hashes = {}, {}
        for name, filename in item['files'].items():
            path = (directory/filename).resolve()
            original = (input_source/item['folder']/filename).resolve()
            if root not in path.parents and not (origin and path == original):
                raise ValueError('Input must be inside this run or link to the exact original sample file')
            wave, rate = sf.read(str(path), dtype='float32')
            if rate != 16000 or wave.shape != (160000,) or not np.isfinite(wave).all():
                raise ValueError(f'Expected finite mono 16 kHz, 10-second audio: {name}')
            arrays[name], audio_hashes[name] = wave, file_sha256(path)
        sum_error = float(np.max(np.abs(arrays['mix']-(arrays['background']+arrays['a']+arrays['b']))))
        if sum_error >= 2e-6 or np.std(arrays['a']) <= 1e-5 or np.any(arrays['wrong']):
            raise ValueError('Mixture/GT construction or nonzero A/zero Wrong checks failed')
        sample = {**item, **arrays}

        def sync_view():
            if not args.view_dir:
                return
            view = Path(args.view_dir).absolute()
            if view.parent != project/'data' or view.is_symlink():
                raise ValueError('--view-dir must be a real directory directly under shared PROJECT/data')
            view.mkdir(parents=True, exist_ok=True)
            if set(path.name for path in view.iterdir()) - {'run.log', 'audio'}:
                raise ValueError('View directory may contain only run.log and audio')
            log = view/'run.log'
            if log.is_symlink():
                if log.resolve() != (root/'run.log').resolve():
                    raise ValueError('Unexpected log link in view directory')
            elif log.exists():
                raise ValueError('View log must be a symlink, never a writable file')
            else:
                log.symlink_to(root/'run.log')
            audio = view/'audio'
            if audio.is_symlink():
                raise ValueError('View audio must be a real folder of read-only links')
            audio.mkdir(exist_ok=True)
            sources = {path.name: path for path in directory.iterdir() if path.suffix == '.wav'}
            for path in audio.iterdir():
                if path.name not in sources or not path.is_symlink() or path.resolve() != sources[path.name].resolve():
                    raise ValueError(f'Unexpected audio entry in view: {path.name}')
            for name, target in sources.items():
                link = audio/name
                if not link.is_symlink():
                    link.symlink_to(target)

        sync_view()
        seed = int(manifest['seed'])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model, loss_fn = create_tse_model(hp).to(device), TSERILoss(hp).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
        step, best_loss = 0, float('inf')
        checkpoint = root/'last.pt'
        if args.resume or source_checkpoint is not None:
            saved = torch.load(source_checkpoint or checkpoint, map_location=device, weights_only=False)
            if saved['manifest_sha256'] != manifest_hash or saved['audio_sha256'] != audio_hashes:
                raise ValueError('Checkpoint metadata or fixed audio changed')
            if saved['source_config'] != vars(hp) or saved['code_sha256'] != hashes:
                raise ValueError('Checkpoint configuration or audited source changed')
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            step, best_loss = int(saved['step']), float(saved['best_loss'])
            if source_checkpoint is not None:
                if step != 1000:
                    raise ValueError('External continuation must start from last.pt at exactly step 1000')
                best_loss = float('inf')  # The old best may belong to step 800, not this branch.
            elif saved.get('source_origin') != origin:
                raise ValueError('Continuation origin record and checkpoint differ')
            if hp.learning_rate != 0.0001 or any(group['lr'] != hp.learning_rate for group in optimizer.param_groups):
                raise ValueError('Continuation must retain Adam learning_rate=0.0001 without overriding it')
            torch.set_rng_state(saved['torch_rng'].cpu())
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
            random.setstate(saved['python_rng'])
            np.random.set_state(saved['numpy_rng'])
            del saved
        elif checkpoint.exists() or (root/'history.jsonl').exists():
            raise ValueError('Run already started; use --resume or a new directory')
        if not 0 <= step <= args.steps:
            raise ValueError('Checkpoint exceeds requested cumulative step bound')
        if source_checkpoint is not None:
            write_json(origin_path, origin)
        with torch.no_grad():
            mix = torch.from_numpy(arrays['mix'][None]).to(device)
            target = torch.from_numpy(arrays['a'][None]).to(device)
            ri = do_stft(target, hp.tse_window, hp.tse_hop)
            targets = dict(mixture_waveform=mix, target_waveform=target)
            perfect = float(loss_fn(dict(target_ri=ri), targets)['loss'])
            zero = float(loss_fn(dict(target_ri=torch.zeros_like(ri)), targets)['loss'])
            reconstructed = do_istft(ri, hp.tse_window, hp.tse_window, hp.tse_hop)[..., :160000]
            roundtrip = float((reconstructed-target).abs().max())
        if perfect != 0 or roundtrip >= 2e-6:
            raise ValueError('GT loss or STFT round-trip check failed')
        write_json(root/'numerical_checks.json', dict(status='passed', perfect_gt_loss=perfect,
            zero_output_loss=zero, stft_roundtrip_max_error=roundtrip, stem_sum_error=sum_error))
        del mix, target, ri, reconstructed
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=project, capture_output=True, text=True)
        status(status='running', stage='training', step=step, gpu_name=torch.cuda.get_device_name(0),
            manifest_sha256=manifest_hash, code_sha256=hashes, audio_sha256=audio_hashes,
            actual_code_sha256=actual_hashes, code_verification=verification,
            runner_sha256=file_sha256(Path(__file__)), zero_output_loss=zero,
            git_revision=revision.stdout.strip() if revision.returncode == 0 else None,
            source_origin=origin, effective_learning_rate=optimizer.param_groups[0]['lr'], output=str(root),
            parameter_count=sum(p.numel() for p in model.parameters()))
        write_json(root/'effective_config.json', dict(source_config=vars(hp), runtime=dict(
            output=str(root), start_step=step, max_steps=args.steps, max_seconds=args.max_seconds,
            gpu_uuid=args.gpu_uuid, gpu_memory_gib=args.gpu_memory_gib,
            threads=args.threads, learning_rate=optimizer.param_groups[0]['lr']),
            actual_code_sha256=actual_hashes, code_verification=verification))
        write_json(root/'provenance.json', dict(git_revision=state['git_revision'],
            source_origin=origin, reference_code_dir=str(reference) if reference else None,
            manifest_sha256=manifest_hash, code_sha256=hashes, actual_code_sha256=actual_hashes,
            code_verification=verification, runner_sha256=state['runner_sha256'], audio_sha256=audio_hashes))
        emit('singlefit_started', **state)

        def save_checkpoint(path=checkpoint):
            temporary = path.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                best_loss=best_loss, source_config=vars(hp), manifest_sha256=manifest_hash,
                code_sha256=hashes, actual_code_sha256=actual_hashes, code_verification=verification,
                audio_sha256=audio_hashes, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(), python_rng=random.getstate(),
                numpy_rng=np.random.get_state(), source_origin=origin), temporary)
            temporary.replace(path)
            if path == checkpoint:
                status(checkpoint_step=step)

        def evaluate():
            nonlocal best_loss
            status(stage='evaluating')
            model.eval()
            with torch.no_grad():
                prediction, losses = predict(model, loss_fn, sample, 'a', device)
                wave = prediction['target_waveform'][0].cpu().numpy()
                report = dict(step=step, loss=float(losses['loss']), zero_output_loss=zero,
                    sample=item['id'], condition='a', **measure(wave, arrays['a'], arrays['mix']))
            if not np.isfinite(wave).all() or not np.isfinite(report['loss']):
                raise FloatingPointError('Nonfinite evaluation output or loss')
            error = float(np.mean((wave.astype(np.float64)-arrays['a'])**2))
            report['sdr_db'] = float(10*np.log10(max(float(np.mean(arrays['a'].astype(np.float64)**2)), 1e-30)/max(error, 1e-30)))
            target64, output64 = arrays['a'].astype(np.float64), wave.astype(np.float64)
            projection = float(np.dot(output64, target64)/np.dot(target64, target64))
            event = item['metadata']['events'][0]
            start, end = int(event['target_start_sample']), int(event['target_end_sample'])
            if not 0 <= start < end <= 160000 or np.any(target64[:start]) or np.any(target64[end:]):
                raise ValueError('Recorded A interval does not match the zero-padded target')
            outside = np.ones(160000, dtype=bool)
            outside[start:end] = False
            outside_energy = float(np.dot(output64[outside], output64[outside]))
            report.update(gt_projection_gain=projection,
                outside_target_energy=outside_energy,
                outside_target_rms=float(np.sqrt(outside_energy/max(int(outside.sum()), 1))),
                outside_to_target_energy=outside_energy/float(np.dot(target64, target64)),
                target_interval_seconds=[start/16000, end/16000])
            filename = f'pred_A_step_{step:06d}_'+label_name(item['conditions']['a']['name'])+'.wav'
            prediction_path = directory/filename
            if prediction_path.exists() or prediction_path.is_symlink():
                old, rate = sf.read(str(prediction_path), dtype='float32')
                if rate != 16000 or not np.array_equal(old, wave):
                    raise ValueError('An existing step prediction differs; refusing to overwrite it')
            else:
                sf.write(str(prediction_path), wave, 16000, subtype='FLOAT')
            if step == 0:
                initial = directory/'initial_A.wav'
                if not initial.exists() and not initial.is_symlink():
                    sf.write(str(initial), wave, 16000, subtype='FLOAT')
            if report['loss'] < best_loss:
                best_loss = report['loss']
                save_checkpoint(root/'best.pt')
                write_json(root/'best_evaluation.json', report)
            write_json(root/'latest_evaluation.json', report)
            baseline_path = root/'baseline_evaluation.json'
            if not baseline_path.exists():
                write_json(baseline_path, report)
            if step == 2000 and origin:
                baseline = json.loads(baseline_path.read_text())
                if baseline['step'] != 1000:
                    raise ValueError('Continuation comparison requires a step-1000 baseline')
                keys = ('loss', 'si_sdr_db', 'si_sdri_db', 'sdr_db', 'gt_projection_gain',
                        'outside_target_energy', 'outside_target_rms', 'outside_to_target_energy')
                write_json(root/'comparison_report.json', dict(baseline_step=1000, final_step=2000,
                    baseline={key: baseline[key] for key in keys},
                    final={key: report[key] for key in keys},
                    final_minus_baseline={key: report[key]-baseline[key] for key in keys},
                    learning_rate=optimizer.param_groups[0]['lr'],
                    definitions=dict(gt_projection_gain='dot(output, GT) / dot(GT, GT), without centering',
                        outside_target_energy='sum(output**2) outside the recorded target interval',
                        sdr_db='10*log10(sum(GT**2)/sum((output-GT)**2)); amplitude sensitive')))
            with (root/'evaluation_history.jsonl').open('a') as handle:
                handle.write(json.dumps(report, allow_nan=False)+'\n')
            status(stage='training', evaluation_step=step, best_loss=best_loss)
            emit('evaluation', **report)
            save_checkpoint()
            sync_view()

        if not args.resume or source_checkpoint is not None:
            save_checkpoint()
            evaluate()
        elif origin and not (root/'baseline_evaluation.json').exists():
            if step != 1000:
                raise ValueError('Missing step-1000 continuation baseline; do not advance further')
            evaluate()
        else:
            previous = root/'latest_evaluation.json'
            if previous.exists():
                state['evaluation_step'] = json.loads(previous.read_text())['step']
        while step < args.steps and not state['stop_signal'] and time.monotonic()-started < args.max_seconds:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            tick = time.monotonic()
            prediction, losses = predict(model, loss_fn, sample, 'a', device)
            loss = losses['loss']
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.gradient_clip_val, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            step += 1
            record = dict(step=step, loss=float(loss.detach()), gradient_norm=float(gradient),
                seconds=time.monotonic()-tick,
                peak_allocated_gib=torch.cuda.max_memory_allocated(0)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(0)/2**30)
            with (root/'history.jsonl').open('a') as handle:
                handle.write(json.dumps(record, allow_nan=False)+'\n')
            del prediction, losses, loss
            status(step=step, last_training_loss=record['loss'])
            emit('fit_step', **record)
            if step % 10 == 0 or state['stop_signal']:
                save_checkpoint()
            if not state['stop_signal'] and (step % 20 == 0 or step == args.steps):
                save_checkpoint()
                evaluate()
        save_checkpoint()
        if step == args.steps and state['evaluation_step'] != step and not state['stop_signal']:
            evaluate()
        complete = step == args.steps and state['evaluation_step'] == step
        status(status='completed' if complete else 'stopped', stage='finished',
            stop_reason='step_limit' if complete else ('signal' if state['stop_signal'] else 'time_limit'),
            note='A completed run does not by itself establish successful fitting')
        emit('singlefit_finished', **state)
    except BaseException as exc:
        status(status='failed', stage='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
