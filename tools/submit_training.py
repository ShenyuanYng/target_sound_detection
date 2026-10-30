"""Submit a standalone TSE experiment with bounded checkpoint continuations.

Default is a read-only preview. --submit queues the first job; --execute runs
inside an existing allocation. Both freeze the same configuration and use the
existing submit_continuation / resume_training implementation.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from slurm.submit_continuation import RESOURCES, submit_chain

VARIANTS = {
    'audioset_strong_fsd50k_tse': 'tse_phase1',
    'audioset_strong_dual_fsd_wrong_tse': 'tse_dual_wrong',
    'audioset_strong_dual_fsd_hierarchy_tse_v2': 'tse_hierarchy_v2',
}
from utils.project_paths import EXPERIMENTS

PROFILES = {
    'ailab01': ('/hdd2/syy12632604/AudioSet',
                '/hdd2/cse30029918/corpora/AudioSetRaw/balance_wav'),
    'ailab02': ('/data/syy12632604/AudioSetMix',
                '/data/syy12632604/AudioSetMix/audio/balance_wav'),
}

TRAINING_OVERRIDES = {
    'batch_size': int, 'learning_rate': float, 'epochs': int, 'patience': int,
    'num_workers': int, 'seed': int, 'train_percent_check': float,
    'val_percent_check': float, 'precision': str, 'accumulate_grad_batches': int,
    'max_steps': int,
}


def validate_tag(value):
    if value and not re.fullmatch(r'[\w-]{1,40}', value):
        raise ValueError('Tags must contain 1-40 letters, digits, underscores or hyphens')
    return value


def run_name_prefix(config, tag=''):
    """Readable experiment identity; the Slurm job ID makes each run distinct."""
    validate_tag(tag)
    name = (f'run_b{config["batch_size"]}_lr{config["learning_rate"]:g}'
            f'_e{config["epochs"]}_{config.get("precision", "32-true")}')
    if config.get('tse_smallfit'):
        name += f'_smallfit_s{config["max_steps"]}'
    return name + (f'_{tag}' if tag else '')


def submission_run(config, spec, job):
    # Existing frozen submissions predate readable names and still use run_job<ID>.
    return Path(config['save_path']) / f'{spec.get("run_name_prefix", "run")}_job{job}'


def prepare_config(config, node, manifest_tag=''):
    config = dict(config)
    if 'max_steps' in config and not config.get('tse_smallfit'):
        raise ValueError('max_steps is currently supported by the fixed-mixture smallfit configuration only')
    validate_tag(manifest_tag)
    if config.get('dataset') not in VARIANTS or config.get('test_after_fit') is not False:
        raise ValueError('Expected TSE Train/CV configuration')
    for name in ('train_percent_check', 'val_percent_check'):
        if type(config.get(name)) is not float or not 0 < config[name] <= 1.0:
            raise ValueError(f'{name} must be a float in (0, 1]')
    for name in ('batch_size', 'epochs'):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name in ('patience', 'num_workers'):
        if type(config.get(name)) is not int or config[name] < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    rate = config.get('learning_rate')
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if type(config.get('seed')) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if (type(config.get('accumulate_grad_batches', 1)) is not int
            or config.get('accumulate_grad_batches', 1) < 1):
        raise ValueError('accumulate_grad_batches must be a positive integer')
    if config.get('precision', '32-true') not in ('32-true', '16-mixed', 'bf16-mixed'):
        raise ValueError('precision must be 32-true, 16-mixed or bf16-mixed')
    root, audio = PROFILES[node]
    variant = VARIANTS[config['dataset']]
    run_variant = variant + ('_smallfit' if config.get('tse_smallfit') else '')
    derived = EXPERIMENTS/'manifests'/run_variant/node
    if manifest_tag:
        derived = derived/manifest_tag
    if variant == 'tse_phase1':
        audio_cache = Path(root)/'tse_phase1_v1/fsd_cache_16k'
    elif variant == 'tse_hierarchy_v2':
        audio_cache = Path('/data/syy12632604/tse_hierarchy_v2/fsd_cache_16k')
    else:
        audio_cache = Path('/data/syy12632604/tse_dual_wrong')/node/'fsd_cache_16k'
    config.update(data_path=root, mix_audio_dirs=[audio],
        mix_fsd_dev_audio='/data/cse12110411/FSD50K/FSD50K.dev_audio',
        mix_fsd_ground_truth='/data/cse12110411/FSD50K/FSD50K.ground_truth',
        mix_cache_dir=str(audio_cache),
        mix_cv_manifest_path=str(derived/'cv_seed7.json'),
        mix_background_split_path=str(derived/'background_split.json'),
        accelerator='gpu', devices=1, save_path=str(EXPERIMENTS/run_variant/node),
        save_name='run_pending')
    if config.get('tse_ontology_path'):
        config['tse_ontology_path'] = str(PROJECT/config['tse_ontology_path'])
    if variant == 'tse_hierarchy_v2':
        config.update(tse_artifact_root=str(derived),
                      tse_condition_vocabulary_path=str(derived/'condition_vocabulary.json'))
    from argparse import Namespace
    from utils.training_config import validate_training_data_config
    validate_training_data_config(Namespace(**config))
    return config


def submit(args):
    config = yaml.safe_load(Path(args.exp_def).read_text())
    if not isinstance(config, dict):
        raise ValueError('Experiment YAML must contain a configuration mapping')
    overrides = {name: getattr(args, name) for name in TRAINING_OVERRIDES if hasattr(args, name)}
    config.update(overrides)
    config = prepare_config(config, args.node, getattr(args, 'manifest_tag', ''))
    if args.num_jobs < 1 or args.cpus < 1:
        raise ValueError('Job/CPU counts must be positive')
    if (config.get('tse_pilot_artifacts') or config.get('tse_smallfit')) and args.num_jobs != 1:
        raise ValueError('Diagnostic runs use one allocation; choose --num-jobs 1')
    # Keep the environment's executable path: resolving a venv symlink can
    # accidentally launch its base interpreter without the installed packages.
    executable = shutil.which(args.python_bin)
    if executable is None:
        raise FileNotFoundError(f'Python executable not found: {args.python_bin}')
    python = Path(executable).absolute()
    partition, gres = RESOURCES[args.node]
    spec = dict(node=args.node, num_jobs=args.num_jobs, python_bin=str(python),
        partition=partition, gres=gres, cpus_per_task=args.cpus, mem=args.mem,
        time_limit=args.time, account=args.account,
        run_name_prefix=run_name_prefix(config, getattr(args, 'tag', '')))
    execute = getattr(args, 'execute', False)
    if args.submit and execute:
        raise ValueError('Choose either Slurm submission or execution in an allocation')
    if not args.submit and not execute:
        print(json.dumps(dict(action='preview', resources=spec, config=config,
            training_overrides=overrides,
            run_directory=str(submission_run(config, spec, '<JOB_ID>'))), indent=2))
        return
    if args.submit and os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Submit outside an allocated Slurm job')
    if execute:
        node = os.environ.get('SLURMD_NODENAME', socket.gethostname().split('.')[0])
        if not os.environ.get('SLURM_JOB_ID', '').isdigit() or node != args.node:
            raise ValueError('Execute only inside the requested Slurm node allocation')
    if PROJECT != EXPERIMENTS.parent:
        raise ValueError(f'Submit from the shared server checkout: {EXPERIMENTS.parent}')
    staging = EXPERIMENTS/'.submissions'
    staging.mkdir(parents=True, exist_ok=True)
    frozen = Path(tempfile.mkdtemp(prefix=args.node+'_', dir=staging))
    (frozen/'effective_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    spec.update(submitted_at=datetime.datetime.now().astimezone().isoformat(),
                source_config=str(Path(args.exp_def).resolve()),
                training_overrides=overrides,
                launch_mode='allocated' if execute else 'submit')
    (frozen/'submission.json').write_text(json.dumps(spec, indent=2)+'\n')
    logs = frozen/'scheduler_logs'
    logs.mkdir(exist_ok=True)
    if execute:
        (frozen/'initial_job_id').write_text(os.environ['SLURM_JOB_ID']+'\n')
        run_submission(frozen)
        return
    command = ['sbatch', '--parsable', '--export=ALL', '--ntasks=1',
        f'--chdir={PROJECT}', f'--job-name={VARIANTS[config["dataset"]]}-{args.node}',
        f'--nodelist={args.node}', f'--partition={partition}', f'--gres={gres}',
        f'--cpus-per-task={args.cpus}', f'--mem={args.mem}', f'--time={args.time}',
        f'--output={logs}/%x-%j.out', f'--error={logs}/%x-%j.err']
    if args.account:
        command.append(f'--account={args.account}')
    command.extend([str(PROJECT/'slurm/tse_train.sbatch'), str(frozen), str(python)])
    result = subprocess.check_output(command, text=True, cwd=PROJECT).strip().split(';')[0]
    if not result.isdigit():
        raise RuntimeError(f'Unexpected sbatch output: {result}')
    # The worker may already be reading submission.json: keep that file immutable.
    (frozen/'initial_job_id').write_text(result+'\n')
    print(json.dumps(dict(initial_job_id=result, total_jobs=args.num_jobs,
        run_dir=str(submission_run(config, spec, result)),
        frozen_config=str(frozen/'effective_config.yaml')), indent=2))


def run_submission(directory):
    frozen = Path(directory).resolve(strict=True)
    spec = json.loads((frozen/'submission.json').read_text())
    node = os.environ.get('SLURMD_NODENAME', socket.gethostname().split('.')[0])
    job = os.environ.get('SLURM_JOB_ID', '')
    if node != spec['node'] or not job.isdigit():
        raise ValueError('Run only inside the requested Slurm node allocation')
    # Preserve the scheduler's GPU visibility; never hard-code a physical GPU.
    os.chdir(PROJECT)
    config = yaml.safe_load((frozen/'effective_config.yaml').read_text())
    run = submission_run(config, spec, job)
    pilot = bool(config.get('tse_pilot_artifacts', False))
    run.parent.mkdir(parents=True, exist_ok=True)
    run.mkdir()  # Exclusive: never overwrite another run.
    if not pilot:
        (run/'logs').mkdir()
        (run/'continuation').mkdir()
        (run/'submitted_config.yaml').write_bytes((frozen/'effective_config.yaml').read_bytes())
    # Do not create the historical compute_node marker, which selects the SED
    # shell dispatcher. Generic resume_training reads our checkpoint directly.
    if not pilot:
        (run/'node.txt').write_text(node+'\n')
    log = run/'run.log' if pilot else run/'logs/training.log'
    print(f'Run directory: {run}\nTraining log: {log}', flush=True)
    with log.open('a') as output:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(output.fileno(), 1)
        os.dup2(output.fileno(), 2)
    try:
        if spec['num_jobs'] > 1:
            submit_chain(run, job, spec['num_jobs']-1, node, spec['python_bin'],
                starting_job=True, partition=spec['partition'], gres=spec['gres'],
                cpus_per_task=spec['cpus_per_task'], mem=spec['mem'],
                time_limit=spec['time_limit'], account=spec['account'])
        from train import main as train
        config.update(save_path=str(run.parent), save_name=run.name)
        hp = argparse.Namespace(**config)
        from utils.tse_config import validate_tse_config
        validate_tse_config(hp)
        if hp.dataset == 'audioset_strong_dual_fsd_hierarchy_tse_v2':
            from datasets.tse_hierarchy_dataset import validate_artifact_paths
            validate_artifact_paths(hp)
        elif hp.dataset == 'audioset_strong_dual_fsd_wrong_tse':
            from datasets.tse_dual_dataset import validate_artifact_paths
            validate_artifact_paths(hp)
        print(f'node={node} job={job} batch_size={hp.batch_size} epochs={hp.epochs} '
              f'total_jobs={spec["num_jobs"]}', flush=True)
        train(hp)
    except Exception as error:
        (run/'continuation').mkdir(exist_ok=True)
        (run/'continuation/failed').write_text(
            f'Initial job {job} failed: {type(error).__name__}: {error}\nSee {log}\n')
        raise


def main(default_config='tse_strong_dual_fsd_hierarchy_v2.yaml'):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--node', choices=PROFILES)
    parser.add_argument('--exp-def', default=str(PROJECT/'exp_defs'/default_config))
    parser.add_argument('--num-jobs', type=int, default=1, help='Total jobs including the first')
    parser.add_argument('--python-bin', default=sys.executable)
    parser.add_argument('--cpus', type=int, default=8)
    parser.add_argument('--mem', default='32G')
    parser.add_argument('--time', default='2-00:00:00')
    parser.add_argument('--account')
    parser.add_argument('--tag', default='', help='Short label appended to the readable run name')
    parser.add_argument('--manifest-tag', default='', help='New namespace for derived manifests; no repartitioning')
    for name, kind in TRAINING_OVERRIDES.items():
        parser.add_argument('--'+name.replace('_', '-'), type=kind, default=argparse.SUPPRESS,
                            help='Override the selected YAML value; omitted values are inherited')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--submit', action='store_true', help='Queue first job; default is preview only')
    action.add_argument('--execute', action='store_true', help='Train inside the requested existing Slurm allocation')
    parser.add_argument('--run-submission', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run_submission:
        run_submission(args.run_submission)
    else:
        if args.node is None:
            parser.error('--node is required')
        submit(args)


if __name__ == '__main__':
    main()
