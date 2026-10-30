"""Submit a bounded chain after an existing job, without changing that job."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


# The L40 partition includes both L40 and RTX 5880 Ada GPUs on ailab01.
RESOURCES = {"ailab01": ("L40", "gpu:1"),
             "ailab02": ("RTX5880Ada", "gpu:rtx5880ada:1"),
             "ailab03": ("RTXPRO5000", "gpu:rtxpro5000:1")}


def submit_chain(run_dir, after_job, num_jobs, node, python_bin, starting_job=False,
                 *, partition=None, gres=None, cpus_per_task=6, mem="32G",
                 time_limit="2-00:00:00", account=None):
    project = Path(__file__).resolve().parents[1]
    run_dir = Path(run_dir).resolve()
    if num_jobs < 1:
        raise ValueError("num_jobs must be positive (number of ADDITIONAL jobs)")
    if cpus_per_task < 1:
        raise ValueError("cpus_per_task must be positive")
    if starting_job and os.environ.get("SLURM_JOB_ID") != str(after_job):
        raise ValueError("starting-job is only allowed inside the original Slurm job")
    if not starting_job and not (run_dir / "mylast.ckpt").is_file():
        raise FileNotFoundError(f"No checkpoint: {run_dir / 'mylast.ckpt'}")
    if not Path(python_bin).is_file():
        raise FileNotFoundError(python_bin)
    # Verify the user selected their own current job and the same node.
    info = subprocess.check_output(["scontrol", "show", "job", str(after_job), "-o"], text=True)
    fields = dict(word.split("=", 1) for word in info.split() if "=" in word)
    if fields.get("UserId", "").split("(")[-1].rstrip(")") != str(os.getuid()):
        raise ValueError("Predecessor job does not belong to the current user")
    if fields.get("JobState") not in {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED"}:
        raise ValueError("Predecessor must still be active when attaching this chain")
    if node not in {fields.get("NodeList"), fields.get("ReqNodeList")}:
        raise ValueError("Choose the same node as the original run to preserve data paths")
    if Path(fields.get("WorkDir", "")).resolve() != project:
        raise ValueError("Predecessor WorkDir differs from this project")
    if not run_dir.name.endswith(f"_job{after_job}"):
        raise ValueError("Experiment name must end with the original job ID")

    # A recorded submission is never silently repeated. Partial submissions are
    # recorded immediately so users can inspect/cancel them instead of duplicating.
    with (run_dir / ".submit_chain.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record_path = run_dir / "continuation_jobs.json"
        if record_path.exists():
            raise FileExistsError(f"Chain already recorded: {record_path}; inspect it before submitting again")
        chain_dir = run_dir / "continuation"
        chain_dir.mkdir(exist_ok=True)
        (chain_dir / "scheduler_logs").mkdir(exist_ok=True)
        default_partition, default_gres = RESOURCES[node]
        partition, gres = partition or default_partition, gres or default_gres
        record = {"original_job": str(after_job), "num_jobs": num_jobs, "node": node,
                  "run_dir": str(run_dir), "submitted_at": datetime.datetime.now().isoformat(),
                  "resources": {"partition": partition, "gres": gres,
                                "cpus_per_task": cpus_per_task, "mem": mem,
                                "time": time_limit, "account": account}, "jobs": []}
        record_path.write_text(json.dumps(record, indent=2))
        predecessor = str(after_job)
        for index in range(num_jobs):
            command = ["sbatch", "--parsable", f"--dependency=afterany:{predecessor}",
                       "--kill-on-invalid-dep=yes", "--job-name=audioset-continue",
                       f"--partition={partition}", f"--nodelist={node}", f"--gres={gres}",
                       f"--cpus-per-task={cpus_per_task}", f"--mem={mem}", f"--time={time_limit}",
                       f"--chdir={project}", f"--output={chain_dir}/scheduler_logs/continue-%j.out",
                       f"--error={chain_dir}/scheduler_logs/continue-%j.err"]
            if account:
                command.append(f"--account={account}")
            command.extend([str(project / "slurm/continue_training.sbatch"), str(run_dir),
                            str(Path(python_bin).resolve()), str(chain_dir)])
            try:
                output = subprocess.check_output(command, text=True).strip()
                job_id = output.split(";")[0]
                if not job_id.isdigit():
                    raise RuntimeError(f"Unexpected sbatch response: {output}")
            except Exception as error:
                record["submission_error"] = str(error)
                record_path.write_text(json.dumps(record, indent=2))
                raise
            record["jobs"].append({"job_id": job_id, "after_job": predecessor})
            record_path.write_text(json.dumps(record, indent=2))
            print(f"Continuation {index + 1}/{num_jobs}: job {job_id} after job {predecessor}", flush=True)
            predecessor = job_id
        return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--after-job", required=True, type=int)
    parser.add_argument("--additional-jobs", "--num-jobs", dest="num_jobs", type=int,
                        default=3, help="Additional jobs, excluding the current job; --num-jobs is a legacy alias")
    parser.add_argument("--starting-job", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--node", choices=RESOURCES, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--gres", default=None)
    parser.add_argument("--cpus-per-task", type=int, default=6)
    parser.add_argument("--mem", default="32G")
    parser.add_argument("--time", dest="time_limit", default="2-00:00:00")
    parser.add_argument("--account", default=None)
    args = parser.parse_args()
    submit_chain(args.run_dir, args.after_job, args.num_jobs, args.node, args.python_bin,
                 args.starting_job, partition=args.partition, gres=args.gres,
                 cpus_per_task=args.cpus_per_task, mem=args.mem,
                 time_limit=args.time_limit, account=args.account)
