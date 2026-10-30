"""Experiment-3 shortcut to the standalone TSE submitter."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.submit_training import prepare_config, run_submission, submit, main

if __name__ == '__main__':
    main(default_config='tse_strong_dual_fsd_wrong.yaml')
