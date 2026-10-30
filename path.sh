#!/usr/bin/env bash

export PATH="$PWD/tools:$PWD:$PATH"
export LC_ALL=C
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
