#!/usr/bin/env bash

for f in $(ls); do
    if [[ $f != 'run.sh' ]] && [[ $f != 'kill.sh' ]] && [[ $f != 'build.py' ]]; then
        rm -f $f
    fi
done

export RAY_DEDUP_LOGS=0
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo 'glob:
    max_step: 100
match:
    by_reference: false
parallel:
    backend: ray
    njobs: -1
' > config.yaml


python build.py 8
# test_n = [8, 12, 16, 20, 25, 32]
# [344, 1156, 2736, 5340, 10425, 21856]
nohup otfkmc structure.xyz --config config.yaml > run.txt 2>&1 | echo $! > pid.run &
