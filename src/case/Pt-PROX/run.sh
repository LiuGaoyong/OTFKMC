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

# Reference:
#   https://linkinghub.elsevier.com/retrieve/pii/S0009250911003022
#       The reaction mixture consisted of 0.8% CO, 1.6% O2, 51% H2,
#       balanced in He, flown at a rate of 200 cm3/min.

echo 'glob:
    max_step: 100
match:
    by_reference: false
gas:
    lst:
      - name: H2
        pressure: 51675.75
      - name: O2
        pressure: 1621.2
      - name: CO
        pressure: 810.6
      - name: CO2
      - name: H2O
    mode: isobaric
    volume_buffer_zise: 10
parallel:
    backend: ray
    njobs: -1
' > config.yaml

python build.py
nohup otfkmc structure.xyz --config config.yaml > run.txt 2>&1 | echo $! > pid.run &