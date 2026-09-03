#!/usr/bin/env bash

if [[ -f pid.run ]]; then
    kill -9 $(cat pid.run)
fi
pkill otfkmc
pkill python
pkill Python

for f in $(ls); do
    if [[ $f != 'run.sh' ]] && [[ $f != 'kill.sh' ]] && [[ $f != 'build.py' ]]; then
        rm -f $f
    fi
done
