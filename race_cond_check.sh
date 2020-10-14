#!/bin/bash

for iteration in {1..20}
do
    for val in {64..1024..64}
    do
        echo "Iteration $iteration: running ./dev_func_fft $val..."
        build/dev_func_fft $val
    done
done


