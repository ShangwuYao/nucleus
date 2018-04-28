#!/bin/bash
for i in `seq 1 7`;
do
    echo "Running cross validation fold $i"
    KERAS_BACKEND=tensorflow python3 nucleus.py train --dataset=../../../dataset --subset=train --weights=imagenet --cross=$i --logs=../../logs_cross_$i
done

