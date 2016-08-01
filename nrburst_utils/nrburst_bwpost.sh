#!/bin/bash

targetdir=${1}

# Make big pickle
pushd ${targetdir}
${HOME}/Projects/numrel_bursts/nrburst_utils/nrburst_bwpostpickle

# Compute overlaps
for fmin in 16 24 32
do
    pickle=`ls *pickle`
    ${HOME}/Projects/numrel_bursts/nrburst_utils/nrburst_bwreducemoments.py ${fmin} ${pickle}
    mv *npz ..
done

popd


