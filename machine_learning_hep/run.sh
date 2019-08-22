#!/bin/bash

#SBATCH --output=slurm-%J.out
#SBATCH --error=slurm-%J.out

function die
{
    echo $1
    exit
}

JOBIDX="-0"
[ -n "${SLURM_ARRAY_TASK_ID}" ] && JOBIDX="-${SLURM_ARRAY_TASK_ID}"
export JOBIDX

export MLPBACKEND=pdf
python do_entire_analysis.py -r default_${1}.yml -d data/database_ml_parameters_${2}.yml
