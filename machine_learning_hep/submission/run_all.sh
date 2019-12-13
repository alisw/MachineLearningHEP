#!/bin/bash

#SBATCH --output=slurm-%J.out
#SBATCH --error=slurm-%J.out

function die
{
    echo $1
    exit
}

CASE=${1}

export MLPBACKEND=pdf
srun python do_entire_analysis.py -r default_pre.yml -d ../data/database_ml_parameters_${CASE}.yml
srun python do_entire_analysis.py -r default_train.yml -d ../data/database_ml_parameters_${CASE}.yml
srun python do_entire_analysis.py -r default_apply.yml -d ../data/database_ml_parameters_${CASE}.yml
