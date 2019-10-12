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

unset DISPLAY
export MLPBACKEND=pdf
#srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_D0pp.yml -a MBvspt_ntrkl 
#srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_D0pp.yml -a MBvspt_v0m 
#srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_D0pp.yml -a MBvspt_perc 
#srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_D0pp.yml -a SPDvspt 
#srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_D0pp.yml -a V0mvspt 
#srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_D0pp.yml -a V0mvspt_perc_v0m 
wait
#srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_LcpK0spp.yml -a MBvspt_ntrkl 
#srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_LcpK0spp.yml -a MBvspt_v0m 
#srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_LcpK0spp.yml -a MBvspt_perc 
#srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_LcpK0spp.yml -a SPDvspt 
#srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_LcpK0spp.yml -a V0mvspt 
#srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_LcpK0spp.yml -a V0mvspt_perc_v0m 
wait
srun python do_entire_analysis.py -r default_pre.yml -d data/database_ml_parameters_Dspp.yml 
srun python do_entire_analysis.py -r default_train.yml -d data/database_ml_parameters_Dspp.yml
srun python do_entire_analysis.py -r default_apply.yml -d data/database_ml_parameters_Dspp.yml


srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_Dspp.yml -a MBvspt_ntrkl 
srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_Dspp.yml -a MBvspt_v0m 
srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_Dspp.yml -a MBvspt_perc 
srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_Dspp.yml -a SPDvspt 
srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_Dspp.yml -a V0mvspt 
srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_Dspp.yml -a V0mvspt_perc_v0m 
wait
#srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_LcpKpipp.yml -a MBvspt_ntrkl 
#srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_LcpKpipp.yml -a MBvspt_v0m 
#srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_LcpKpipp.yml -a MBvspt_perc 
#srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_LcpKpipp.yml -a SPDvspt 
#srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_LcpKpipp.yml -a V0mvspt 
#srun python do_entire_analysis.py -r default_ana.yml -d data/database_ml_parameters_LcpKpipp.yml -a V0mvspt_perc_v0m 

wait

srun python plot_hfmassfitter.py
srun python plot_hfptspectrum.py
srun python plot_hfptspectrum_years.py
