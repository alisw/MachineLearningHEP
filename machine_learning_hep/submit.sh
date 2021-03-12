#!/bin/bash

#STAGE="pre"
#STAGE="train"
#STAGE="apply"
#STAGE="complete"
#STAGE="analyzer"
STAGE="variations"
#STAGE="systematics"

#DBDIR="data_prod_20200417"
DBDIR="data_prod_20200824"
#DBDIR="pKpi"
#DBDIR="JetAnalysis"

#DATABASE="D0pp"
#DATABASE="Dspp"
DATABASE="LcpK0spp"
#DATABASE="LcpKpi"

#SUFFIX="_0417"
SUFFIX="_0824_jet_2_6"
#SUFFIX="_0824_jet_6_12"
#SUFFIX="_0824_6_12"
#SUFFIX="_0304_jet" # Lc
#SUFFIX="010"
#SUFFIX="3050"

#ANALYSIS="MBvspt"
#ANALYSIS="MBvspt_perc_v0m"
#ANALYSIS="MBvspt_ntrkl"
#ANALYSIS="SPDvspt_ntrkl"
#ANALYSIS="jet_FF"
ANALYSIS="jet_r_shape"
#ANALYSIS="jet_rg"
#ANALYSIS="jet_nsd"

DATABASE_DEFAULT="${DATABASE}${SUFFIX}"
DATABASE_VARIATION="${DATABASE}_${ANALYSIS}"
#DO_PROC=1

CONFIG="submission/default_${STAGE}.yml"
DB_DEFAULT="data/${DBDIR}/database_ml_parameters_${DATABASE_DEFAULT}.yml"
DB_VARIATION="data/${DBDIR}/database_variations_${DATABASE_VARIATION}.yml"
DIR_RESULTS="/data/Derived_testResults/Jets/Lc/2_6/vAN-20200824_ROOT6-1/"

CMD_ANA="python do_entire_analysis.py -a ${ANALYSIS} -r ${CONFIG} -d ${DB_DEFAULT} -c"

if [[ "${STAGE}" == "variations" ]]
then
    echo "Running the variation script for the ${ANALYSIS} analysis of ${DATABASE_DEFAULT}"
    ./submit_variations.sh ${DB_DEFAULT} ${DB_VARIATION} ${ANALYSIS} ${DO_PROC}
else
    echo "Running the ${STAGE} stage of the ${ANALYSIS} analysis of ${DATABASE_DEFAULT}"
    \time -f "time: %E\nCPU: %P" ${CMD_ANA}
fi

# Exit if error.
if [ ! $? -eq 0 ]; then echo "Error"; exit 1; fi

echo -e "\n$(date)"

echo -e "\nCleaning ${DIR_RESULTS}"
./clean_results.sh ${DIR_RESULTS}

echo -e "\nDone"

exit 0
