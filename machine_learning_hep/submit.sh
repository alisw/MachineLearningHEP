#!/bin/bash

#STAGE="pre"
#STAGE="train"
#STAGE="apply"
#STAGE="complete"
#STAGE="analyzer"
#STAGE="variations"
STAGE="systematics"

#DBDIR="data_prod_20200417"
DBDIR="data_prod_20200304"
#DBDIR="pKpi"
#DBDIR="JetAnalysis"

DATABASE="jetinclusive"
#DATABASE="Dspp"
#DATABASE="LcpK0spp"
#DATABASE="LcpKpi"

#SUFFIX="_0417"
#SUFFIX="_0304"
#SUFFIX="010"
#SUFFIX="3050"
SUFFIX=""
#ANALYSIS="MBvspt"
#ANALYSIS="MBvspt_perc_v0m"
#ANALYSIS="MBvspt_ntrkl"
#ANALYSIS="SPDvspt_ntrkl"
#ANALYSIS="jet_FF"
#ANALYSIS="jet_zg"
ANALYSIS="jet_rg"
#ANALYSIS="jet_nsd"

DATABASE_DEFAULT="${DATABASE}${SUFFIX}"
DATABASE_VARIATION="${DATABASE}_${ANALYSIS}"

CONFIG="submission/default_${STAGE}.yml"
DB_DEFAULT="data/${DBDIR}/database_${DATABASE_DEFAULT}.yml"
DB_VARIATION="data/${DBDIR}/database_variations_${DATABASE_VARIATION}.yml"
DIR_RESULTS="/data/DerivedResultsJets/Inclusive/vAN-20200304_ROOT6-1/"

CMD_ANA="python do_entire_analysis.py -a ${ANALYSIS} -r ${CONFIG} -d ${DB_DEFAULT} -c"

if [[ "${STAGE}" == "variations" ]]
then
    echo "Running the variation script for the ${ANALYSIS} analysis of ${DATABASE_DEFAULT}"
    ./submit_variations.sh ${DB_DEFAULT} ${DB_VARIATION} ${ANALYSIS}
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

