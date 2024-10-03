#!/bin/bash

##### Configuration

# Analysis stage

# STAGE="all_off"        # all steps disabled
STAGE="full_analysis"    # stage preprocess + stage analysis (requires train output in "(data|mc)/prefix_dir")
# STAGE="preprocess"     # conversion, skimming (requires train output in "(data|mc)/prefix_dir")
# STAGE="data"           # stage preprocess: data (requires train output in "data/prefix_dir")
# STAGE="mc"             # stage preprocess: mc (requires train output in "mc/prefix_dir")
# STAGE="mltrain"        # ml_study
# STAGE="mlapp"          # mlapplication
# STAGE="analysis"       # stage processor + stage analyzer (requires stage preprocess done)
# STAGE="processor"      # analysis/(data|mc)/(histomass|efficiency) (requires stage preprocess done)
# STAGE="analyzer"       # analysis/steps (requires stage processor done)
# STAGE="variations"     # run analysis variations (requires stage analyzer done)
# STAGE="systematics"    # calculate and plot systematics (requires stage variations done)
# STAGE="plotting"       # make analysis plots (requires stage systematics done)

# Suffix of the analysis database name

DATABASE="D0Jet_pp"
# DATABASE="LcJet_pp"

# Name of the analysis section in the analysis database

ANALYSIS="jet_obs"

##### Initialisation

DIR_THIS="$(dirname "$(realpath "$0")")"  # This directory
DBDIR="data/data_run3"
DB_DEFAULT="${DIR_THIS}/${DBDIR}/database_ml_parameters_${DATABASE}.yml"
LOG="log_${STAGE}_${DATABASE}_${ANALYSIS}.log"

##### Execution

echo "$(date) Start"
echo "Running the \"${STAGE}\" stage of the \"${ANALYSIS}\" analysis from the \"${DATABASE}\" database"

if [[ "${STAGE}" == "plotting" ]]; then
    echo "Log file: $LOG"
    python "${DIR_THIS}/plotting/plot_jetsubstructure_run3.py" -d "${DB_DEFAULT}" -a "${ANALYSIS}" > "${LOG}" 2>&1
elif [[ "${STAGE}" == "systematics" ]]; then
    echo "Log file: $LOG"
    python "${DIR_THIS}/analysis/do_systematics.py" -d "${DB_DEFAULT}" -a "${ANALYSIS}" > "${LOG}" 2>&1
elif [[ "${STAGE}" == "variations" ]]; then
    DB_VARIATION="${DIR_THIS}/${DBDIR}/database_variations_${DATABASE}_${ANALYSIS}.yml"
    CONFIG_FILE="${DIR_THIS}/submission/analysis.yml"
    "${DIR_THIS}/submit_variations.sh" "${DB_DEFAULT}" "${DB_VARIATION}" "${ANALYSIS}" "${CONFIG_FILE}"
else
    echo "Log file: $LOG"
    CONFIG_FILE="${DIR_THIS}/submission/${STAGE}.yml"
    CMD_ANA="mlhep -a ${ANALYSIS} -r ${CONFIG_FILE} -d ${DB_DEFAULT} -b --delete"
    ${CMD_ANA} > "${LOG}" 2>&1
fi || echo "Error"

LOG_ERR="${LOG/.log/_err.log}"
echo "Grepping issues into ${LOG_ERR}"
grep -e "Error in " -e "Failed " "${LOG}" > "${LOG_ERR}"
grep -A 1 -e WARN -e ERROR -e FATAL -e CRITICAL "${LOG}" >> "${LOG_ERR}"

echo "$(date) Done"
