#!/bin/bash

# Gather MLHEP efficiencies and mass fits for all chosen non-prompt cuts into a single directory.
# The cut variation script requires files from a single directory.

MLHEP_DIR="/data8/majak/MLHEP"
OUTPUT_DIR="${MLHEP_DIR}/input-fd-012025"

RESDIR_PATTERN="${MLHEP_DIR}/results-24012025-hyp-ml-luigi-cuts_"
PERM_PATTERN="fd_"

for dir in "${RESDIR_PATTERN}${PERM_PATTERN}"0.[0-9][0-9][0-9]* ; do
  suffix=${dir##"${RESDIR_PATTERN}"}
  echo "$suffix"

  cp "${dir}/LHC24pp_mc/Results/resultsmctot/efficienciesLcpKpiRun3analysis.root" \
     "${OUTPUT_DIR}/efficienciesLcpKpiRun3analysis_${suffix}.root"
  cp "${dir}/LHC23pp_pass4/Results/resultsdatatot/yields_LcpKpi_Run3analysis.root" \
     "${OUTPUT_DIR}/yieldsLcpKpiRun3analysis-${suffix}"
done
