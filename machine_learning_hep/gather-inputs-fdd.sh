#!/bin/bash

MLHEP_DIR="/data8/majak/MLHEP"
OUTPUT_DIR="${MLHEP_DIR}/input-fd-10092024"

RESDIR_PATTERN="${MLHEP_DIR}/results-2308-hyp-ml_fd_precise_"

for dir in "${RESDIR_PATTERN}"1224_split* ; do
  suffix=${dir##"${RESDIR_PATTERN}"}
  echo "$suffix"

  cp "${dir}/LHC22pp_mc/Results/prod_LHC24d3b/resultsmctot/efficienciesLcpKpiRun3analysis.root" \
     "${OUTPUT_DIR}/efficienciesLcpKpiRun3analysis_${suffix}.root"
  #cp "${dir}/LHC22pp/Results/resultsdatatot/Yields_LcpKpi_Run3analysis.root" \
  #   "${OUTPUT_DIR}/yieldsLcpKpiRun3analysis-${suffix}.root"
done
