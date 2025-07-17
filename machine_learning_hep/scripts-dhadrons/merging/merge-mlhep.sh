#!/bin/bash

MLHEP_DIR="/data8/majak/MLHEP"
OUTPUT_DIR="/data8/majak/MLHEP/input-d2h-fitter-012025"
OUTPUT_DIR_EFF="/data8/majak/MLHEP/input-fd-012025"

RESDIR_PATTERN="${MLHEP_DIR}/results-24012025-hyp-ml-luigi-cuts_fd_"
PERM_PATTERN="fd_precise_"

FD_12=(0.000 0.200 0.250 0.300 0.350 0.380 0.415 0.430 0.470 0.500 0.520 0.550 0.570 0.590 0.610 0.630 0.650 0.670 0.690)
FD_23=(0.000 0.290 0.320 0.350 0.380 0.410 0.430 0.450 0.470 0.490 0.510 0.530 0.550 0.570 0.590 0.610 0.630 0.650 0.670)
FD_34=(0.000 0.290 0.320 0.350 0.370 0.390 0.410 0.425 0.450 0.470 0.490 0.510 0.530 0.550 0.570 0.590 0.610 0.630 0.650)
FD_45=(0.000 0.130 0.150 0.170 0.190 0.210 0.230 0.250 0.270 0.290 0.320 0.350 0.370 0.390 0.410 0.430 0.450 0.470 0.490)
FD_56=(0.000 0.110 0.130 0.150 0.170 0.190 0.210 0.230 0.250 0.270 0.290 0.310 0.330 0.350 0.370 0.390 0.410 0.430 0.450)
FD_67=(0.000 0.130 0.150 0.170 0.190 0.210 0.230 0.250 0.270 0.290 0.320 0.350 0.370 0.390 0.410 0.430 0.450 0.470 0.490)
FD_78=(0.000 0.130 0.150 0.170 0.190 0.210 0.230 0.250 0.270 0.290 0.320 0.350 0.370 0.390 0.410 0.430 0.450 0.470 0.490)
FD_810=(0.000 0.130 0.150 0.170 0.190 0.210 0.230 0.250 0.270 0.290 0.320 0.350 0.370 0.390 0.410 0.430 0.450 0.470 0.490)
FD_1012=(0.000 0.210 0.230 0.250 0.270 0.290 0.310 0.330 0.350 0.370 0.390 0.410 0.430 0.450 0.470 0.490 0.510 0.530 0.550)
FD_1216=(0.000 0.210 0.230 0.250 0.270 0.290 0.310 0.330 0.350 0.370 0.390 0.410 0.430 0.450 0.470 0.490 0.510 0.530 0.550)
FD_1624=(0.000 0.090 0.110 0.130 0.150 0.170 0.190 0.210 0.230 0.250 0.270 0.290 0.310 0.330 0.350 0.370 0.390 0.410 0.430)

for i in "${!FD_12[@]}" ; do
  fd12=${FD_12[i]}
  fd23=${FD_23[i]}
  fd34=${FD_34[i]}
  fd45=${FD_45[i]}
  fd56=${FD_56[i]}
  fd67=${FD_67[i]}
  fd78=${FD_78[i]}
  fd810=${FD_810[i]}
  fd1012=${FD_1012[i]}
  fd1216=${FD_1216[i]}
  fd1624=${FD_1624[i]}
  echo "${i} fd ${fd12} ${fd23} ${fd34} ${fd45} ${fd56} ${fd67} ${fd78} ${fd810} ${fd1012} ${fd1216} ${fd1624}"

  RESPATH="${OUTPUT_DIR}/projections_fd_precise_${fd12}_${fd23}_${fd34}_${fd45}_${fd56}_${fd67}_${fd78}_${fd810}_${fd1012}_${fd1216}_${fd1624}.root"

  python merge_histomass.py \
    -n hmassfPt \
    -o ${RESPATH} \
    -i "${RESDIR_PATTERN}${fd12}/LHC23pp_pass4/Results/resultsdatatot/masshisto.root" \
    -i "${RESDIR_PATTERN}${fd23}/LHC23pp_pass4/Results/resultsdatatot/masshisto.root" \
    -i "${RESDIR_PATTERN}${fd34}/LHC23pp_pass4/Results/resultsdatatot/masshisto.root" \
    -i "${RESDIR_PATTERN}${fd45}/LHC23pp_pass4/Results/resultsdatatot/masshisto.root" \
    -i "${RESDIR_PATTERN}${fd56}/LHC23pp_pass4/Results/resultsdatatot/masshisto.root" \
    -i "${RESDIR_PATTERN}${fd67}/LHC23pp_pass4/Results/resultsdatatot/masshisto.root" \
    -i "${RESDIR_PATTERN}${fd78}/LHC23pp_pass4/Results/resultsdatatot/masshisto.root" \
    -i "${RESDIR_PATTERN}${fd810}/LHC23pp_pass4/Results/resultsdatatot/masshisto.root" \
    -i "${RESDIR_PATTERN}${fd1012}/LHC23pp_pass4/Results/resultsdatatot/masshisto.root" \
    -i "${RESDIR_PATTERN}${fd1216}/LHC23pp_pass4/Results/resultsdatatot/masshisto.root" \
    -i "${RESDIR_PATTERN}${fd1624}/LHC23pp_pass4/Results/resultsdatatot/masshisto.root"

  RESPATH="${OUTPUT_DIR_EFF}/eff_fd_precise_${fd12}_${fd23}_${fd34}_${fd45}_${fd56}_${fd67}_${fd78}_${fd810}_${fd1012}_${fd1216}_${fd1624}.root"

  python merge_histos.py \
    -n eff \
    -n eff_fd \
    -o ${RESPATH} \
    -i "${RESDIR_PATTERN}${fd12}/LHC24pp_mc/Results/resultsmctot/efficienciesLcpKpiRun3analysis.root" \
    -i "${RESDIR_PATTERN}${fd23}/LHC24pp_mc/Results/resultsmctot/efficienciesLcpKpiRun3analysis.root" \
    -i "${RESDIR_PATTERN}${fd34}/LHC24pp_mc/Results/resultsmctot/efficienciesLcpKpiRun3analysis.root" \
    -i "${RESDIR_PATTERN}${fd45}/LHC24pp_mc/Results/resultsmctot/efficienciesLcpKpiRun3analysis.root" \
    -i "${RESDIR_PATTERN}${fd56}/LHC24pp_mc/Results/resultsmctot/efficienciesLcpKpiRun3analysis.root" \
    -i "${RESDIR_PATTERN}${fd67}/LHC24pp_mc/Results/resultsmctot/efficienciesLcpKpiRun3analysis.root" \
    -i "${RESDIR_PATTERN}${fd78}/LHC24pp_mc/Results/resultsmctot/efficienciesLcpKpiRun3analysis.root" \
    -i "${RESDIR_PATTERN}${fd810}/LHC24pp_mc/Results/resultsmctot/efficienciesLcpKpiRun3analysis.root" \
    -i "${RESDIR_PATTERN}${fd1012}/LHC24pp_mc/Results/resultsmctot/efficienciesLcpKpiRun3analysis.root" \
    -i "${RESDIR_PATTERN}${fd1216}/LHC24pp_mc/Results/resultsmctot/efficienciesLcpKpiRun3analysis.root" \
    -i "${RESDIR_PATTERN}${fd1624}/LHC24pp_mc/Results/resultsmctot/efficienciesLcpKpiRun3analysis.root"
done
