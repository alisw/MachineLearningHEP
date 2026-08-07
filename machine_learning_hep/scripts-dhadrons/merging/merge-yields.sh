#!/bin/bash

# Merge yields at 0.0 for prompt cross section
python merge_histos.py -o "/data8/majak/crosssec/202502/yieldsLcpKpiRun3analysis_fd_0.000.root" \
  -i "/data8/majak/MLHEP/input-fd-012025/yields-fd_0.000-poly-fixed-sigma.root" \
  -i "/data8/majak/MLHEP/input-fd-012025/yields-fd_0.000-cheb-fixed-sigma-120-190.root" \
  -i "/data8/majak/MLHEP/input-fd-012025/yields-fd_0.000-cheb-fixed-sigma-120-190.root" \
  -i "/data8/majak/MLHEP/input-fd-012025/yields-fd_0.000-cheb-fixed-sigma-120-190.root" \
  -i "/data8/majak/MLHEP/input-fd-012025/yields-fd_0.000-cheb-fixed-sigma-120-190.root" \
  -i "/data8/majak/MLHEP/input-fd-012025/yields-fd_0.000-cheb-fixed-sigma-120-190.root" \
  -i "/data8/majak/MLHEP/input-fd-012025/yields-fd_0.000-cheb-fixed-sigma-120-190.root" \
  -i "/data8/majak/MLHEP/input-fd-012025/yields-fd_0.000-cheb-fixed-sigma-120-190.root" \
  -i "/data8/majak/MLHEP/input-fd-012025/yields-fd_0.000-cheb-fixed-sigma-120-190.root" \
  -i "/data8/majak/MLHEP/input-fd-012025/yields-fd_0.000-cheb-fixed-sigma-120-190.root" \
  -i "/data8/majak/MLHEP/input-fd-012025/yields-fd_0.000-cheb-fixed-sigma-120-190.root"
