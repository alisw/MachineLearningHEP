#!/bin/bash

FD_12=(0.00 0.21 0.24 0.27 0.30 0.33 0.35 0.37 0.39 0.41 0.44 0.46 0.48 0.50 0.52 0.54 0.56 0.58)
FD_12_OLD=(0.00 0.21 0.24 0.27 0.30 0.33 0.35 0.37 0.39 0.41 0.44 0.46 0.48 0.50 0.52 0.54 0.55 0.58)

DIR_12="/data8/majak/MLHEP/input-fd-23082024"
PTRN_12=("${DIR_12}/yields-bkg_0.20_0.60_fd_" "${DIR_12}/efficienciesLcpKpiRun3analysis_pt-weight_bkg_0.20_0.60_fd_")
SUFFIX_12=("-rebin-1-fixed-sigma.root" ".root")
DIR_212="/data8/majak/MLHEP/input-fd-23082024"
PTRN_212=("${DIR_212}/yields-fd_precise_rebin4_bkg_0.20_0.60_fd_" "${DIR_212}/efficienciesLcpKpiRun3analysis_fd_precise_rebin4_bkg_0.20_0.60_fd_")
SUFFIX_212=("-fixed-sigma.root" ".root")
DIR_1224="/data8/majak/MLHEP/input-fd-10092024"
PTRN_1224=("${DIR_1224}/yields-fd_precise_1224_split_bkg_0.60_0.60_fd_" "${DIR_1224}/efficienciesLcpKpiRun3analysis_1224_split_bkg_0.60_0.60_fd_")
SUFFIX_1224=("-fixed-sigma.root" ".root")

OUTFILE_PTRN=("merged_yields_fdd_approvals_fd_" "merged_eff_fdd_approvals_fd_")

for k in "${!PTRN_12[@]}" ; do
  echo "k ${k}"
  echo "PTRN_12: ${PTRN_12}"
  echo "PTRN_12[k]: ${PTRN_12[k]}"
  echo "PTRN_212[k]: ${PTRN_212[k]}"
  echo "PTRN_1224[k]: ${PTRN_1224[k]}"

  for i in "${!FD_12[@]}" ; do
    INPUT_12=${PTRN_12[k]}${FD_12[i]}${SUFFIX_12[k]}
    INPUT_212=${PTRN_212[k]}${FD_12_OLD[i]}*[0-9][0-9]${SUFFIX_212[k]}

    # dummy loop to get shell expansion in INPUT_1224
    for f in ${PTRN_1224[k]}${FD_12_OLD[i]}*[0-9][0-9]${SUFFIX_1224[k]} ; do
      INPUT_1224=${f}
      suffix=${INPUT_1224[0]##${PTRN_1224[k]}}
      suffix=${suffix%%${SUFFIX_1224[k]}}
      OUTFILE=${OUTFILE_PTRN[k]}${suffix}.root

      echo "i ${i} k ${k}"
      echo "INPUT_12: ${INPUT_12}"
      echo "INPUT_212: " ${INPUT_212}
      echo "INPUT_1224: " ${INPUT_1224}
      echo "suffix: " ${suffix}
      echo "outfile: " ${OUTFILE}

      python merge_histos.py -o /data8/majak/crosssec/${OUTFILE} \
        -i ${INPUT_12} \
        -i ${INPUT_212} \
        -i ${INPUT_212} \
        -i ${INPUT_212} \
        -i ${INPUT_212} \
        -i ${INPUT_212} \
        -i ${INPUT_212} \
        -i ${INPUT_1224} \
        -i ${INPUT_1224}
    done
  done
done

# Merge yields and efficiencies for repeating September cut variation
#python merge_histos.py -o /data8/majak/crosssec/merged_yields_fdd_approvals_fd_0.21_0.20_0.26_0.17_0.10_0.15_0.08_0.17_0.09.root \
#  -i /data8/majak/MLHEP/input-fd-23082024/yields-bkg_0.20_0.60_fd_0.21-rebin-1-fixed-sigma.root
#  -i /data8/majak/MLHEP/input-fd-23082024/yields-fd_precise_rebin4_bkg_0.20_0.60_fd_0.21_0.20_0.26_0.17_0.10_0.15_0.08_0.08-fixed-sigma.root \
#  -i /data8/majak/MLHEP/input-fd-23082024/yields-fd_precise_rebin4_bkg_0.20_0.60_fd_0.21_0.20_0.26_0.17_0.10_0.15_0.08_0.08-fixed-sigma.root \
#  -i /data8/majak/MLHEP/input-fd-23082024/yields-fd_precise_rebin4_bkg_0.20_0.60_fd_0.21_0.20_0.26_0.17_0.10_0.15_0.08_0.08-fixed-sigma.root \
#  -i /data8/majak/MLHEP/input-fd-23082024/yields-fd_precise_rebin4_bkg_0.20_0.60_fd_0.21_0.20_0.26_0.17_0.10_0.15_0.08_0.08-fixed-sigma.root \
#  -i /data8/majak/MLHEP/input-fd-23082024/yields-fd_precise_rebin4_bkg_0.20_0.60_fd_0.21_0.20_0.26_0.17_0.10_0.15_0.08_0.08-fixed-sigma.root \
#  -i /data8/majak/MLHEP/input-fd-23082024/yields-fd_precise_rebin4_bkg_0.20_0.60_fd_0.21_0.20_0.26_0.17_0.10_0.15_0.08_0.08-fixed-sigma.root \
#  -i /data8/majak/MLHEP/input-fd-10092024/yields-fd_precise_1224_split_bkg_0.60_0.60_fd_0.21_0.20_0.26_0.17_0.10_0.15_0.08_0.17_0.09-fixed-sigma.root \
#  -i /data8/majak/MLHEP/input-fd-10092024/yields-fd_precise_1224_split_bkg_0.60_0.60_fd_0.21_0.20_0.26_0.17_0.10_0.15_0.08_0.17_0.09-fixed-sigma.root
