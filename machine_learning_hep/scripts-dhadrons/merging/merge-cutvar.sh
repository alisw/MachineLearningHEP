#!/bin/bash

python merge_histos.py -o /data8/majak/systematics/032025/bdt/CutVarLc_pp13TeV_LHC23_pass4_wide_both.root \
  -n hCorrYieldsPrompt -n hCorrYieldsNonPrompt -n hCorrFracPrompt -n hCorrFracNonPrompt \
  -n hCovPromptPrompt -n hCovPromptNonPrompt -n hCovNonPromptNonPrompt \
  -i /data8/majak/fdd-results/012025/fdd-results-1-2-wide-both/CutVarLc_pp13TeV_LHC23_pass4.root \
  -i /data8/majak/fdd-results/012025/fdd-results-2-3-cheb-wide-both/CutVarLc_pp13TeV_LHC23_pass4.root \
  -i /data8/majak/fdd-results/012025/fdd-results-3-4-cheb-wide-both/CutVarLc_pp13TeV_LHC23_pass4.root \
  -i /data8/majak/fdd-results/012025/fdd-results-4-5-cheb-wide-both/CutVarLc_pp13TeV_LHC23_pass4.root \
  -i /data8/majak/fdd-results/012025/fdd-results-5-6-cheb-wide-both/CutVarLc_pp13TeV_LHC23_pass4.root \
  -i /data8/majak/fdd-results/012025/fdd-results-4-5-cheb-wide-both/CutVarLc_pp13TeV_LHC23_pass4.root \
  -i /data8/majak/fdd-results/012025/fdd-results-4-5-cheb-wide-both/CutVarLc_pp13TeV_LHC23_pass4.root \
  -i /data8/majak/fdd-results/012025/fdd-results-4-5-cheb-wide-both/CutVarLc_pp13TeV_LHC23_pass4.root \
  -i /data8/majak/fdd-results/012025/fdd-results-10-12-cheb-wide-both/CutVarLc_pp13TeV_LHC23_pass4.root \
  -i /data8/majak/fdd-results/012025/fdd-results-10-12-cheb-wide-both/CutVarLc_pp13TeV_LHC23_pass4.root \
  -i /data8/majak/fdd-results/012025/fdd-results-16-24-cheb-wide-both/CutVarLc_pp13TeV_LHC23_pass4.root
