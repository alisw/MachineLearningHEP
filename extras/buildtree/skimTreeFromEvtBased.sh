#!/bin/bash
#source clean.sh

MCSAMPLE="$HOME/MLproductions/MLDmesonsproductionsEventBased/LHC18a4a2_fast_run282343/AnalysisResultsDmesonsMC.root"
MCSAMPLEOUT="$HOME/MLproductions/MLDmesonsproductionsEventBased/LHC18a4a2_fast_run282343/AnalysisResultsDmesonsMC_skimmed.root"
DATASAMPLE="$HOME/MLproductions/MLDmesonsproductionsEventBased/LHC17p_FAST_run282343/AnalysisResultsData.root"  
DATASAMPLEOUT="$HOME/MLproductions/MLDmesonsproductionsEventBased/LHC17p_FAST_run282343/AnalysisResultsData_skimmed.root"

rm $MCSAMPLEOUT
rm $DATASAMPLEOUT

if [ ! -f $MCSAMPLE ] || [ ! -f $DATASAMPLE ]; then
  echo "******************** ATTENTION ********************"
  echo "You need to download the files"
  echo "******************** THIS IS GOING TO FAIL *******************"
  exit
fi

TreeName="tree_Ds"

g++ skimTreeDsFromEvt.C $(root-config --cflags --libs) -g -o skimTreeDsFromEvt.exe 
./skimTreeDsFromEvt.exe "$MCSAMPLE" "$MCSAMPLEOUT" "$TreeName" 
./skimTreeDsFromEvt.exe "$DATASAMPLE" "$DATASAMPLEOUT" "$TreeName" 
rm -rf skimTreeDsFromEvt.exe skimTreeDsFromEvt.exe.dSYM

