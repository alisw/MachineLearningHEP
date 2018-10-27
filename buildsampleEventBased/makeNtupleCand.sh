#!/bin/bash
#source clean.sh

MCSAMPLE="$HOME/MLproductions/MLDmesonsproductionsEventBased/LHC18a4a2_fast_run282343/AnalysisResultsDmesonsMC.root"
MCSAMPLEOUT="rootfiles/LHC18a4a2_fast_run282343_AnalysisResultsDmesonsMC_CandBased.root"
DATASAMPLE="$HOME/MLproductions/MLDmesonsproductionsEventBased/LHC17p_FAST_run282343/AnalysisResultsData.root"  
DATASAMPLEOUT="rootfiles/LHC17p_FAST_run282343_AnalysisResultsData_CandBased.root"  

rm $MCSAMPLEOUT
rm $DATASAMPLEOUT

if [ ! -f $MCSAMPLE ] || [ ! -f $DATASAMPLE ]; then
  echo "******************** ATTENTION ********************"
  echo "You need to download the files"
  echo "******************** THIS IS GOING TO FAIL *******************"
  exit
fi

TreeName="tree_Ds"

g++ makeNtupleCandBased.C $(root-config --cflags --libs) -g -o makeNtupleCandBased.exe 
./makeNtupleCandBased.exe "$MCSAMPLE" "$MCSAMPLEOUT" "$TreeName" 
./makeNtupleCandBased.exe "$DATASAMPLE" "$DATASAMPLEOUT" "$TreeName" 
rm -rf makeNtupleCandBased.exe makeNtupleCandBased.exe.dSYM

