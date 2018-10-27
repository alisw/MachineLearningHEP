#!/bin/bash
#source clean.sh

MCSAMPLE="$HOME/MLproductions/MLDmesonsproductionsEventBased/LHC18a4a2_fast_run282343/AnalysisResultsDmesonsMC.root"
MCSAMPLEOUT="$HOME/MLproductions/MLDmesonsproductionsEventBased/LHC18a4a2_fast_run282343/AnalysisResultsDmesonsMC_CandBased.root"
DATASAMPLE="$HOME/MLproductions/MLDmesonsproductionsEventBased/LHC17p_FAST_run282343/AnalysisResultsData.root"  
DATASAMPLEOUT="$HOME/MLproductions/MLDmesonsproductionsEventBased/LHC17p_FAST_run282343/AnalysisResultsData_CandBased.root"  

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

