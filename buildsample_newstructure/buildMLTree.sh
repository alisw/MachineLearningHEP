#!/bin/bash
#source clean.sh

MCSAMPLE="$HOME/MLDsproductions/trainD0D2OneRun/MC/891_20181008-1808/0001/AnalysisResults.root"
DATASAMPLE="$HOME/MLDsproductions/trainD0D2OneRun/data/1300_20181009-1047/0001/AnalysisResults.root"  
DATASAMPLE_ROOT6="$HOME/MLDsproductions/trainD0D2OneRun/data/AnalysisResults_ROOT6.root"  
toteventsdata=1000000

if [ ! -f $MCSAMPLE ] || [ ! -f $DATASAMPLE ]; then
  echo "******************** ATTENTION ********************"
  echo "You need to download the files"
  echo "Please download the folder MLDsproductions from ginnocen@lxplus.cern.ch:/afs/cern.ch/work/g/ginnocen/public/MLDsproductions"
  echo "And place it in your home directory"
  echo "If you want to use a different path you will have to change the MCSAMPLE and DATASAMPLE above accordingly "
  echo "******************** THIS IS GOING TO FAIL *******************"
  exit
fi


MCTree="tree_Ds"
DataTree="tree_Ds"

source clean.sh
for neventspersample in 1000 5000 10000 50000 100000
do
g++ buildMLTree.C $(root-config --cflags --libs) -g -o buildMLTree.exe 
./buildMLTree.exe "$MCSAMPLE"  "$MCTree" "$neventspersample" "$DATASAMPLE"  "$DataTree" "$neventspersample"
python preparesample.py "$neventspersample"
done

rm buildMLTree.exe
rm -rf buildMLTree.exe.dSYM

g++ convertTreeROOT6Loop.C $(root-config --cflags --libs) -g -o convertTreeROOT6Loop.exe 
./convertTreeROOT6Loop.exe "$DATASAMPLE"  "$DataTree" "$DATASAMPLE_ROOT6" "$toteventsdata"

rm convertTreeROOT6Loop.exe
rm -rf convertTreeROOT6Loop.exe.dSYM
