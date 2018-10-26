#!/bin/bash
#source clean.sh

MCSAMPLE="$HOME/MLDsproductions/MC/2018Sep21_LHC18a4a2_cent_fast/AnalysisResults_001.root"
DATASAMPLE="$HOME/MLDsproductions/Data/2018Sep21_LHC15o_pass1_pidfix/AnalysisResults_000.root"  
DATASAMPLE_ROOT6="$HOME/MLDsproductions/Data/2018Sep21_LHC15o_pass1_pidfix/AnalysisResults_000_root6.root"  
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


MCTree="PWG3_D2H_InvMassDs_MB_Loose_consPID_MVA_kINT7/fTreeDs"
DataTree="PWG3_D2H_InvMassDs_010_PbPb_Loose_consPID_MVA_kINT7/fTreeDs"

source clean.sh
for neventspersample in 1000 10000
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
