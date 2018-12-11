#!/bin/bash
#source clean.sh

doPIDSingletrack=1

DATASAMPLE="../MLproductions/AnalysisResults_PID_Data.root"
DATASAMPLEOUT="../MLproductions/AnalysisResults_PID_Data_skimmed.root"
MCSAMPLE="../MLproductions/AnalysisResults_PID_Tree_MC_LHC17f6.root"
MCSAMPLEOUT="../MLproductions/AnalysisResults_PID_Tree_MC_LHC17f6_skimmed.root"
MCTree="Trees/TrackTree"
DATATree="Trees/TrackTree"
totevents=500000

if [ $doPIDSingletrack -eq 1 ]
then

rm $MCSAMPLEOUT
rm $DATASAMPLEOUT

g++ skimTreeSingleTrackPID.C $(root-config --cflags --libs) -g -o skimTreeSingleTrackPID.exe 
./skimTreeSingleTrackPID.exe "$DATASAMPLE" "$DATASAMPLEOUT" "$DATATree" "$totevents"
./skimTreeSingleTrackPID.exe "$MCSAMPLE" "$MCSAMPLEOUT" "$MCTree" "$totevents"
rm -rf skimTreeSingleTrackPID.exe skimTreeSingleTrackPID.exe.dSYM
fi
