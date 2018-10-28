#!/bin/bash
#source clean.sh

doDsFromCand=1
doLcFromCand=0
doDsFromEvt=0

MCSAMPLE="$HOME/MLproductions/AnalysisResults_Ds_MC_2018Sep21_LHC18a4a2_cent_fast_CandBased.root"
DATASAMPLE="$HOME/MLproductions/AnalysisResults_Ds_Data_2018Sep21_LHC15o_pass1_pidfix_CandBased.root"  
MCSAMPLEOUT="$HOME/MLproductions/AnalysisResults_Ds_MC_2018Sep21_LHC18a4a2_cent_fast_CandBased_skimmed.root"
DATASAMPLEOUT="$HOME/MLproductions/AnalysisResults_Ds_Data_2018Sep21_LHC15o_pass1_pidfix_CandBased_skimmed.root"  
totevents=500000
MCTree="PWG3_D2H_InvMassDs_MB_Loose_consPID_MVA_kINT7/fTreeDs"
DataTree="PWG3_D2H_InvMassDs_010_PbPb_Loose_consPID_MVA_kINT7/fTreeDs"

rm $MCSAMPLEOUT
rm $DATASAMPLEOUT

if [ $doDsFromCand -eq 1 ]
then

g++ skimTreeDs.C $(root-config --cflags --libs) -g -o skimTreeDs.exe 
./skimTreeDs.exe "$MCSAMPLE" "$MCSAMPLEOUT" "$MCTree" "$toteventsdata" 
./skimTreeDs.exe "$DATASAMPLE" "$DATASAMPLEOUT" "$DataTree" "$toteventsdata"
rm skimTreeDs.exe
rm -rf skimTreeDs.exe.dSYM

fi



MCSAMPLE="$HOME/MLproductions/AnalysisResults_Lambdac_MC_CandBased.root"
DATASAMPLE="$HOME/MLproductions/AnalysisResults_Lambdac_Data_CandBased.root"
MCSAMPLEOUT="$HOME/MLproductions/AnalysisResults_Lambdac_MC_CandBased_skimmed.root"
DATASAMPLEOUT="$HOME/MLproductions/AnalysisResults_Lambdac_Data_CandBased_skimmed.root"

totevents=500000
MCTree="fNtupleLambdac5TeVprod_Proc"
DataTree="fNtupleLambdacProdcuts_Proc"

rm $MCSAMPLEOUT
rm $DATASAMPLEOUT

if [ $doLcFromCand -eq 1 ]
then

g++ skimTreeLambdac.C $(root-config --cflags --libs) -g -o skimTreeLambdac.exe 
./skimTreeLambdac.exe "$MCSAMPLE" "$MCSAMPLEOUT" "$MCTree" "$toteventsdata" 
./skimTreeLambdac.exe "$DATASAMPLE" "$DATASAMPLEOUT" "$DataTree" "$toteventsdata"
rm skimTreeLambdac.exe
rm -rf skimTreeLambdac.exe.dSYM
fi


MCSAMPLE="$HOME/MLproductions/AnalysisResults_D0DplusDs_MC_EvtBased.root"
DATASAMPLE="$HOME/MLproductions/AnalysisResults_D0DplusDs_Data_LHC17p_FAST_run282343_EvtBased.root"
MCSAMPLEOUT="$HOME/MLproductions/AnalysisResults_D0DplusDs_MC_EvtBased_skimmed.root"
DATASAMPLEOUT="$HOME/MLproductions/AnalysisResults_D0DplusDs_Data_LHC17p_FAST_run282343_EvtBased_skimmed.root"

rm $MCSAMPLEOUT
rm $DATASAMPLEOUT

if [ $doDsFromEvt -eq 1 ]
then

g++ skimTreeDsFromEvt.C $(root-config --cflags --libs) -g -o skimTreeDsFromEvt.exe 
./skimTreeDsFromEvt.exe "$MCSAMPLE" "$MCSAMPLEOUT" "$TreeName" 
./skimTreeDsFromEvt.exe "$DATASAMPLE" "$DATASAMPLEOUT" "$TreeName" 
rm -rf skimTreeDsFromEvt.exe skimTreeDsFromEvt.exe.dSYM
fi
