#!/bin/bash
#source clean.sh

doDsFromCand=0
doLcFromCand=0
doDsFromEvt=1

MCSAMPLE="$HOME/MLproductions/AnalysisResults_Ds_MC_2018Sep21_LHC18a4a2_cent_fast_CandBased.root"
DATASAMPLE="$HOME/MLproductions/AnalysisResults_Ds_Data_2018Sep21_LHC15o_pass1_pidfix_CandBased.root"  
MCSAMPLEOUT="$HOME/MLproductions/AnalysisResults_Ds_MC_2018Sep21_LHC18a4a2_cent_fast_CandBased_skimmed.root"
DATASAMPLEOUT="$HOME/MLproductions/AnalysisResults_Ds_Data_2018Sep21_LHC15o_pass1_pidfix_CandBased_skimmed.root"  
totevents=500000
MCTree="PWG3_D2H_InvMassDs_MB_Loose_consPID_MVA_kINT7/fTreeDs"
DataTree="PWG3_D2H_InvMassDs_010_PbPb_Loose_consPID_MVA_kINT7/fTreeDs"


if [ $doDsFromCand -eq 1 ]
then

rm $MCSAMPLEOUT
rm $DATASAMPLEOUT

g++ skimTreeDs.C $(root-config --cflags --libs) -g -o skimTreeDs.exe 
./skimTreeDs.exe "$MCSAMPLE" "$MCSAMPLEOUT" "$MCTree" "$totevents" 
./skimTreeDs.exe "$DATASAMPLE" "$DATASAMPLEOUT" "$DataTree" "$totevents"
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

if [ $doLcFromCand -eq 1 ]
then

rm $MCSAMPLEOUT
rm $DATASAMPLEOUT

g++ skimTreeLambdac.C $(root-config --cflags --libs) -g -o skimTreeLambdac.exe 
./skimTreeLambdac.exe "$MCSAMPLE" "$MCSAMPLEOUT" "$MCTree" "$totevents" 
./skimTreeLambdac.exe "$DATASAMPLE" "$DATASAMPLEOUT" "$DataTree" "$totevents"
rm skimTreeLambdac.exe
rm -rf skimTreeLambdac.exe.dSYM
fi


MCSAMPLE="$HOME/MLproductions/AnalysisResults_D0DplusDs_MC_EvtBased.root"
DATASAMPLE="$HOME/MLproductions/AnalysisResults_D0DplusDs_Data_LHC17p_FAST_run282343_EvtBased.root"
MCSAMPLEOUT="$HOME/MLproductions/AnalysisResults_D0DplusDs_MC_EvtBased_skimmed.root"
DATASAMPLEOUT="$HOME/MLproductions/AnalysisResults_D0DplusDs_Data_LHC17p_FAST_run282343_EvtBased_skimmed.root"
MCTree="tree_Ds"
DataTree="tree_Ds"


if [ $doDsFromEvt -eq 1 ]
then

rm $MCSAMPLEOUT
rm $DATASAMPLEOUT

g++ skimTreeDsFromEvt.C $(root-config --cflags --libs) -g -o skimTreeDsFromEvt.exe 
./skimTreeDsFromEvt.exe "$MCSAMPLE" "$MCSAMPLEOUT" "$MCTree" 
./skimTreeDsFromEvt.exe "$DATASAMPLE" "$DATASAMPLEOUT" "$DataTree" 
rm -rf skimTreeDsFromEvt.exe skimTreeDsFromEvt.exe.dSYM
fi
