#!/bin/bash
#source clean.sh

doDsFromCand=0
doLcFromCand=0
doDsFromEvt=0
doBplusFromEvt=0
doPID=1

MCSAMPLE="../MLproductions/AnalysisResults_Ds_MC_2018Sep21_LHC18a4a2_cent_fast_CandBased.root"
DATASAMPLE="../MLproductions/AnalysisResults_Ds_Data_2018Sep21_LHC15o_pass1_pidfix_CandBased.root"  
MCSAMPLEOUT="../MLproductions/AnalysisResults_Ds_MC_2018Sep21_LHC18a4a2_cent_fast_CandBased_skimmed.root"
DATASAMPLEOUT="../MLproductions/AnalysisResults_Ds_Data_2018Sep21_LHC15o_pass1_pidfix_CandBased_skimmed.root"  
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



MCSAMPLE="../MLproductions/AnalysisResults_Lambdac_MC_CandBased.root"
DATASAMPLE="../MLproductions/AnalysisResults_Lambdac_Data_CandBased.root"
MCSAMPLEOUT="../MLproductions/AnalysisResults_Lambdac_MC_CandBased_skimmed.root"
DATASAMPLEOUT="../MLproductions/AnalysisResults_Lambdac_Data_CandBased_skimmed.root"

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


MCSAMPLE="../MLproductions/AnalysisResults_D0DplusDs_MC_EvtBased.root"
DATASAMPLE="../MLproductions/AnalysisResults_D0DplusDs_Data_LHC17p_FAST_run282343_EvtBased.root"
MCSAMPLEOUT="../MLproductions/AnalysisResults_D0DplusDs_MC_EvtBased_skimmed.root"
DATASAMPLEOUT="../MLproductions/AnalysisResults_D0DplusDs_Data_LHC17p_FAST_run282343_EvtBased_skimmed.root"
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

#-------

MCSAMPLE="../MLproductions/AnalysisResults_TreeForBplus_MC_EventBased.root"
DATASAMPLE="../MLproductions/AnalysisResults_TreeForBplus_MC_EventBased.root"
MCSAMPLEOUT="../MLproductions/AnalysisResults_TreeForBplus_MC_EventBased_skimmed.root"
DATASAMPLEOUT="../MLproductions/AnalysisResults_TreeForBplus_Data_EventBased_skimmed.root"
MCTree="tree_Bplus"
DataTree="tree_Bplus"

if [ $doBplusFromEvt -eq 1 ]
then

rm $MCSAMPLEOUT
rm $DATASAMPLEOUT

g++ skimTreeBplusFromEvt.C $(root-config --cflags --libs) -g -o skimTreeBplusFromEvt.exe 
./skimTreeBplusFromEvt.exe "$MCSAMPLE" "$MCSAMPLEOUT" "$MCTree" 
./skimTreeBplusFromEvt.exe "$DATASAMPLE" "$DATASAMPLEOUT" "$DataTree" 
rm -rf skimTreeBplusFromEvt.exe 
rm -rf skimTreeBplusFromEvt.exe.dSYM
fi

# -----------

MCSAMPLE="../MLproductions/AnalysisResults_TreeForPIDwithML_Dplus_CandBased.root"
MCSAMPLEOUT="../MLproductions/AnalysisResults_TreeForPIDwithML_Dplus_CandBased_skimmed.root"
MCTree="DplusPID/candTree"
totevents=500000

if [ $doPID -eq 1 ]
then

rm $MCSAMPLEOUT
rm $DATASAMPLEOUT

g++ skimTreePID.C $(root-config --cflags --libs) -g -o skimTreePID.exe 
./skimTreePID.exe "$MCSAMPLE" "$MCSAMPLEOUT" "$MCTree" "$totevents"
rm -rf skimTreePID.exe skimTreePID.exe.dSYM
fi
