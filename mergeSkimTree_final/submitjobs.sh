#!/bin/bash
#Arguments to this bash:
#   $1 is path to lsOutputMergedList_$TRAINNAME$STAGE.txt

START=$(date +%s)

isMC=1
doDplusFromEvt=1
doDsFromEvt=1
doDzeroFromEvt=1
doBplusFromEvt=0 #Classes not yet ready + not tested
doLcFromEvt=0 #Classes not yet ready + not tested
#doPID=0 #to be added

myfile=$1

if [ $doDplusFromEvt -eq 1 ]
then

DataTree="tree_Dplus"

while IFS='' read -r line || [[ -n "$line" ]]; do
g++ includeSkim/skimTreeDplusFromEvt.C $(root-config --cflags --libs) -g -o skimTreeDplusFromEvt.exe
./skimTreeDplusFromEvt.exe "${line}.root" "${line}_Dplus_skimmed.root" "$DataTree" "$isMC"
done < "$myfile"

## would wait until those are completed
## before displaying all done message
wait
echo "Dplus = All done"
rm -rf skimTreeDplusFromEvt.exe skimTreeDplusFromEvt.exe.dSYM

fi



if [ $doDsFromEvt -eq 1 ]
then

DataTree="tree_Ds"

while IFS='' read -r line || [[ -n "$line" ]]; do
g++ includeSkim/skimTreeDsFromEvt.C $(root-config --cflags --libs) -g -o skimTreeDsFromEvt.exe
./skimTreeDsFromEvt.exe "${line}.root" "${line}_Ds_skimmed.root" "$DataTree" "$isMC"
done < "$myfile"

## would wait until those are completed
## before displaying all done message
wait
echo "Ds = All done"
rm -rf skimTreeDsFromEvt.exe skimTreeDsFromEvt.exe.dSYM

fi



if [ $doDzeroFromEvt -eq 1 ]
then

DataTree="tree_D0"

while IFS='' read -r line || [[ -n "$line" ]]; do
g++ includeSkim/skimTreeDzeroFromEvt.C $(root-config --cflags --libs) -g -o skimTreeDzeroFromEvt.exe
./skimTreeDzeroFromEvt.exe "${line}.root" "${line}_Dzero_skimmed.root" "$DataTree" "$isMC"
done < "$myfile"

## would wait until those are completed
## before displaying all done message
wait
echo "Dzero = All done"
rm -rf skimTreeDzeroFromEvt.exe skimTreeDzeroFromEvt.exe.dSYM

fi



if [ $doBplusFromEvt -eq 1 ]
then

DataTree="tree_Bplus"

while IFS='' read -r line || [[ -n "$line" ]]; do
g++ includeSkim/skimTreeBplusFromEvt.C $(root-config --cflags --libs) -g -o skimTreeBplusFromEvt.exe
./skimTreeBplusFromEvt.exe "${line}.root" "${line}_Bplus_skimmed.root" "$DataTree"
done < "$myfile"

## would wait until those are completed
## before displaying all done message
wait
echo "Bplus = All done"
rm -rf skimTreeBplusFromEvt.exe skimTreeBplusFromEvt.exe.dSYM

fi



if [ $doLcFromEvt -eq 1 ]
then

DataTree="tree_Lc"

while IFS='' read -r line || [[ -n "$line" ]]; do
g++ includeSkim/skimTreeLcFromEvt.C $(root-config --cflags --libs) -g -o skimTreeLcFromEvt.exe
./skimTreeLcFromEvt.exe "${line}.root" "${line}_Lc_skimmed.root" "$DataTree"
done < "$myfile"

## would wait until those are completed
## before displaying all done message
wait
echo "Lc = All done"
rm -rf skimTreeLcFromEvt.exe skimTreeLcFromEvt.exe.dSYM

fi


END=$(date +%s)
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds"
