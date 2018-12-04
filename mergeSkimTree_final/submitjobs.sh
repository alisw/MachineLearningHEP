#!/bin/bash
START=$(date +%s)

#TODO: add if-statement to switch between different mesons

DataTree="tree_Dplus"

myfile=lsoutputmergedlist.txt
while IFS='' read -r line || [[ -n "$line" ]]; do
g++ ../buildtree/skimTreeDplusFromEvt.C $(root-config --cflags --libs) -g -o skimTreeDplusFromEvt.exe
./skimTreeDplusFromEvt.exe "${line}.root" "${line}_skimmed.root" "$DataTree" #"$totevents"
done < "$myfile"

## would wait until those are completed
## before displaying all done message
wait
echo "All done"
rm -rf skimTreeDplusFromEvt.exe skimTreeDplusFromEvt.exe.dSYM

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds"
