#!/bin/bash
# Our custom function
print_something () {
echo Hello $1
}

DataTree="PWG3_D2H_InvMassDs_010_PbPb_Loose_consPID_MVA_kINT7/fTreeDs"
totevents=50000000

START=$(date +%s)

#!/bin/bash
myfile=listfiles.txt
while IFS='' read -r line || [[ -n "$line" ]]; do
g++ ../buildtree/skimTreeDs.C $(root-config --cflags --libs) -g -o skimTreeDs.exe 
./skimTreeDs.exe "${line}.root" "${line}_skimmed.root" "$DataTree" "$totevents"
done < "$myfile"

## would wait until those are completed
## before displaying all done message
wait
echo "All done"
rm -rf skimTreeDs.exe skimTreeDs.exe.dSYM

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "It took $DIFF seconds"
