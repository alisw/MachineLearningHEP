#!/bin/bash
#Arguments to this bash:
#   $1 is trainname (e.g. 297_20181120-2315_child_1)
#   $2 is path to place to save output (e.g. "" or ../MLproductions/)
#   $3 is GRID merging Stage_X (e.g. "" or Stage_1)
#   $4 is how many files to merge into one

BASEDIR=$2
if [ -z "$BASEDIR" ]; then
  BASEDIR=$(pwd)
fi
TRAINNAME=$1
STAGE=$3

inputfile=$(printf "%s/%s/%s/listfilesMerging_%s%s.txt" $BASEDIR $TRAINNAME $STAGE $TRAINNAME $STAGE)
if [ -z "$STAGE" ]; then
  inputfile=$(printf "%s/%s/listfilesMerging_%s.txt" $BASEDIR $TRAINNAME $TRAINNAME)
fi
echo "Reading $inputfile for files to merge\n"

nfilesformerging=$4
if [ -z "$nfilesformerging" ]; then
  nfilesformerging=4
fi
echo "Merging with $nfilesformerging inputfiles\n"

nameoutput=$BASEDIR/$TRAINNAME/$STAGE/mergeSkimOutputDir_$nfilesformerging
echo "Saving merged output in directory: $nameoutput\n"
nameoutputlist=$(printf "lsOutputMergedList_%s%s.txt" $TRAINNAME $STAGE)
echo "Writing merged output in: $nameoutputlist\n"

rm -rf $nameoutput
mkdir $nameoutput
split -l $nfilesformerging $inputfile $nameoutput/split-file
ls $nameoutput/split-file*> $nameoutputlist

while IFS='' read -r line || [[ -n "$line" ]]; do
echo $line
#sed 's/$/.root /g' "${line}" > "${line}_rootflag"
#mv "${line}_rootflag" "${line}"
hadd "${line}.root" @"$line"
done < "$nameoutputlist"

mv $nameoutputlist $nameoutput/
