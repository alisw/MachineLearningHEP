#!/bin/bash
#Arguments to this bash:
#   $1 is trainname (e.g. 297_20181120-2315_child_1)
#   $2 is path to place to save output
#   $3 is how many files to merge into one
#To set in script:
#   STAGE      ("" if all Lego train merging failed, otherwise /Stage_#/

#inputfile=listfilesMerging.txt
inputfile=$(printf "listfilesMerging_%s.txt" $1)

BASEDIR=$2
if [ -z "$BASEDIR" ]; then
BASEDIR=$(pwd)
fi
TRAINNAME=$1
STAGE="" #Stage_1

#nameoutput="../MLproductions/mergeSkimOutputDir_test"
nameoutput=$BASEDIR/$TRAINNAME/$STAGE/mergeSkimOutputDir
#nameoutputlist="lsoutputmergedlist.txt"
nameoutputlist=$(printf "lsOutputMergedList_%s.txt" $1)

nfilesformerging=$3
if [ -z "$nfilesformerging" ]; then
  nfilesformerging=4
fi

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

cp $nameoutputlist $nameoutput/
