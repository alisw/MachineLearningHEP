#!/bin/bash

#Needs as input "listfilesMerging.txt", where the paths of all files to be merged should be stated. TODO: Create a script that creates this file.
#Set variables below as arguments of the script? TODO
nfilesformerging=4
inputfile=listfilesMerging.txt
nameoutput="../MLproductions/mergeSkimOutputDir_test"
nameoutputlist="lsoutputmergedlist.txt"

rm -rf $nameoutput
mkdir $nameoutput
split -l $nfilesformerging $inputfile $nameoutput/split-file
ls $nameoutput/split-file*> $nameoutputlist


while IFS='' read -r line || [[ -n "$line" ]]; do
echo $line
sed 's/$/.root /g' "${line}" > "${line}_rootflag"
mv "${line}_rootflag" "${line}"
hadd "${line}.root" @"$line"
done < "$nameoutputlist"

cp $nameoutputlist $nameoutput/
