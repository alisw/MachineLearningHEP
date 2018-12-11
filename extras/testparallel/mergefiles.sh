#!/bin/bash

nfilesformerging=3
inputfile=listfilesMerging.txt
nameoutput="outputmerged"
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



