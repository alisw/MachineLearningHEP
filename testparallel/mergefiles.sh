#!/bin/bash

myfile=listfilesMC.txt
nameoutput="outputmerged"
nameoutputlist="lsoutputmergedlist.txt"

rm -rf $nameoutput
mkdir $nameoutput
split -l 3 listfilesMerging.txt $nameoutput/split-file
ls $nameoutput/split-file*> $nameoutputlist


while IFS='' read -r line || [[ -n "$line" ]]; do
echo $line
sed 's/$/.root /g' "${line}" > "${line}_rootflag"
mv "${line}_rootflag" "${line}"
hadd "${line}.root" @"$line"b
done < "$nameoutputlist"



