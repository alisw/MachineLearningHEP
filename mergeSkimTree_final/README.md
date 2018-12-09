# Getting and processing TTreeHandler output

Instructions to download the output from the LEGO train, merge the files, and skim for specific mesons.

## 1) Download

The following script works out of the box on lxplus, using
```
/cvmfs/alice.cern.ch/bin/alienv enter JAliEn
jalien
#Enter Grid Certificate password
exit
./downloadOutputTrain.sh $TRAINNAME $PLACETOSAVEOUTPUT
```
where $TRAINNAME = 297_20181120-2315_child_1 (for example) and $PLACETOSAVEOUTPUT = ../MLproductions or can be omitted to save train output in current directory.

On a local system one should build AliPhysics enabling jalien
```
aliBuild build AliPhysics --defaults jalien-root6
```
and follow the same instructions as above (entering the alienv in the correct way).

Four train-specific variables have to be set in the script:
* OUTPUTPATH       (output of train)
* STAGE       ("" for no GRID merging, otherwise /Stage_#/)
* NFILES       (/*/ = download all files, /000*/ is 10 files, /00*/ is 100 files, etc)
* OUTPUTFILE       (name of file to download)

## 2) Merging

Exit jAliEn environment, and load normal AliPhysics. For lxplus one uses:
```
/cvmfs/alice.cern.ch/bin/alienv enter VO_ALICE@AliPhysics::vAN-20181208-1
```
Run the merging script
```
./mergefiles.sh $TRAINNAME $PLACETOSAVEOUTPUT $NFILESFORMERGING
```
where $TRAINNAME = 297_20181120-2315_child_1 (for example) and $PLACETOSAVEOUTPUT = ../MLproductions or can be omitted to save train output in current directory. $NFILESFORMERGING is the amount of files to be merged using hadd, with default value 4.

## 3) Skimming

Enable the mesons you want to skim in the macro, and run:
```
./submitjobs.sh $TRAINNAME
```
where $TRAINNAME = 297_20181120-2315_child_1 (for example)

## In case of problems:

For problems luuk.vermunt@cern.ch

TODO: Use .txt files for list-to-merge and list-to-skim from save directory instead of from this directory, as this will end up quite chaotic with multiple trains.
