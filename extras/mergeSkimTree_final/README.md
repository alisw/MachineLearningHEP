# Getting and processing TTreeHandler output

Instructions to download the output from the LEGO train, merge the files, and skim for specific mesons.

## 1) Download

The following script works out of the box on lxplus, using
```
/cvmfs/alice.cern.ch/bin/alienv enter JAliEn
jalien
#Enter Grid Certificate password
exit
./downloadOutputTrain.sh $TRAINNAME $PLACETOSAVEOUTPUT $STAGE
```
where $TRAINNAME = 297_20181120-2315_child_1 (for example) and $PLACETOSAVEOUTPUT = ../MLproductions or can be omitted to save train output in current directory. $STAGE = "Stage_#" or "" if all GRID merging failed.

On a local system one should build AliPhysics enabling jalien
```
aliBuild build AliPhysics --defaults jalien-root6
```
and follow the same instructions as above (entering the alienv in the correct way).

Four train-specific variables have to be set in the script:
* OUTPUTPATH       (output of train)
* NFILES       (/&#42;/ = download all files, /000&#42;/ is 10 files, /00&#42;/ is 100 files, etc)
* OUTPUTFILE       (name of file to download)

## 2) Merging

Exit jAliEn environment, and load normal AliPhysics. For lxplus one uses:
```
/cvmfs/alice.cern.ch/bin/alienv enter VO_ALICE@AliPhysics::vAN-20181208-1
```
Run the merging script
```
./mergefiles.sh $TRAINNAME $PLACETOSAVEOUTPUT $STAGE $NFILESFORMERGING
```
where $TRAINNAME = 297_20181120-2315_child_1 (for example) and $PLACETOSAVEOUTPUT = ../MLproductions or can be omitted to save train output in current directory. $STAGE = "Stage_#" or "" if all GRID merging failed, and $NFILESFORMERGING is the amount of files to be merged using hadd, with default value 4.

## 3) Skimming

Enable the mesons you want to skim in the macro, and run:
```
./submitjobs.sh $path-to/lsOutputMergedList_$TRAINNAME$STAGE.txt
```
where the mergefiles.sh saved the lsOutputMergedList_$TRAINNAME$STAGE.txt file. If no merging was applied, one has to tweak a bit the output of the downloading stage.

## In case of problems:

For problems luuk.vermunt@cern.ch
