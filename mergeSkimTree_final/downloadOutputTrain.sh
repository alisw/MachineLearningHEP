#Before running one should have entered jAliEn environment
#On lxplus (on local system one has to build:  aliBuild build AliPhysics --defaults jalien-root6):
#    /cvmfs/alice.cern.ch/bin/alienv enter JAliEn  (or: alienv enter AliPhysics/latest-jalien-root6)
#    jalien
#    Enter Grid certificate password
#    exit
#    ./downloadOutputTrain.sh $TRAINNAME $PLACETOSAVEOUTPUT
#
#Arguments to this bash:
#   $1 is trainname (e.g. 297_20181120-2315_child_1)
#   $2 is path to place to save output
#To set in script:
#   OUTPUTPATH (output of train)
#   STAGE      ("" if all Lego train merging failed, otherwise /Stage_#/
#   NFILES     (/*/ = download all files, /000*/ is 10 files, /00*/ is 100 files, etc)
#   OUTPUTFILE (name of file to download)

#set -x #echo on

OUTPUTPATH=/alice/data/2017/LHC17p/000282341/pass1_FAST/PWGZZ/Devel_2
STAGE="" #Stage_1
NFILES="000*" #"*" "0* "00*"
OUTPUTFILE=AnalysisResults

TRAINNAME=$1
OUTPUTPATH=$OUTPUTPATH/$TRAINNAME/$STAGE
mkdir $TRAINNAME
mkdir $TRAINNAME$STAGE
printf "cd %s\n" $TRAINNAME$STAGE
BASEDIR=$2
if [ -z "$BASEDIR" ]; then
  BASEDIR=$(pwd)
fi

cmd=$(printf "cp -T 32 %s/%s/%s.root file:%s/%s/%s/\n" $OUTPUTPATH "$NFILES" $OUTPUTFILE $BASEDIR $TRAINNAME $STAGE)

jalien << EOF
$cmd
exit
EOF

#nameoutputlist=$(printf "listfilesMerging_%s%s.txt" $TRAINNAME $STAGE)
nameoutputlist=$(printf "listfilesMerging_%s.txt" $TRAINNAME)
find $BASEDIR/$TRAINNAME/$STAGE/*/ -maxdepth 1 -not -type d> $nameoutputlist

cp $nameoutputlist $BASEDIR/$TRAINNAME/$STAGE/

