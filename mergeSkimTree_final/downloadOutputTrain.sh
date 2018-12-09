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
#   $3 is GRID merging Stage_X (e.g. "" or Stage_1)
#To set in script:
#   OUTPUTPATH (output directory of train)
#   NFILES     (/*/ = download all files, /000*/ is 10 files, /00*/ is 100 files, etc)
#   OUTPUTFILE (name of file to download)

#set -x #echo on

TRAINNAME=$1
BASEDIR=$2
if [ -z "$BASEDIR" ]; then
  BASEDIR=$(pwd)
fi
STAGE=$3
mkdir $BASEDIR/$TRAINNAME
mkdir $BASEDIR/$TRAINNAME/$STAGE

OUTPUTPATH=/alice/data/2017/LHC17p/000282341/pass1_FAST/PWGZZ/Devel_2
NFILES="000*" #"*" "0* "00*" #Assumed 1000 < jobs < 9999, if different, change number of zeros
OUTPUTFILE=AnalysisResults

OUTPUTPATH=$OUTPUTPATH/$TRAINNAME/$STAGE
echo "Downloading LEGO train files from: $OUTPUTPATH\n"

cmd=$(printf "cp -T 32 %s/%s/%s.root file:%s/%s/%s/\n" $OUTPUTPATH "$NFILES" $OUTPUTFILE $BASEDIR $TRAINNAME $STAGE)

jalien << EOF
$cmd
exit
EOF

nameoutputlist=$(printf "listfilesMerging_%s%s.txt" $TRAINNAME $STAGE)
find $BASEDIR/$TRAINNAME/$STAGE/*/ -maxdepth 1 -not -type d> $nameoutputlist

mv $nameoutputlist $BASEDIR/$TRAINNAME/$STAGE/

