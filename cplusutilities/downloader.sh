#!/bin/bash

DWNLDOUTPUTPATH=$1
CHILD=$2
NFILES=$3
DWNLDOUTPUTFILE=$4
BASEDIR=$5
TRAINNAME=$6
DATASETWITHCHILDS=$7
LOCALCHILD=$8
STAGE=$9

SAVEDIR=$(printf "%s/%s/unmerged/child_%s" $BASEDIR $TRAINNAME $LOCALCHILD)
mkdir -p -m 777 $SAVEDIR
if [ $? -ne 0 ]; then
  printf "Error: Could not create output directory. Is $SAVEDIR writable? Returning... \n\n"
  exit
else
  printf "Created directory: $SAVEDIR \n"
fi

if [ -z "$9" ]; then
  #do nothing, if-statement to be reversed
  dummy=1
else
  SAVEDIR=$(printf "%s/%s/unmerged/child_%s/%s/" $BASEDIR $TRAINNAME $LOCALCHILD $STAGE)
  mkdir -p -m 777 $SAVEDIR
  if [ $? -ne 0 ]; then
    printf "Error: Could not create output directory. Is $SAVEDIR writable? Returning... \n\n"
    exit
  else
    printf "Created directory: $SAVEDIR \n"
  fi
fi

if [ $DATASETWITHCHILDS -eq 1 ]; then
  DWNLDOUTPUTPATH=$(printf "%s/%s_child_%s/%s" $DWNLDOUTPUTPATH $TRAINNAME $CHILD $STAGE)
else
  DWNLDOUTPUTPATH=$(printf "%s/%s/%s" $DWNLDOUTPUTPATH $TRAINNAME $STAGE)
fi
printf "Downloading LEGO train files from: %s\n" $DWNLDOUTPUTPATH

cmd=$(printf "cp -T 32 %s/%s/%s.root file:%s/\n" $DWNLDOUTPUTPATH "$NFILES" $DWNLDOUTPUTFILE $SAVEDIR)

/opt/jalien/src/jalien/jalien << EOF
$cmd
exit
EOF

nameoutputlist=$(printf "listfiles_%s_child_%s%s.txt" $TRAINNAME $LOCALCHILD $STAGE)
find $SAVEDIR/$NFILES/$DWNLDOUTPUTFILE.root -maxdepth 1 -not -type d> $nameoutputlist
if [ $? -ne 0 ]; then
  printf "\r                         \e[1;31mWarning: No files were downloaded. Did you enter JAliEn environment before? Are you connected to internet? Did you set the correct path?\e[0m" > /dev/tty
  printf "$SAVEDIR/printing-line-to-give-a-warning-as-no-files-were-downloaded/$DWNLDOUTPUTFILE.root" >> $nameoutputlist
else
  NDWNLFILES=$(wc -l < "$nameoutputlist")
  printf "\r                         \e[1;32mSuccessfully. %s files downloaded.\e[0m" $NDWNLFILES > /dev/tty
fi

mv $nameoutputlist $SAVEDIR
