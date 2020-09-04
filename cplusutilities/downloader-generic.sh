#!/bin/bash

DOWNLDOUTPUTPATH=$1
SAVEDIR=$2
DOWNLOAD_FILENAME=$3



printf "Downloading LEGO train files from: %s to %s \n" $DOWNLDOUTPUTPATH $SAVEDIR


cmd=$(printf "cp -T 32 %s file:%s/\n" $DOWNLDOUTPUTPATH $SAVEDIR)

jalien << EOF
$cmd
exit
EOF

nameoutputlist=$(printf "%s/listfiles.txt" $SAVEDIR)
find $SAVEDIR -not -type d -name $DOWNLOAD_FILENAME> $nameoutputlist
if [ $? -ne 0 ]; then
  printf "\r                         \e[1;31mWarning: No files were downloaded. Did you enter JAliEn environment before? Are you connected to internet? Did you set the correct path?\e[0m" > /dev/tty
  printf "$SAVEDIR/printing-line-to-give-a-warning-as-no-files-were-downloaded/$DOWNLOAD_FILENAME" >> $nameoutputlist
else
  NDWNLFILES=$(wc -l < "$nameoutputlist")
  printf "\r                         \e[1;32mSuccessfully. %s files downloaded.\e[0m" $NDWNLFILES > /dev/tty
fi
