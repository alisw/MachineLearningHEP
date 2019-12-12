#!/bin/bash
#IMPORTANT: Before running one should have entered jAliEn environment:
#    jalien
#    <Enter Grid certificate password>
#    exit
#
#Arguments to this bash:
#   $1 is trainname (e.g. 297_20181120-2315)
#   $2 is path to place to save output (e.g. "" or ../ALICEanalysis/MLproductions/)
#   $3 is GRID merging Stage_X (e.g. "" for no merging, or Stage_1)
#
#To set in script (find with "#toset"):
#   NFILES     (/*/ = download all files, /000*/ is 10 files, /00*/ is 100 files, etc)
#   OUTPUTFILE (name of file to download)

printf "\n\n\n\e[1m----RUNNING THE DOWNLOADER----\e[0m\n\n"



#----THINGS TO SET----#
nfiles="/*/" #toset   For testing: "0*", "00*", or "000*" (Assuming 1000 < jobs < 9999)
outputfile="AnalysisResults" #toset
stage=""

#Confirm with user if hardcoded values are what he/she wants
printf "\e[1mYou set the following setters in the script. Please check them carefully before continuing.\e[0m\n"
printf "   Number of files to download from grid: \e[1m$nfiles\e[0m\n"
printf "   Outputfile to be downloaded from grid: \e[1m$outputfile.root\e[0m\n"
printf "   I will download \e[1mnon-merged files\e[0m from GRID\n"
printf "   Did you authenticate to \e[1mJAliEn\e[0m before running this script?\n"

printf "\n\e[1m   Are you okay with these settings [y/n]: \e[0m"
read answer
if [ "$answer" == "y" ]; then
  printf "   Thanks for confirming. Continuing...\n\n"
elif [ "$answer" == "Y" ]; then
  printf "   Thanks for confirming. Continuing...\n\n"
else
  printf "   \e[1;31mERROR: Please correct in cplusutilities/Download.sh. \e[0m\n\n"
  exit
fi

#----INITIALIZING----#
#When arguments are not given, user should provide them while macro is started
#Checking argument 1, trainname
if [ -z "$1" ]; then
  printf "Please enter train name: "
  read trainname
  printf "  Will download \e[1m$outputfile.root\e[0m output from train: \e[1m$trainname\e[0m \n\n"
else
  trainname=$1
  printf "Will download \e[1m$outputfile.root\e[0m output from train: \e[1m$trainname\e[0m \n\n"
fi

#Checking argument 2, output directoy
if [ -z "$2" ]; then
  printf "Please enter output directory: "
  read placetosave
  printf "  Output will be saved in: \e[1m$placetosave\e[0m \n\n"
else
  placetosave=$2
  printf "\nOutput will be saved in: \e[1m$placetosave\e[0m \n\n"
fi

#Checking argument 3, GRID merging stage
if [ -z "$3" ] && [ -z "$stage" ] ; then
  printf "No GRID merging stage was entered. I will download non-merged files\n"
elif [ -n "$stage" ] ; then
  printf "I will download files from GRID merging: \e[1m$stage\e[0m    (if not in format Stage_#, download will fail)\n\n"
else
  stage=$3
  printf "I will download files from GRID merging: \e[1m$stage\e[0m    (if not in format Stage_#, download will fail)\n\n"
fi

#Copy settings of this train
printf "\nWill now download \e[1m'env.sh'\e[0m using JAliEn to export the train configuration from HF_TreeCreator for train \e[1m$trainname\e[0m \n\n"
cmd=$(printf "cp -T 32 /alice/cern.ch/user/a/alitrain/PWGHF/HF_TreeCreator/%s/env.sh file:%s/HF_TreeCreator_env.sh\n" $trainname $PWD)
/opt/jalien/src/jalien/jalien << EOF
$cmd
exit
EOF

if [ ! -e "HF_TreeCreator_env.sh" ]; then
  printf "\e[1;31mERROR: Downloading env.sh failed, trying again for child_2\e[0m\n"
  cmd=$(printf "cp -T 32 /alice/cern.ch/user/a/alitrain/PWGHF/HF_TreeCreator/%s_child_2/env.sh file:%s/HF_TreeCreator_env.sh\n" $trainname $PWD)
  /opt/jalien/src/jalien/jalien << EOF2
  $cmd
  exit
EOF2
fi

source HF_TreeCreator_env.sh
dataset=$PERIOD_NAME

#Hardcoded information about dataset
splitchildsdifferentpaths=0
if [ "$dataset" == "LHC17pq_woSDD" ] || [  "$dataset" == "LHC17pq_pass1" ]; then
  #Data: pp 5 TeV
  inputpaths=(/alice/data/2017/LHC17p/000282341/pass1_FAST/PWGHF/HF_TreeCreator
              /alice/data/2017/LHC17q/000282366/pass1_FAST/PWGHF/HF_TreeCreator
              /alice/data/2017/LHC17p/000282341/pass1_CENT_woSDD/PWGHF/HF_TreeCreator
              /alice/data/2017/LHC17q/000282366/pass1_CENT_woSDD/PWGHF/HF_TreeCreator)
  datasetwithchilds=1
  dataset_short="pp5_2017_data"
elif [ "$dataset" == "LHC18a4a2" ]; then
  #D2H MC: pp 5 TeV
  inputpaths=(/alice/sim/2018/LHC18a4a2_fast/282341/PWGHF/HF_TreeCreator
              /alice/sim/2018/LHC18a4a2_fast/282366/PWGHF/HF_TreeCreator
              /alice/sim/2018/LHC18a4a2_cent/282341/PWGHF/HF_TreeCreator
              /alice/sim/2018/LHC18a4a2_cent/282366/PWGHF/HF_TreeCreator)
  datasetwithchilds=1
  dataset_short="pp5_2017_mc_prodD2H"
elif [ "$dataset" == "LHC2018_pp" ] || [  "$dataset" == "LHC2018_AOD208_bdefghijklmnop_13TeV_pp" ]; then
  #Data: pp 13 TeV 2018
  inputpaths=(/alice/data/2018/LHC18b/000285064/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18d/000286313/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18e/000286653/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18f/000287784/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18g/000288750/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18h/000288806/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18i/000288908/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18j/000288943/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18k/000289177/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18l/000289444/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18m/000292460/pass1_withTRDtracking/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18n/000293359/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18o/000293741/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18p/000294925/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18m/000291397/pass1_withTRDtracking/AOD208/PWGHF/HF_TreeCreator)
  childdownloadpath=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 11)
  datasetwithchilds=1
  dataset_short="pp_2018_data"
elif [ "$dataset" == "LHC2016_kl_pp" ] || [  "$dataset" == "LHC2016_AOD208_kl_13TeV_pp" ]; then
  #Data: pp 13 TeV 2016 kl
  inputpaths=(/alice/data/2016/LHC16k/000257630/pass2/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2016/LHC16l/000259164/pass2/AOD208/PWGHF/HF_TreeCreator)
  datasetwithchilds=1
  dataset_short="pp_2016_data"
elif [ "$dataset" == "LHC2016_de_pp" ] || [  "$dataset" == "LHC2016_AOD208_deghjop_13TeV_pp" ]; then
  #Data: pp 13 TeV 2016 deghjop
  inputpaths=(/alice/data/2016/LHC16d/000252371/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2016/LHC16e/000253563/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2016/LHC16g/000254331/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2016/LHC16h/000255440/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2016/LHC16j/000256307/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2016/LHC16o/000262849/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2016/LHC16p/000264109/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2016/LHC16h/000255180/pass1/AOD208/PWGHF/HF_TreeCreator)
  childdownloadpath=(1 2 3 4 5 6 7 4)
  datasetwithchilds=1
  dataset_short="pp_2016_data"
elif [  "$dataset" == "LHC2016_AOD208_deghjklop_13TeV_pp" ]; then
  #Data: pp 13 TeV 2016 deghjklop
  printf "\e[1;31m  Warning: New dataset, hardcoded paths not yet tested.\e[0m\n\n"
  inputpaths=(/alice/data/2016/LHC16d/000252371/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2016/LHC16e/000253563/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2016/LHC16g/000254331/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2016/LHC16h/000255440/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2016/LHC16j/000256307/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2016/LHC16k/000257630/pass2/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2016/LHC16l/000259164/pass2/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2016/LHC16o/000262849/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2016/LHC16p/000264109/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2016/LHC16h/000254653/pass1/AOD208/PWGHF/HF_TreeCreator)
  childdownloadpath=(1 2 3 4 5 6 7 8 9 4)
  datasetwithchilds=1
  dataset_short="pp_2016_data"
elif [ "$dataset" == "LHC2017_pp" ] || [  "$dataset" == "LHC2017_AOD208_cefhijklmor_13TeV_pp" ]; then
  #Data: pp 13 TeV 2017
  inputpaths=(/alice/data/2017/LHC17e/000270824/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2017/LHC17f/000270861/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2017/LHC17h/000272123/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2017/LHC17i/000274329/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2017/LHC17j/000274653/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2017/LHC17k/000274978/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2017/LHC17l/000277117/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2017/LHC17m/000279830/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2017/LHC17r/000282573/pass1/AOD208/PWGHF/HF_TreeCreator
              /alice/data/2017/LHC17o/000281243/pass1/AOD208/PWGHF/HF_TreeCreator)
  childdownloadpath=(2 3 4 5 6 7 8 9 10 11)
  datasetwithchilds=1
  dataset_short="pp_2017_data"
elif [ "$dataset" == "LHC2016_pass1_MC_pp" ] || [  "$dataset" == "LHC17h8a_MC_pp" ]; then
  #D2H MC: pp 13 TeV 2016 pass1
  inputpaths=(/alice/sim/2017/LHC17h8a/253529/PWGHF/HF_TreeCreator)
  datasetwithchilds=0
  dataset_short="pp_2016_mc_prodD2H"
elif [ "$dataset" == "LHC2016_pass2_MC_pp" ] || [  "$dataset" == "LHC18f4a_MC_pp" ]; then
  #D2H MC: pp 13 TeV 2016 pass2
  inputpaths=(/alice/sim/2018/LHC18f4a/257630/PWGHF/HF_TreeCreator)
  datasetwithchilds=0
  dataset_short="pp_2016_mc_prodD2H"
elif [ "$dataset" == "LHC2017_MC_pp" ] || [  "$dataset" == "LHC18l4a_MC_pp" ]; then
  #D2H MC: pp 13 TeV 2017
  inputpaths=(/alice/sim/2018/LHC18l4a/272123/PWGHF/HF_TreeCreator)
  datasetwithchilds=0
  dataset_short="pp_2017_mc_prodD2H"
elif [ "$dataset" == "LHC2018_MC_pp" ] || [  "$dataset" == "LHC18l4b_MC_pp" ]; then
  #D2H MC: pp 13 TeV 2018
  inputpaths=(/alice/sim/2018/LHC18l4b/285064/PWGHF/HF_TreeCreator)
  datasetwithchilds=0
  dataset_short="pp_2018_mc_prodD2H"
elif [ "$dataset" == "LHC17j4_adeghjop_MC_pp" ]; then
  #Lc->pKpi MC: pp 13 TeV 2016 pass1
  inputpaths=(/alice/sim/2017/LHC17j4a/253529/PWGHF/HF_TreeCreator)
  datasetwithchilds=0
  dataset_short="pp_2016_mc_prodLcpKpi"
elif [ "$dataset" == "LHC17j4_kl_MC_pp" ]; then
  #Lc->pKpi MC: pp 13 TeV 2016 pass1 (data = pass2!)
  inputpaths=(/alice/sim/2017/LHC17j4a/257630/PWGHF/HF_TreeCreator)
  datasetwithchilds=0
  dataset_short="pp_2016_mc_prodLcpKpi"
elif [ "$dataset" == "LHC17j4b_MC_pp" ]; then
  #Lc->pK0s MC: pp 13 TeV 2016 childs 5 and 6 correspond are pass1 but corresponding data is pass2
  inputpaths=(/alice/sim/2017/LHC17j4b/252371/PWGHF/HF_TreeCreator
              /alice/sim/2017/LHC17j4b/253529/PWGHF/HF_TreeCreator
              /alice/sim/2017/LHC17j4b/254331/PWGHF/HF_TreeCreator
              /alice/sim/2017/LHC17j4b/254651/PWGHF/HF_TreeCreator
              /alice/sim/2017/LHC17j4b/257630/PWGHF/HF_TreeCreator
              /alice/sim/2017/LHC17j4b/259164/PWGHF/HF_TreeCreator
              /alice/sim/2017/LHC17j4b/256307/PWGHF/HF_TreeCreator
              /alice/sim/2017/LHC17j4b/263741/PWGHF/HF_TreeCreator
              /alice/sim/2017/LHC17j4b/264076/PWGHF/HF_TreeCreator)
  datasetwithchilds=1
  dataset_short="pp_2016_mc_prodLcpK0s"
elif [ "$dataset" == "MCpp13TeV_MB_all" ]; then
  #D2H and dedicated MB MC's for 2016-17-18. See short names below/LEGO train page for order of childs.
  printf "\e[1;31m  Warning: New dataset, hardcoded paths not yet tested.\e[0m\n\n"
  inputpaths=(/alice/sim/2017/LHC17h8a/253529/PWGHF/HF_TreeCreator
              /alice/sim/2018/LHC18f4a/257630/PWGHF/HF_TreeCreator
              /alice/sim/2018/LHC18l4a/272123/PWGHF/HF_TreeCreator
              /alice/sim/2018/LHC18l4b/288908/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h4c1/253529/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h4b1/272123/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h4a1/288908/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h4c2/253529/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h4b2/272123/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h4a2/288908/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h4c3/253529/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h4b3/272123/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h4a3/288908/PWGHF/HF_TreeCreator)
  datasetwithchilds=1
  splitchildsdifferentpaths=1
  dataset_short_arr=("pp_2016_mc_prodD2H"
                     "pp_2016_mc_prodD2H"
                     "pp_2017_mc_prodD2H"
                     "pp_2018_mc_prodD2H"
                     "pp_2016_mc_prodLcpKpi"
                     "pp_2017_mc_prodLcpKpi"
                     "pp_2018_mc_prodLcpKpi"
                     "pp_2016_mc_prodLcpK0s"
                     "pp_2017_mc_prodLcpK0s"
                     "pp_2018_mc_prodLcpK0s"
                     "pp_2016_mc_prodDs"
                     "pp_2017_mc_prodDs"
                     "pp_2018_mc_prodDs")
elif [ "$dataset" == "MCpp13TeV_HM_all" ]; then
  #D2H and dedicated HM MC's for 2016-17-18 (V0M) and 18 (SPD). See short names below/LEGO train page for order of childs.
  printf "\e[1;31m  Warning: New dataset, hardcoded paths not yet tested.\e[0m\n\n"
  inputpaths=(/alice/sim/2019/LHC19h5c/258454/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h5b/277117/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h5a/286653/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h5a2/294925/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h10c1/258454/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h10b1/277117/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h10a1/286653/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h10a1b/294925/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h10c2/258454/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h10b2/277117/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h10a2/286653/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h10a2b/294925/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h10c3/258454/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h10b3/277117/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h10a3/286653/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19h10a3b/294925/PWGHF/HF_TreeCreator)
  datasetwithchilds=1
  splitchildsdifferentpaths=1
  dataset_short_arr=("pp_2016_mc_prodHMV0MD2H"
                     "pp_2017_mc_prodHMV0MD2H"
                     "pp_2018_mc_prodHMV0MD2H"
                     "pp_2018_mc_prodHMSPDD2H"
                     "pp_2016_mc_prodHMV0MLcpKpi"
                     "pp_2017_mc_prodHMV0MLcpKpi"
                     "pp_2018_mc_prodHMV0MLcpKpi"
                     "pp_2018_mc_prodHMSPDLcpKpi"
                     "pp_2016_mc_prodHMV0MLcpK0s"
                     "pp_2017_mc_prodHMV0MLcpK0s"
                     "pp_2018_mc_prodHMV0MLcpK0s"
                     "pp_2018_mc_prodHMSPDLcpK0s"
                     "pp_2016_mc_prodHMV0MDs"
                     "pp_2017_mc_prodHMV0MDs"
                     "pp_2018_mc_prodHMV0MDs"
                     "pp_2018_mc_prodHMSPDDs")
elif [ "$dataset" == "LHC17j4d2" ]; then
  #Lc->pK0s MC: pPb 5 TeV (used for pp for now)
  inputpaths=(/alice/sim/2017/LHC17j4d2_fast/265343/PWGHF/HF_TreeCreator
              /alice/sim/2017/LHC17j4d2_fast/267163/PWGHF/HF_TreeCreator
              /alice/sim/2017/LHC17j4d2_cent_wSDD/265343/PWGHF/HF_TreeCreator
              /alice/sim/2017/LHC17j4d2_cent_wSDD/267163/PWGHF/HF_TreeCreator)
  datasetwithchilds=1
  dataset_short="pPb_2016_mc_prodLcpK0s"
elif [ "$dataset" == "LHC19c2b_all_q" ]; then
  #Lc->pK0s MC: PbPb 5 TeV 2018 30-50
  inputpaths=(/alice/sim/2019/LHC19c2b/296433/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19c2b2/296433/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19c2b_extra/296433/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19c2b2_extra/296433/PWGHF/HF_TreeCreator)
  datasetwithchilds=1
  dataset_short="PbPb3050_2018_mc_prodLcpK0s"
elif [ "$dataset" == "LHC19c2b_all_r" ]; then
  #Lc->pK0s MC: PbPb 5 TeV 2018 30-50
  inputpaths=(/alice/sim/2019/LHC19c2b/297481/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19c2b2/297481/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19c2b_extra/297481/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19c2b2_extra/297481/PWGHF/HF_TreeCreator)
  datasetwithchilds=1
  dataset_short="PbPb3050_2018_mc_prodLcpK0s"
elif [ "$dataset" == "LHC19c2a_all_q" ]; then
  #Lc->pK0s MC: PbPb 5 TeV 2018 0-10
  inputpaths=(/alice/sim/2019/LHC19c2a/296433/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19c2a2/296244/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19c2a_extra/296433/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19c2a2_extra/296244/PWGHF/HF_TreeCreator)
  datasetwithchilds=1
  dataset_short="PbPb010_2018_mc_prodLcpK0s"
elif [ "$dataset" == "LHC19c2a_all_r" ]; then
  #Lc->pK0s MC: PbPb 5 TeV 2018 0-10
  inputpaths=(/alice/sim/2019/LHC19c2a/297481/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19c2a2/297481/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19c2a_extra/297481/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19c2a2_extra/297481/PWGHF/HF_TreeCreator)
  datasetwithchilds=1
  dataset_short="PbPb010_2018_mc_prodLcpK0s"
elif [ "$dataset" == "LHC18l8b2_q" ]; then
  #GP MC: PbPb 5 TeV 2018 0-10
  inputpaths=(/alice/sim/2018/LHC18l8b2/296433/PWGHF/HF_TreeCreator)
  datasetwithchilds=0
  dataset_short="PbPb010_2018_mc_prodGP"
elif [ "$dataset" == "LHC18l8b2_r" ]; then
  #GP MC: PbPb 5 TeV 2018 0-10
  inputpaths=(/alice/sim/2018/LHC18l8b2/297481/PWGHF/HF_TreeCreator)
  datasetwithchilds=0
  dataset_short="PbPb010_2018_mc_prodGP"
elif [ "$dataset" == "LHC18l8c2_q" ]; then
  #GP MC: PbPb 5 TeV 2018 30-50
  inputpaths=(/alice/sim/2018/LHC18l8c2/296433/PWGHF/HF_TreeCreator)
  datasetwithchilds=0
  dataset_short="PbPb3050_2018_mc_prodGP"
elif [ "$dataset" == "LHC18l8c2_r" ]; then
  #GP MC: PbPb 5 TeV 2018 30-50
  inputpaths=(/alice/sim/2018/LHC18l8c2/297481/PWGHF/HF_TreeCreator)
  datasetwithchilds=0
  dataset_short="PbPb3050_2018_mc_prodGP"
elif [ "$dataset" == "LHC18r_pass1" ]; then
  #Data: PbPb 5 TeV 2018
  inputpaths=(/alice/data/2018/LHC18r/000296749/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18r/000296750/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18r/000296785/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18r/000296848/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18r/000296849/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18r/000296932/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18r/000296966/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18r/000297029/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18r/000297123/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18r/000297196/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18r/000297219/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18r/000297332/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18r/000297379/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18r/000297415/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18r/000297451/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18r/000297481/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18r/000297542/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18r/000297595/pass1/PWGHF/HF_TreeCreator)
  datasetwithchilds=0
  dataset_short="PbPb_2018_data"
elif [ "$dataset" == "LHC18q_pass1" ]; then
  #Data: PbPb 5 TeV 2018
  inputpaths=(/alice/data/2018/LHC18q/000296623/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18q/000296549/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18q/000296509/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18q/000296415/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18q/000296379/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18q/000296309/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18q/000296273/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18q/000296244/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18q/000296194/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18q/000296133/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18q/000295913/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18q/000295822/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18q/000295753/pass1/PWGHF/HF_TreeCreator
              /alice/data/2018/LHC18q/000295725/pass1/PWGHF/HF_TreeCreator)
  datasetwithchilds=0
  dataset_short="PbPb_2018_data"
elif [ "$dataset" == "LHC19c3a_all_q" ]; then
  #D2H MC: PbPb 5 TeV 2018 0-10
  inputpaths=(/alice/sim/2019/LHC19c3a/296433/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19c3a2/296433/PWGHF/HF_TreeCreator)
  datasetwithchilds=1
  dataset_short="PbPb010_2018_mc_prodD2H"
elif [ "$dataset" == "LHC19c3a_all_r" ]; then
  #D2H MC: PbPb 5 TeV 2018 0-10
  inputpaths=(/alice/sim/2019/LHC19c3a/297481/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19c3a2/297481/PWGHF/HF_TreeCreator)
  datasetwithchilds=1
  dataset_short="PbPb010_2018_mc_prodD2H"
elif [ "$dataset" == "LHC19c3b_all_q" ]; then
  #D2H MC: PbPb 5 TeV 2018 30-50
  inputpaths=(/alice/sim/2019/LHC19c3b/296433/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19c3b2/296433/PWGHF/HF_TreeCreator)
  datasetwithchilds=1
  dataset_short="PbPb3050_2018_mc_prodD2H"
elif [ "$dataset" == "LHC19c3b_all_r" ]; then
  #D2H MC: PbPb 5 TeV 2018 30-50
  inputpaths=(/alice/sim/2019/LHC19c3b/297481/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19c3b2/297481/PWGHF/HF_TreeCreator)
  datasetwithchilds=1
  dataset_short="PbPb3050_2018_mc_prodD2H"
elif [ "$dataset" == "LHC19d4a_q" ]; then
  #Ds->KKpi MC: PbPb 5 TeV 2018 0-10
  inputpaths=(/alice/sim/2019/LHC19d4a/296433/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19d4a2/296244/PWGHF/HF_TreeCreator)
  datasetwithchilds=1
  dataset_short="PbPb010_2018_mc_prodDs"
elif [ "$dataset" == "LHC19d4a_r" ]; then
  #Ds->KKpi MC: PbPb 5 TeV 2018 0-10
  inputpaths=(/alice/sim/2019/LHC19d4a/297481/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19d4a2/297481/PWGHF/HF_TreeCreator)
  datasetwithchilds=1
  dataset_short="PbPb010_2018_mc_prodDs"
elif [ "$dataset" == "LHC19d4b_q" ]; then
  #Ds->KKpi MC: PbPb 5 TeV 2018 30-50
  inputpaths=(/alice/sim/2019/LHC19d4b/296433/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19d4b2/296433/PWGHF/HF_TreeCreator)
  datasetwithchilds=1
  dataset_short="PbPb3050_2018_mc_prodDs"
elif [ "$dataset" == "LHC19d4b_r" ]; then
  #Ds->KKpi MC: PbPb 5 TeV 2018 30-50
  inputpaths=(/alice/sim/2019/LHC19d4b/297481/PWGHF/HF_TreeCreator
              /alice/sim/2019/LHC19d4b2/297481/PWGHF/HF_TreeCreator)
  datasetwithchilds=1
  dataset_short="PbPb3050_2018_mc_prodDs"
elif [ "$dataset" == "LHC13d19" ]; then
  #MC: PbPb 5 TeV ITS2 upgrade
  inputpaths=(/alice/sim/2013/LHC13d19/138275/PWGHF/HF_TreeCreator)
  datasetwithchilds=0
  dataset_short="ITS2_13d19"
elif [ "$dataset" == "LHC14j5_new" ]; then
  #MC: PbPb 5 TeV ITS2 upgrade (new)
  inputpaths=(/alice/sim/2014/LHC14j5_new/138364/PWGHF/HF_TreeCreator
              /alice/sim/2014/LHC14j5_new/137844/PWGHF/HF_TreeCreator
              /alice/sim/2014/LHC14j5_new/137686/PWGHF/HF_TreeCreator
              /alice/sim/2014/LHC14j5_new/137541/PWGHF/HF_TreeCreator
              /alice/sim/2014/LHC14j5_new/138275/PWGHF/HF_TreeCreator
              /alice/sim/2014/LHC14j5_new/137608/PWGHF/HF_TreeCreator
              /alice/sim/2014/LHC14j5_new/138396/PWGHF/HF_TreeCreator
              /alice/sim/2014/LHC14j5_new/138225/PWGHF/HF_TreeCreator
              /alice/sim/2014/LHC14j5_new/137162/PWGHF/HF_TreeCreator
              /alice/sim/2014/LHC14j5_new/138197/PWGHF/HF_TreeCreator)
  datasetwithchilds=0
  dataset_short="ITS2_14j5new"
elif [ "$dataset" == "LHC15o" ]; then
  #Data: PbPb 5.02 TeV, all passes together (2015)
  inputpaths=(/alice/data/2015/LHC15o/000246222/pass1/AOD194/PWGHF/HF_TreeCreator
              /alice/data/2015/LHC15o/000245454/pass1_pidfix/AOD194/PWGHF/HF_TreeCreator
              /alice/data/2015/LHC15o/000245064/pass3_lowIR_pidfix/AOD194/PWGHF/HF_TreeCreator
              /alice/data/2015/LHC15o/000245829/pass1/AOD194/PWGHF/HF_TreeCreator
              /alice/data/2015/LHC15o/000246804/pass1/AOD194/PWGHF/HF_TreeCreator
              /alice/data/2015/LHC15o/000246087/pass1/AOD194/PWGHF/HF_TreeCreator
              /alice/data/2015/LHC15o/000246042/pass1/AOD194/PWGHF/HF_TreeCreator
              /alice/data/2015/LHC15o/000246272/pass1/AOD194/PWGHF/HF_TreeCreator
              /alice/data/2015/LHC15o/000246805/pass1/AOD194/PWGHF/HF_TreeCreator
              /alice/data/2015/LHC15o/000245439/pass1_pidfix/AOD194/PWGHF/HF_TreeCreator
              /alice/data/2015/LHC15o/000245349/pass1_pidfix/AOD194/PWGHF/HF_TreeCreator
              /alice/data/2015/LHC15o/000245407/pass1_pidfix/AOD194/PWGHF/HF_TreeCreator
              /alice/data/2015/LHC15o/000245501/pass1_pidfix/AOD194/PWGHF/HF_TreeCreator)
  childdownloadpath=(1 2 3 1 1 1 1 1 1 2 2 2 2)
  datasetwithchilds=1
  dataset_short="LHC15o"
elif [ "$dataset" == "LHC19h1b2" ]; then
  #MC ITS2 Upgrade 2019: PbPb, heavily enriched with HF (charm and beauty signal)
  inputpaths=(/alice/sim/2019/LHC19h1b2/280235/PWGHF/HF_TreeCreator)
  datasetwithchilds=0
  dataset_short="ITS2_19h1b2"
else
  printf "\e[1;31mError: Dataset not yet implemented. Returning...\e[0m\n\n"
  exit
fi

printf "\n\n\e[1mFor debugging hardcoded GRID paths (might change over time)\e[0m\n"
printf " From train config (not all, some childs can be splitted and not show up here)\n"
if [ -n "$ALIEN_JDL_child_1_OUTPUTDIR" ]; then printf "   1: $ALIEN_JDL_child_1_OUTPUTDIR/PWGHF/HF_TreeCreator\n"; fi
if [ -n "$ALIEN_JDL_child_2_OUTPUTDIR" ]; then printf "   2: $ALIEN_JDL_child_2_OUTPUTDIR/PWGHF/HF_TreeCreator\n"; fi
if [ -n "$ALIEN_JDL_child_3_OUTPUTDIR" ]; then printf "   3: $ALIEN_JDL_child_3_OUTPUTDIR/PWGHF/HF_TreeCreator\n"; fi
if [ -n "$ALIEN_JDL_child_4_OUTPUTDIR" ]; then printf "   4: $ALIEN_JDL_child_4_OUTPUTDIR/PWGHF/HF_TreeCreator\n"; fi
if [ -n "$ALIEN_JDL_child_5_OUTPUTDIR" ]; then printf "   5: $ALIEN_JDL_child_5_OUTPUTDIR/PWGHF/HF_TreeCreator\n"; fi
if [ -n "$ALIEN_JDL_child_6_OUTPUTDIR" ]; then printf "   6: $ALIEN_JDL_child_6_OUTPUTDIR/PWGHF/HF_TreeCreator\n"; fi
if [ -n "$ALIEN_JDL_child_7_OUTPUTDIR" ]; then printf "   7: $ALIEN_JDL_child_7_OUTPUTDIR/PWGHF/HF_TreeCreator\n"; fi
if [ -n "$ALIEN_JDL_child_8_OUTPUTDIR" ]; then printf "   8: $ALIEN_JDL_child_8_OUTPUTDIR/PWGHF/HF_TreeCreator\n"; fi
if [ -n "$ALIEN_JDL_child_9_OUTPUTDIR" ]; then printf "   9: $ALIEN_JDL_child_9_OUTPUTDIR/PWGHF/HF_TreeCreator\n"; fi
if [ -n "$ALIEN_JDL_child_10_OUTPUTDIR" ]; then printf "   10: $ALIEN_JDL_child_10_OUTPUTDIR/PWGHF/HF_TreeCreator\n"; fi
if [ -n "$ALIEN_JDL_child_11_OUTPUTDIR" ]; then printf "   11: $ALIEN_JDL_child_11_OUTPUTDIR/PWGHF/HF_TreeCreator\n"; fi
if [ -n "$ALIEN_JDL_child_12_OUTPUTDIR" ]; then printf "   12: $ALIEN_JDL_child_12_OUTPUTDIR/PWGHF/HF_TreeCreator\n"; fi
if [ -n "$ALIEN_JDL_child_13_OUTPUTDIR" ]; then printf "   13: $ALIEN_JDL_child_13_OUTPUTDIR/PWGHF/HF_TreeCreator\n"; fi
if [ -n "$ALIEN_JDL_child_14_OUTPUTDIR" ]; then printf "   14: $ALIEN_JDL_child_14_OUTPUTDIR/PWGHF/HF_TreeCreator\n"; fi
if [ -z "$ALIEN_JDL_child_1_OUTPUTDIR" ]; then printf "   1: $ALIEN_JDL_OUTPUTDIR/PWGHF/HF_TreeCreator\n"; fi
printf " Hardcoded in Download.sh script\n"
for input_index in ${!inputpaths[*]}
do
  childX=$(($input_index+1))
  printf "   ${childX}: ${inputpaths[$input_index]}\n"
done
printf "\n\n"

TAG=${ALIROOT_VERSION:21:20}

#Creating output directory, and checking if user has writing permission. + dir with used TAG, so it is easier to see which production should be used/can be deleted
mkdir -p -m 777 $placetosave/$TAG
if [ $? -ne 0 ]; then
  printf "\n\e[1;31mError: Could not create output directory. Is $placetosave writable? Returning... \e[0m\n\n"
  exit
else
  printf "Created directory: \e[1m$placetosave/$TAG\e[0m \n"
fi
placetosave=$placetosave/$TAG

for input_index in ${!inputpaths[*]}
do
  if [ $splitchildsdifferentpaths -eq 1 ]; then
    dataset_short=${dataset_short_arr[$input_index]}
  fi
  placetosavearr[$input_index]=$placetosave/$dataset_short
done

for input_index in ${!inputpaths[*]}
do
  placetosave=${placetosavearr[$input_index]}

  #Adding short description of dataset to path (set above)
  mkdir -p -m 777 $placetosave
  if [ $? -ne 0 ]; then
    printf "\n\e[1;31mError: Could not create output directory. Is $placetosave writable? Returning... \e[0m\n\n"
    exit
  else
    printf "Created directory: \e[1m$placetosave\e[0m \n"
  fi

  #Adding train ID to path
  mkdir -p -m 777 $placetosave/$trainname
  if [ $? -ne 0 ]; then
    printf "\n\e[1;31mError: Could not create output directory. Is $placetosave writable? Returning... \e[0m\n\n"
    exit
  else
     printf "Created directory: \e[1m$placetosave/$trainname\e[0m \n"
  fi
  mkdir -p -m 777 $placetosave/$trainname/unmerged
  if [ $? -ne 0 ]; then
    printf "\n\e[1;31mError: Could not create output directory. Is $placetosave/$trainname writable? Returning... \e[0m\n\n"
    exit
  else
     printf "Created directory: \e[1m$placetosave/$trainname/unmerged\e[0m \n"
  fi

  if [ $splitchildsdifferentpaths -eq 0 ]; then
    break
  fi
done

#Log files ('D' is for download) + trainID, date, and timestamp
datestamp="$(date +"%d-%m-%Y")"
timestamp="$(date +"%H-%M-%S")"
if [ -z "$4" ]; then
  stdoutputfile=$(printf "D_stdout_%s_%s-%s.txt" $trainname $datestamp $timestamp)
  stderrorfile=$(printf "D_stderr_%s_%s-%s.txt" $trainname $datestamp $timestamp)
else
  stdoutputfile=$(printf "D_stdout_%s_%s_%s-%s.txt" $trainname $stage $datestamp $timestamp)
  stderrorfile=$(printf "D_stderr_%s_%s_%s-%s.txt" $trainname $stage $datestamp $timestamp)
fi

#----RUNNING THE DOWNLOADER----#
printf "\n\n\e[1m----RUNNING THE DOWNLOADER----\e[0m\n\n"
printf "  Output of downloaders stored in:            \e[1m%s\e[0m\n  Warnings/Errors of downloader stored in:    \e[1m%s\e[0m\n" $i $stdoutputfile $stderrorfile
rundownloader="sh ../cplusutilities/downloader.sh"

printf "\n\n\n\nOutput downloading starts here\n\n" > "$stdoutputfile"
printf "\n\n\n\nErrors downloading starts here\n\n" > "$stderrorfile"

for input_index in ${!inputpaths[*]}
do
  ithinput=${childdownloadpath[$input_index]}
  if [ -z "$ithinput" ]; then
    ithinput=$(($input_index+1))
  fi
  localchild=$(($input_index+1))
  placetosave=${placetosavearr[$input_index]}

  sh ../cplusutilities/run_downloader $rundownloader ${inputpaths[$input_index]} $ithinput "$nfiles" $outputfile $placetosave $trainname $datasetwithchilds $localchild $stage >> "$stdoutputfile" 2>> "$stderrorfile"
done


for input_index in ${!inputpaths[*]}
do
  placetosave=${placetosavearr[$input_index]}
  #give all permissions to all directories downloaded from the GRID
  chmod -R 777 $placetosave/$trainname/unmerged/
  if [ $splitchildsdifferentpaths -eq 0 ]; then
    break
  fi
done

#Check logs for the comman 'jalien command not found' error. If this is the case, no files where downloaded.
if grep -q "jalien\|command not found" "$stderrorfile"
then
  printf "\e[1;31m  Warning: The 'jalien' command was not found, so no new files where downloaded. Did you already connect to JAliEn? Check log if this was not intended!\e[0m\n\n"
fi
#Check logs for the comman 'JBox agent could not be started' error. If this is the case, no files where downloaded.
if grep -q "JBox agent\|could not be started" "$stderrorfile"
then
printf "\e[1;31m  Warning: The 'JBox agent' could not be started, so no new files where downloaded. Did you already connect to JAliEn? Check log if this was not intended!\e[0m\n\n"
fi

rm HF_TreeCreator_env.sh
source clean_HF_TreeCreator_env.sh

for input_index in ${!inputpaths[*]}
do
  placetosave=${placetosavearr[$input_index]}
  #Saving log files in output directory
  cp $stdoutputfile $placetosave/$trainname/
  cp $stderrorfile $placetosave/$trainname/
  printf "\e[1mMoved log files to $placetosave/$trainname/\e[0m\n"
  printf "\e[1m----DOWNLOADER FINISHED----\e[0m\n\n"
  if [ $splitchildsdifferentpaths -eq 0 ]; then
    break
  fi
done
rm $stdoutputfile
rm $stderrorfile

printf "\n\e[1m<<<Ready downloading? Please kill JAliEn daemons>>>\e[0m\n"
printf "  killall java\n"
printf "\e[1m<<<And remove alien logs if you like>>>\e[0m\n"
printf "  rm alien-config* alien-fine* alien-info* alien-severe* alien-warning*\n"
printf "  rm ../cplusutilities/alien-config* ../cplusutilities/alien-fine* ../cplusutilities/alien-info* ../cplusutilities/alien-severe* ../cplusutilities/alien-warning*\n"

